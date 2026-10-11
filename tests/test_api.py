"""Tests for the HTTP API.

Every test goes through a real server on a loopback port and a real HTTP
client, so routing, status codes, headers and JSON bodies are all verified as
the browser (or curl) would see them.

Translation is only covered on its failure paths: a successful
``POST /translate`` would load the 1.8 GB M2M100 model, which is not something
a unit test should trigger.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from captioner import api, automation, config, db, worker


@pytest.fixture
def data_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    return tmp_path


@pytest.fixture
def server(data_paths):
    httpd = api.create_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def secured(data_paths):
    httpd = api.create_server(host="127.0.0.1", port=0, token="hunter2")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def call(base, path, method="GET", body=None, token=None, headers=None):
    """One HTTP request; returns (status, parsed_json_or_text)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read().decode("utf-8")
            status, content_type = response.status, response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        status, content_type = exc.code, exc.headers.get("Content-Type", "")
    try:
        return status, content_type, json.loads(raw)
    except ValueError:
        return status, content_type, raw


def call_raw(base, path, method="GET"):
    """Like ``call``, but for binary bodies: returns (status, type, bytes)."""
    request = urllib.request.Request(base + path, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return (response.status, response.headers.get("Content-Type", ""),
                    response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


def make_media(tmp_path, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"media")
    return path


def queue_job(data_paths, name="clip.mp4", status=None, **options):
    """Insert a job straight into the queue, optionally forced to a status."""
    path = make_media(data_paths, name)
    conn = db.connect()
    try:
        job_id = worker.enqueue(conn, str(path), **options)
        if status:
            conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
            conn.commit()
        return job_id
    finally:
        conn.close()


#static UI and spec

def test_the_browser_ui_is_served(server):
    _httpd, base = server
    status, content_type, body = call(base, "/")
    assert status == 200
    assert content_type.startswith("text/html")
    assert "Captioner" in body and "/static/app.js" in body


def test_ui_assets_are_served_with_their_types(server):
    _httpd, base = server
    status, content_type, _body = call(base, "/static/app.js")
    assert status == 200 and "javascript" in content_type
    status, content_type, _body = call(base, "/static/style.css")
    assert status == 200 and content_type.startswith("text/css")


def test_the_page_links_a_favicon_and_it_is_the_app_icon(server):
    """The tab icon is captioner.ico itself, served without a second copy."""
    _httpd, base = server
    _status, _type, body = call(base, "/")
    assert 'href="/favicon.ico"' in body

    status, content_type, icon = call_raw(base, "/favicon.ico")
    assert status == 200
    assert content_type == "image/x-icon"
    assert icon.startswith(b"\x00\x00\x01\x00")  # the ICO magic number
    assert icon == (config.PROJECT_ROOT / "captioner.ico").read_bytes()


def test_ui_assets_cannot_escape_their_folder(server):
    _httpd, base = server
    for path in ("/static/../../.env", "/static/..%2f..%2f.env", "/static/nope.js"):
        status, _type, payload = call(base, path)
        assert status == 404, path
        assert "error" in payload


def test_openapi_describes_every_route(server):
    _httpd, base = server
    status, _type, spec = call(base, "/openapi.json")
    assert status == 200
    assert spec["openapi"].startswith("3.0")
    assert "/api/jobs" in spec["paths"]
    assert set(spec["paths"]["/api/jobs"]) == {"get", "post"}
    assert "parameters" in spec["paths"]["/api/jobs/{job_id}"]["get"]


#read endpoints

def test_health_reports_version_queue_and_engines(server, data_paths):
    _httpd, base = server
    queue_job(data_paths, "clip.mp4")
    status, _type, payload = call(base, "/api/health")
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["version"] == api.VERSION
    assert payload["engines"]["asr"]["selected"]
    assert payload["queue"].get("queued") == 1


def test_models_and_languages(server):
    _httpd, base = server
    _status, _type, models_payload = call(base, "/api/models")
    assert "small.en" in models_payload["batch"]
    assert "tiny.en" in models_payload["live"]
    _status, _type, languages = call(base, "/api/languages")
    assert languages["languages"]["hi"] == "Hindi"
    assert "te" not in languages["languages"]  # no Telugu in the M2M100 shortlist


def test_engines_lists_backends_and_selection(server):
    _httpd, base = server
    status, _type, payload = call(base, "/api/engines?probe=0")
    assert status == 200
    assert payload["asr"]["selected"] == "faster-whisper"
    assert set(payload["asr"]["backends"]) >= {"faster-whisper", "openai"}
    assert "available" not in payload["asr"]  # probing was switched off


def test_jobs_list_is_newest_first(server, data_paths):
    _httpd, base = server
    first = queue_job(data_paths, "one.mp4")
    second = queue_job(data_paths, "two.mp4")
    status, _type, payload = call(base, "/api/jobs")
    assert status == 200
    assert [job["id"] for job in payload["jobs"]] == [second, first]
    assert payload["jobs"][0]["file_name"] == "two.mp4"


def test_jobs_list_filters_and_validates_limit(server, data_paths):
    _httpd, base = server
    done = queue_job(data_paths, "done.mp4", status="done")
    queue_job(data_paths, "queued.mp4")
    _status, _type, payload = call(base, "/api/jobs?status=done")
    assert [job["id"] for job in payload["jobs"]] == [done]
    status, _type, payload = call(base, "/api/jobs?limit=abc")
    assert status == 400 and "limit" in payload["error"]["message"]


def test_job_detail_includes_caption_blocks(server, data_paths):
    _httpd, base = server
    job_id = queue_job(data_paths)
    conn = db.connect()
    conn.execute("INSERT INTO captions (job_id, seq, start_ms, end_ms, text) "
                 "VALUES (?, 1, 0, 1200, 'Hello there')", (job_id,))
    conn.commit()
    conn.close()
    status, _type, payload = call(base, f"/api/jobs/{job_id}")
    assert status == 200
    assert payload["job"]["caption_blocks"] == [
        {"seq": 1, "start_ms": 0, "end_ms": 1200, "text": "Hello there"}
    ]


def test_job_detail_rejects_bad_and_unknown_ids(server):
    _httpd, base = server
    status, _type, payload = call(base, "/api/jobs/abc")
    assert status == 400 and "must be a number" in payload["error"]["message"]
    status, _type, payload = call(base, "/api/jobs/424242")
    assert status == 404 and "no job" in payload["error"]["message"]


def test_captions_in_every_format(server, data_paths):
    _httpd, base = server
    job_id = queue_job(data_paths)
    conn = db.connect()
    conn.execute("INSERT INTO captions (job_id, seq, start_ms, end_ms, text) "
                 "VALUES (?, 1, 0, 1200, 'Hello there')", (job_id,))
    conn.commit()
    conn.close()
    out_dir = config.OUTPUT_DIR / f"job_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "captions.srt").write_text("1\n00:00:00,000 --> 00:00:01,200\nHello there\n",
                                          encoding="utf-8")

    _status, _type, payload = call(base, f"/api/jobs/{job_id}/captions?format=json")
    assert payload["captions"][0]["text"] == "Hello there"
    _status, _type, payload = call(base, f"/api/jobs/{job_id}/captions?format=text")
    assert payload["text"] == "Hello there"
    _status, _type, payload = call(base, f"/api/jobs/{job_id}/captions?format=srt")
    assert payload["text"].startswith("1\n00:00:00,000")
    status, _type, _payload = call(base, f"/api/jobs/{job_id}/captions?format=vtt")
    assert status == 404  # no vtt written for this job


#writes

def test_queue_a_file(server, data_paths):
    _httpd, base = server
    clip = make_media(data_paths, "clip.mp4")
    status, _type, payload = call(base, "/api/jobs", "POST",
                                  {"path": str(clip), "model": "small.en",
                                   "diarize": True, "translate_to": "hi"})
    assert status == 202
    job = payload["job"]
    assert job["status"] == "queued" and job["model"] == "small.en"
    assert job["options"] == {"diarize": True, "align": False, "translate_to": "hi"}
    _status, _type, listing = call(base, "/api/jobs")
    assert [entry["id"] for entry in listing["jobs"]] == [job["id"]]


def test_queue_rejects_bad_input(server, data_paths):
    _httpd, base = server
    status, _type, payload = call(base, "/api/jobs", "POST", {})
    assert status == 400 and "missing field" in payload["error"]["message"]

    status, _type, payload = call(base, "/api/jobs", "POST", {"path": str(data_paths / "ghost.mp4")})
    assert status == 404 and "no such file" in payload["error"]["message"]

    notes = data_paths / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    status, _type, payload = call(base, "/api/jobs", "POST", {"path": str(notes)})
    assert status == 400 and "unsupported media type" in payload["error"]["message"]

    clip = make_media(data_paths, "clip2.mp4")
    status, _type, payload = call(base, "/api/jobs", "POST",
                                  {"path": str(clip), "translate_to": "te"})
    assert status == 400 and "unknown target language" in payload["error"]["message"]


def test_queue_rejects_a_bad_body(server):
    _httpd, base = server
    request = urllib.request.Request(base + "/api/jobs", data=b"{not json",
                                     method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    assert status == 400


def test_retry_and_delete_semantics(server, data_paths):
    _httpd, base = server
    done = queue_job(data_paths, "done.mp4", status="done")
    running = queue_job(data_paths, "running.mp4", status="running")

    status, _type, payload = call(base, f"/api/jobs/{done}/retry", "POST", {})
    assert status == 200 and payload["job"]["status"] == "queued"
    status, _type, payload = call(base, f"/api/jobs/{running}/retry", "POST", {})
    assert status == 409 and "is running" in payload["error"]["message"]

    status, _type, payload = call(base, f"/api/jobs/{done}", "DELETE")
    assert status == 200 and payload["deleted"] == done
    status, _type, payload = call(base, f"/api/jobs/{running}", "DELETE")
    assert status == 409


def test_translate_validates_before_loading_a_model(server, data_paths):
    _httpd, base = server
    job_id = queue_job(data_paths, status="done")
    status, _type, payload = call(base, f"/api/jobs/{job_id}/translate", "POST",
                                  {"to": "te"})
    assert status == 400 and "unknown target language" in payload["error"]["message"]
    status, _type, payload = call(base, f"/api/jobs/{job_id}/translate", "POST",
                                  {"to": "hi"})
    assert status == 404 and "transcribe the job first" in payload["error"]["message"]


#protocol and security

def test_unknown_routes_and_methods(server):
    _httpd, base = server
    status, _type, payload = call(base, "/api/nothing")
    assert status == 404 and "no route" in payload["error"]["message"]
    status, _type, payload = call(base, "/api/health", "POST", {})
    assert status == 405 and payload["error"]["allowed"] == ["GET"]


def test_a_token_is_required_when_configured(secured):
    _httpd, base = secured
    status, _type, payload = call(base, "/api/jobs")
    assert status == 401 and "bearer token" in payload["error"]["message"]
    assert call(base, "/api/jobs", token="wrong")[0] == 401
    assert call(base, "/api/jobs", token="hunter2")[0] == 200
    assert call(base, "/api/jobs?token=hunter2")[0] == 200


def test_a_non_loopback_bind_requires_a_token(data_paths):
    with pytest.raises(SystemExit) as excinfo:
        api.create_server(host="0.0.0.0", port=0, token="")
    assert config.DB_PATH  # the DB was never touched: nothing was started
    assert "refusing to bind" in str(excinfo.value)


def test_cross_origin_writes_are_rejected(server, data_paths):
    _httpd, base = server
    clip = make_media(data_paths, "clip.mp4")
    status, _type, payload = call(base, "/api/jobs", "POST", {"path": str(clip)},
                                  headers={"Origin": "http://evil.example"})
    assert status == 403 and "cross-origin" in payload["error"]["message"]
    # The same request from the page itself is fine.
    status, _type, _payload = call(base, "/api/jobs", "POST", {"path": str(clip)},
                                   headers={"Origin": base})
    assert status == 202


def test_oversized_and_malformed_bodies(server):
    _httpd, base = server
    big = urllib.request.Request(base + "/api/jobs", data=b"x" * (api.MAX_BODY_BYTES + 1),
                                 method="POST")
    big.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(big, timeout=15) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    assert status == 413
    status, _type, payload = call(base, "/api/jobs", "POST", ["not", "an", "object"])
    assert status == 400 and "JSON object" in payload["error"]["message"]


def test_the_watch_scan_helper_matches_the_scanner(data_paths):
    """The API's automation helpers are the same ones the CLI watches with."""
    uploads = config.UPLOADS_DIR
    uploads.mkdir(parents=True, exist_ok=True)
    clip = uploads / "clip.mp4"
    clip.write_bytes(b"x")
    old = time.time() - 60
    os.utime(clip, (old, old))
    conn = db.connect()
    try:
        assert automation.find_new_media(conn) == [clip]
    finally:
        conn.close()
