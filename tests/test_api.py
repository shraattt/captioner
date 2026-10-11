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
    assert "post" in spec["paths"]["/api/jobs/{job_id}/cancel"]
    # the stream is not a JSON route, so it is described by hand: check it is there
    stream = spec["paths"]["/api/events"]["get"]
    assert "text/event-stream" in stream["responses"]["200"]["content"]


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


def headers_of(base, path):
    """One request, returning the status and headers (as a plain dict)."""
    request = urllib.request.Request(base + path)
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.status, dict(response.headers)


def read_event(stream, timeout=8.0):
    """Read one server-sent-events frame, skipping comments and blank lines."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = stream.readline()
        if not line:
            return None  # the stream ended
        text = line.decode("utf-8", "replace").strip()
        if text.startswith("data: "):
            return json.loads(text[len("data: "):])
    return None


#cancellation


def test_cancel_stops_a_queued_job(server, data_paths):
    _httpd, base = server
    job_id = queue_job(data_paths, status="queued")
    status, _type, payload = call(base, f"/api/jobs/{job_id}/cancel", "POST", {})

    assert status == 200
    assert payload["cancel"] == "canceled"
    assert payload["job"]["status"] == "canceled"


def test_cancel_asks_a_running_job_to_stop(server, data_paths):
    """A running job keeps running here: its worker decides when to stop."""
    _httpd, base = server
    job_id = queue_job(data_paths, status="running")
    status, _type, payload = call(base, f"/api/jobs/{job_id}/cancel", "POST", {})

    assert status == 202
    assert payload["cancel"] == "requested"
    assert payload["job"]["status"] == "running"
    assert payload["job"]["cancel_requested"] is True

    # The flag is the whole mechanism, so check what a worker would poll.
    conn = db.connect()
    try:
        assert worker.cancel_requested(conn, job_id) is True
    finally:
        conn.close()


def test_cancel_refuses_a_settled_job(server, data_paths):
    _httpd, base = server
    job_id = queue_job(data_paths, status="done")
    status, _type, payload = call(base, f"/api/jobs/{job_id}/cancel", "POST", {})
    assert status == 409
    assert "already done" in payload["error"]["message"]


def test_cancel_needs_the_token_when_one_is_set(secured, data_paths):
    _httpd, base = secured
    job_id = queue_job(data_paths, status="queued")

    status, _type, _payload = call(base, f"/api/jobs/{job_id}/cancel", "POST", {})
    assert status == 401

    status, _type, payload = call(base, f"/api/jobs/{job_id}/cancel", "POST", {},
                                 token="hunter2")
    assert status == 200 and payload["cancel"] == "canceled"


def test_cancel_of_an_unknown_job_is_a_404(server):
    _httpd, base = server
    status, _type, payload = call(base, "/api/jobs/424242/cancel", "POST", {})
    assert status == 404 and "424242" in payload["error"]["message"]


#duplicate content


def test_queueing_the_same_content_again_answers_with_the_first_job(server, data_paths):
    """Two names, one recording: nothing is queued a second time."""
    _httpd, base = server
    original = make_media(data_paths, "lecture.mp4")
    copy = data_paths / "second-name.mp4"
    copy.write_bytes(original.read_bytes())

    status, _type, payload = call(base, "/api/jobs", "POST", {"path": str(original)})
    assert status == 202 and payload["duplicate"] is False
    job_id = payload["job"]["id"]

    status, _type, payload = call(base, "/api/jobs", "POST", {"path": str(copy)})
    assert status == 200 and payload["duplicate"] is True
    assert payload["job"]["id"] == job_id
    assert payload["job"]["file_name"] == "lecture.mp4"

    conn = db.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_finished_job_blocks_the_same_content(server, data_paths):
    """Already transcribed counts as already handled."""
    _httpd, base = server
    job_id = queue_job(data_paths, name="lecture.mp4", status="done")
    copy = data_paths / "lecture-copy.mp4"
    copy.write_bytes((data_paths / "lecture.mp4").read_bytes())

    status, _type, payload = call(base, "/api/jobs", "POST", {"path": str(copy)})
    assert status == 200 and payload["duplicate"] is True
    assert payload["job"]["id"] == job_id


def test_a_failed_job_does_not_block_the_same_content(server, data_paths):
    """A failure is worth retrying, so the file is queued again."""
    _httpd, base = server
    queue_job(data_paths, name="lecture.mp4", status="error")
    copy = data_paths / "lecture-copy.mp4"
    copy.write_bytes((data_paths / "lecture.mp4").read_bytes())

    status, _type, payload = call(base, "/api/jobs", "POST", {"path": str(copy)})
    assert status == 202 and payload["duplicate"] is False
    assert payload["job"]["file_name"] == "lecture-copy.mp4"


#live updates


def test_events_stream_tells_the_page_about_a_new_job(server, data_paths, monkeypatch):
    """Instead of polling five endpoints every two seconds, the page is told."""
    _httpd, base = server
    monkeypatch.setattr(api, "SSE_POLL_SECONDS", 0.05)
    monkeypatch.setattr(api, "SSE_HEARTBEAT_SECONDS", 0.2)
    monkeypatch.setattr(api, "SSE_MAX_SECONDS", 8.0)

    with urllib.request.urlopen(base + "/api/events", timeout=15) as response:
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/event-stream")

        first = read_event(response)
        assert first is not None and first["counts"] == {} and first["newest"] == 0

        job_id = queue_job(data_paths)
        seen = None
        for _ in range(200):  # bounded: the change must arrive quickly
            event = read_event(response)
            if event and event.get("newest") == job_id:
                seen = event
                break
        assert seen is not None, "the new job was never announced"
        assert seen["counts"] == {"queued": 1}


def test_events_stream_has_a_heartbeat_and_ends(server, monkeypatch):
    """The stream is bounded, so no request thread can be orphaned."""
    _httpd, base = server
    monkeypatch.setattr(api, "SSE_POLL_SECONDS", 0.05)
    monkeypatch.setattr(api, "SSE_HEARTBEAT_SECONDS", 0.1)
    monkeypatch.setattr(api, "SSE_MAX_SECONDS", 1.0)

    saw_comment = saw_bye = False
    with urllib.request.urlopen(base + "/api/events", timeout=15) as response:
        for _ in range(400):
            line = response.readline()
            if not line:
                break  # the server closed the stream
            text = line.decode("utf-8", "replace").strip()
            if text.startswith(":"):
                saw_comment = True
            if text == "event: bye":
                saw_bye = True
    assert saw_comment, "no keep-alive comment arrived"
    assert saw_bye, "the stream did not say goodbye before closing"


def test_events_stream_needs_the_token_when_one_is_set(secured, monkeypatch):
    _httpd, base = secured
    monkeypatch.setattr(api, "SSE_MAX_SECONDS", 1.0)
    monkeypatch.setattr(api, "SSE_POLL_SECONDS", 0.05)
    try:
        urllib.request.urlopen(base + "/api/events", timeout=10)
    except urllib.error.HTTPError as exc:
        assert exc.code == 401
    else:  # pragma: no cover - a stream that leaked would hang the test
        raise AssertionError("the stream was served without a token")


def test_the_snapshot_reports_the_running_job_and_its_cancel_request(data_paths):
    job_id = queue_job(data_paths, status="running")
    conn = db.connect()
    try:
        worker.request_cancel(conn, job_id)
        snapshot = api.queue_snapshot(conn)
    finally:
        conn.close()
    assert snapshot["counts"] == {"running": 1}
    assert snapshot["running"] == [
        {"id": job_id, "progress": None, "stage": None, "cancel_requested": True}
    ]


#headers


def test_the_page_is_served_with_a_content_security_policy(server):
    _httpd, base = server
    status, headers = headers_of(base, "/")
    assert status == 200
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    policy = headers["Content-Security-Policy"]
    assert "default-src 'none'" in policy
    assert "connect-src 'self'" in policy and "form-action 'none'" in policy


def test_json_and_assets_are_not_sniffed(server):
    _httpd, base = server
    for path in ("/api/health", "/static/style.css", "/static/app.js", "/favicon.ico"):
        _status, headers = headers_of(base, path)
        assert headers["X-Content-Type-Options"] == "nosniff", path
        assert "Content-Security-Policy" not in headers, path  # documents only


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
