"""HTTP API and browser UI for the caption queue.

Deliberately built on ``http.server`` instead of a web framework. The whole
product is local-first and offline, so pulling in a server framework and its
dependency tree for one machine would cost more than it buys; the routes below
are small enough that a framework would only be in the way. A machine that can
run the captioner can serve this, with nothing installed.

Routes (all JSON unless noted):

    GET    /                          the browser UI (static page)
    GET    /favicon.ico               the app icon, for the browser tab
    GET    /openapi.json              OpenAPI 3.0 description of this API
    GET    /api/health                version, queue counts, selected engines
    GET    /api/engines               registered backends and whether they work
    GET    /api/models                model labels the pickers offer
    GET    /api/languages             translation targets
    GET    /api/jobs                  the queue, newest first
    POST   /api/jobs                  queue a file
    GET    /api/jobs/{id}             one job, with its caption blocks
    GET    /api/jobs/{id}/captions    captions as json, srt, vtt or text
    POST   /api/jobs/{id}/retry       requeue a settled job
    DELETE /api/jobs/{id}             remove a job and its caption rows
    POST   /api/jobs/{id}/translate   write captions.<lang>.srt/.vtt

Security: the bind address is loopback unless changed. The moment it is not,
a bearer token is mandatory - the server refuses to start without one, because
this API can queue local files and read their transcripts. With a token set,
every request must carry it (``Authorization: Bearer`` or ``?token=``), even
on loopback.

    python cli.py api                       # 127.0.0.1:8765
    python cli.py api --host 0.0.0.0 --token ...   # LAN, token required
"""
import hmac
import json
import logging
import os
import re
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__, automation, config, db, models, worker
from .engines import describe_all, probe
from .translate import LANGUAGES, language_code

logger = logging.getLogger(__name__)

VERSION = __version__
APP_NAME = "Captioner"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
HOST_ENV = "CAPTIONER_API_HOST"
PORT_ENV = "CAPTIONER_API_PORT"
TOKEN_ENV = "CAPTIONER_API_TOKEN"
MAX_BODY_BYTES = 64 * 1024
# A body this much larger than the limit is not worth reading out.
DRAIN_LIMIT = 8 * MAX_BODY_BYTES
DEFAULT_JOB_LIMIT = 200

WEB_ROOT = Path(__file__).resolve().parent.parent / "ui" / "web"
# The app icon doubles as the browser favicon, so the tab is branded without
# committing a second copy of a binary to the repository.
ICON_PATH = config.PROJECT_ROOT / "captioner.ico"

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}


class ApiError(Exception):
    """An error that becomes a JSON response with a real status code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Request:
    """Everything a route handler is allowed to need."""

    method: str
    path: str
    query: dict[str, list[str]]
    body: dict
    params: dict[str, str]
    conn: object
    server: "CaptionerHTTPServer"

    def param(self, name: str) -> str:
        return self.params.get(name, "")

    def job_id(self) -> int:
        raw = self.param("job_id")
        if not raw.isdigit():
            raise ApiError(400, f"job id must be a number, got {raw!r}")
        return int(raw)

    def body_str(self, name: str, required: bool = False) -> str | None:
        value = self.body.get(name)
        if value is None:
            if required:
                raise ApiError(400, f"missing field {name!r}")
            return None
        if not isinstance(value, str) or not value.strip():
            raise ApiError(400, f"field {name!r} must be a non-empty string")
        return value.strip()

    def body_flag(self, name: str) -> bool:
        value = self.body.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)


@dataclass
class Route:
    method: str
    path: str  # {job_id} marks a path parameter
    summary: str
    handler: Callable[[Request], tuple[int, object]]
    regex: object = field(init=False, repr=False)

    def __post_init__(self):
        self.regex = _compile_path(self.path)


def _compile_path(path: str):
    """Turn ``/api/jobs/{job_id}`` into a matching regular expression."""
    out, last = "", 0
    for match in re.finditer(r"\{(\w+)\}", path):
        out += re.escape(path[last:match.start()])
        out += f"(?P<{match.group(1)}>[^/]+)"
        last = match.end()
    return re.compile("^" + out + re.escape(path[last:]) + "$")


#helpers

def _query_dict(query: str) -> dict[str, list[str]]:
    from urllib.parse import parse_qs

    return parse_qs(query or "")


def _job_dict(row, conn=None) -> dict:
    """One job row as the API's JSON shape."""
    keys = set(row.keys())
    job = {
        "id": row["id"],
        "file_name": row["file_name"],
        "file_path": row["file_path"],
        "status": row["status"],
        "model": row["model_size"],
        "error": row["error"],
        "progress": row["progress"],
        "stage": row["stage"],
        "created_at": row["created_at"],
        "started_at": row["started_at"] if "started_at" in keys else None,
        "finished_at": row["finished_at"],
        "heartbeat_at": row["heartbeat_at"] if "heartbeat_at" in keys else None,
        "options": {
            "diarize": bool(row["diarize"]),
            "align": bool(row["align"]),
            "translate_to": row["translate_to"],
        },
    }
    out_dir = Path(config.OUTPUT_DIR) / f"job_{row['id']}"
    job["outputs"] = (
        sorted(path.name for path in out_dir.glob("captions*")) if out_dir.is_dir() else []
    )
    job["output_dir"] = str(out_dir) if out_dir.is_dir() else None
    if conn is not None:
        job["captions"] = int(conn.execute(
            "SELECT COUNT(*) FROM captions WHERE job_id = ?", (row["id"],)
        ).fetchone()[0])
    return job


def _fetch_job(conn, job_id: int):
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise ApiError(404, f"no job #{job_id}")
    return row


def _caption_rows(conn, job_id: int) -> list[dict]:
    return [
        {"seq": row["seq"], "start_ms": row["start_ms"], "end_ms": row["end_ms"],
         "text": row["text"]}
        for row in conn.execute(
            "SELECT seq, start_ms, end_ms, text FROM captions WHERE job_id = ? ORDER BY seq",
            (job_id,),
        )
    ]


#route handlers

def h_health(request: Request) -> tuple[int, object]:
    counts = {
        row["status"]: row["n"]
        for row in request.conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        )
    }
    return 200, {
        "status": "ok",
        "app": APP_NAME,
        "version": VERSION,
        "queue": counts,
        "engines": request.server.engine_summary(),
        "watch_dirs": [str(path) for path in automation.watch_dirs()],
        "webhook": bool(automation.webhook_url()),
    }


def h_engines(request: Request) -> tuple[int, object]:
    want_probe = request.query.get("probe", ["1"])[0] not in {"0", "false", "no"}
    payload = {
        "asr": {"backends": describe_all("asr"),
                "selected": request.server.selected("asr")},
        "mt": {"backends": describe_all("mt"),
               "selected": request.server.selected("mt")},
    }
    if want_probe:
        payload["asr"]["available"] = {k: list(v) for k, v in probe("asr").items()}
        payload["mt"]["available"] = {k: list(v) for k, v in probe("mt").items()}
    return 200, payload


def h_models(request: Request) -> tuple[int, object]:
    return 200, {"batch": models.batch_choices(), "live": models.live_choices()}


def h_languages(request: Request) -> tuple[int, object]:
    return 200, {"languages": LANGUAGES}


def h_jobs_list(request: Request) -> tuple[int, object]:
    status = request.query.get("status", [""])[0].strip()
    limit = request.query.get("limit", [""])[0]
    try:
        limit_value = int(limit) if limit else DEFAULT_JOB_LIMIT
    except ValueError:
        raise ApiError(400, "limit must be a number") from None
    limit_value = max(1, min(limit_value, 1000))

    sql = "SELECT * FROM jobs"
    args: list[object] = []
    if status:
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit_value)
    rows = request.conn.execute(sql, args).fetchall()
    return 200, {"jobs": [_job_dict(row, request.conn) for row in rows],
                 "count": len(rows)}


def h_jobs_create(request: Request) -> tuple[int, object]:
    raw_path = request.body_str("path", required=True)
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise ApiError(404, f"no such file: {raw_path}")
    if not path.is_file():
        raise ApiError(400, f"not a file: {raw_path}")
    if path.suffix.lower() not in automation.MEDIA_EXTS:
        raise ApiError(400, f"unsupported media type {path.suffix or '(none)'}")

    translate_to = request.body_str("translate_to")
    if translate_to:
        try:
            language_code(translate_to)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from None
    model = request.body_str("model")
    job_id = worker.enqueue(
        request.conn, str(path), model_size=model,
        diarize=request.body_flag("diarize"),
        align=request.body_flag("align"),
        translate_to=translate_to,
    )
    row = _fetch_job(request.conn, job_id)
    return 202, {"job": _job_dict(row, request.conn)}


def h_job_detail(request: Request) -> tuple[int, object]:
    row = _fetch_job(request.conn, request.job_id())
    payload = _job_dict(row, request.conn)
    payload["caption_blocks"] = _caption_rows(request.conn, request.job_id())
    return 200, {"job": payload}


def h_job_captions(request: Request) -> tuple[int, object]:
    job_id = request.job_id()
    _fetch_job(request.conn, job_id)
    fmt = request.query.get("format", ["json"])[0].lower()
    out_dir = Path(config.OUTPUT_DIR) / f"job_{job_id}"

    if fmt in {"srt", "vtt"}:
        artifact = out_dir / f"captions.{fmt}"
        if not artifact.exists():
            raise ApiError(404, f"job #{job_id} has no {fmt} output yet")
        return 200, {"text": artifact.read_text(encoding="utf-8"), "format": fmt,
                     "job_id": job_id}
    blocks = _caption_rows(request.conn, job_id)
    if fmt == "text":
        return 200, {"text": "\n".join(block["text"] for block in blocks),
                     "format": "text", "job_id": job_id}
    return 200, {"captions": blocks, "format": "json", "job_id": job_id}


def h_job_retry(request: Request) -> tuple[int, object]:
    job_id = request.job_id()
    row = _fetch_job(request.conn, job_id)
    if row["status"] == "running":
        raise ApiError(409, f"job #{job_id} is running; cancel or wait first")
    worker.retry(request.conn, job_id)
    return 200, {"job": _job_dict(_fetch_job(request.conn, job_id), request.conn)}


def h_job_delete(request: Request) -> tuple[int, object]:
    job_id = request.job_id()
    if not worker.delete_job(request.conn, job_id):
        raise ApiError(409, f"job #{job_id} is running or does not exist")
    return 200, {"deleted": job_id}


def h_job_translate(request: Request) -> tuple[int, object]:
    from .translate import translate_subtitles

    job_id = request.job_id()
    row = _fetch_job(request.conn, job_id)
    target = request.body_str("to", required=True)
    try:
        language_code(target)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None
    out_dir = Path(config.OUTPUT_DIR) / f"job_{job_id}"
    try:
        paths = translate_subtitles(out_dir, target, speaker_names=bool(row["diarize"]))
    except FileNotFoundError as exc:
        raise ApiError(404, str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - reported as a 502 to the caller
        logger.exception("translation of job %s failed", job_id)
        raise ApiError(502, f"translation failed: {type(exc).__name__}: {exc}") from exc
    request.conn.execute("UPDATE jobs SET translate_to = ? WHERE id = ?", (target, job_id))
    request.conn.commit()
    return 200, {"job_id": job_id, "target": target, "files": [p.name for p in paths]}


ROUTES: tuple[Route, ...] = (
    Route("GET", "/api/health", "service, queue and engine status", h_health),
    Route("GET", "/api/engines", "registered backends and their availability", h_engines),
    Route("GET", "/api/models", "model labels the pickers offer", h_models),
    Route("GET", "/api/languages", "translation target languages", h_languages),
    Route("GET", "/api/jobs", "list jobs, newest first", h_jobs_list),
    Route("POST", "/api/jobs", "queue a media file", h_jobs_create),
    Route("GET", "/api/jobs/{job_id}", "one job with its caption blocks", h_job_detail),
    Route("GET", "/api/jobs/{job_id}/captions", "captions as json, srt, vtt or text",
          h_job_captions),
    Route("POST", "/api/jobs/{job_id}/retry", "requeue a settled job", h_job_retry),
    Route("DELETE", "/api/jobs/{job_id}", "remove a job and its captions", h_job_delete),
    Route("POST", "/api/jobs/{job_id}/translate", "write translated caption files",
          h_job_translate),
)


#the server

class CaptionerHTTPServer(ThreadingHTTPServer):
    """Threaded server carrying the API token and a cached engine summary."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, token: str = ""):
        super().__init__(address, handler)
        self.token = token
        self._engine_lock = threading.Lock()
        self._engine_cache: dict | None = None

    def selected(self, kind: str) -> str:
        from .engines import resolve_name

        try:
            return resolve_name(kind)
        except Exception:  # noqa: BLE001 - a bad env var must not break health
            return ""

    def engine_summary(self) -> dict:
        """Selected backends, cached: /api/health can be polled every second."""
        with self._engine_lock:
            if self._engine_cache is None:
                self._engine_cache = {
                    "asr": {"selected": self.selected("asr"),
                            "backends": list(describe_all("asr"))},
                    "mt": {"selected": self.selected("mt"),
                           "backends": list(describe_all("mt"))},
                }
            return self._engine_cache


class ApiHandler(BaseHTTPRequestHandler):
    """Routes one request; every response is JSON except the UI and its assets."""

    server_version = f"CaptionerAPI/{VERSION}"
    protocol_version = "HTTP/1.1"

    #plumbing
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def do_GET(self):  # noqa: N802 - stdlib naming
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self):  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        from urllib.parse import urlparse

        parsed = urlparse(self.path)
        path, query = parsed.path, _query_dict(parsed.query)
        try:
            # Read the body before deciding anything: answering a request whose
            # body the client is still sending resets the connection under it.
            body = self._read_body() if method in {"POST", "DELETE"} else {}
            if not self._authorized(query):
                raise ApiError(401, "missing or wrong bearer token")
            if method in {"POST", "DELETE"} and not self._same_origin():
                raise ApiError(403, "cross-origin request rejected")
            if path == "/" or path.startswith("/static/"):
                self._send_asset(path)
                return
            if path == "/favicon.ico":
                self._send_icon()
                return
            if path == "/openapi.json":
                self._send_json(200, openapi_spec())
                return
            for route in ROUTES:
                match = route.regex.match(path)
                if match is None or route.method != method:
                    continue
                conn = db.connect()
                try:
                    request = Request(method, path, query, body, match.groupdict(),
                                      conn, self.server)
                    status, payload = route.handler(request)
                finally:
                    conn.close()
                self._send_json(status, payload)
                return
            allowed = sorted({r.method for r in ROUTES if r.regex.match(path)})
            if allowed:
                self._send_json(405, {"error": {"message": f"{method} not allowed",
                                                "allowed": allowed}})
                return
            raise ApiError(404, f"no route for {path}")
        except ApiError as exc:
            self._send_json(exc.status, {"error": {"message": exc.message, "path": path}})
        except Exception as exc:  # noqa: BLE001 - a traceback must never leak out
            logger.exception("API error handling %s %s", method, path)
            self._send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

    def _authorized(self, query: dict[str, list[str]]) -> bool:
        token = getattr(self.server, "token", "")
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        presented = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not presented and query.get("token"):
            presented = query["token"][0]
        return hmac.compare_digest(presented, token)

    def _same_origin(self) -> bool:
        """Reject browser requests from another site (CSRF guard)."""
        origin = self.headers.get("Origin")
        if not origin:
            return True  # curl, scripts, the UI's own same-origin fetches
        host = self.headers.get("Host", "")
        return origin == f"http://{host}" or origin == f"https://{host}"

    def _read_body(self) -> dict:
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length else 0
        except ValueError:
            raise ApiError(400, "bad Content-Length") from None
        if size <= 0:
            return {}
        if size > MAX_BODY_BYTES:
            # The client is still sending: read what it has queued before
            # answering, or the socket resets and the 413 never arrives.
            self._drain(size)
            raise ApiError(413, "request body too large")
        raw = self.rfile.read(size)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ApiError(400, "body must be JSON") from None
        if not isinstance(payload, dict):
            raise ApiError(400, "body must be a JSON object")
        return payload

    def _drain(self, size: int) -> None:
        """Discard an over-sized body so the client can read our answer."""
        if size > DRAIN_LIMIT:
            self.close_connection = True  # too big to bother: just hang up
            return
        remaining = size
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    #responses
    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_icon(self) -> None:
        """Serve captioner.ico, which the page links as its favicon."""
        if not ICON_PATH.is_file():
            raise ApiError(404, "no icon in this install")
        body = ICON_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/x-icon")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_asset(self, path: str) -> None:
        """Serve the UI from ui/web, with the path confined to that folder."""
        relative = "" if path == "/" else path[len("/static/"):]
        relative = relative or "index.html"
        target = (WEB_ROOT / relative).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            raise ApiError(404, "no such asset") from None
        if not target.is_file():
            raise ApiError(404, f"no such asset: {path}")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _content_type(target))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
    }.get(suffix, "application/octet-stream")


def openapi_spec() -> dict:
    """A hand-written OpenAPI 3.0 description of the routes above."""
    paths: dict[str, dict] = {}
    for route in ROUTES:
        operation = {
            "summary": route.summary,
            "responses": {"200": {"description": "success"},
                          "400": {"description": "bad request"},
                          "404": {"description": "not found"}},
        }
        for name in re.findall(r"\{(\w+)\}", route.path):
            operation.setdefault("parameters", []).append({
                "name": name, "in": "path", "required": True,
                "schema": {"type": "string"},
            })
        paths.setdefault(route.path, {})[route.method.lower()] = operation
    return {
        "openapi": "3.0.3",
        "info": {
            "title": f"{APP_NAME} API",
            "version": VERSION,
            "description": "Local captioning queue: transcribe, translate and fetch "
                           "subtitles over HTTP.",
        },
        "servers": [{"url": f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"}],
        "paths": paths,
    }


#entry points

def _env_port() -> int:
    raw = os.environ.get(PORT_ENV, "")
    try:
        return int(raw) if raw else DEFAULT_PORT
    except ValueError:
        return DEFAULT_PORT


def _is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def create_server(host: str = None, port: int = None, token: str = None,
                  bind_and_activate: bool = True) -> CaptionerHTTPServer:
    """Build (and by default bind) the server; port 0 picks a free port."""
    host = host or os.environ.get(HOST_ENV) or DEFAULT_HOST
    port = _env_port() if port is None else int(port)
    token = (token if token is not None else os.environ.get(TOKEN_ENV, "")).strip()
    if not _is_loopback(host) and not token:
        raise SystemExit(
            f"refusing to bind {host} without a token: this API queues local files "
            f"and returns their transcripts. Set {TOKEN_ENV} (or pass --token)."
        )
    return CaptionerHTTPServer((host, port), ApiHandler, token=token)


def serve(host: str = None, port: int = None, token: str = None,
          open_browser: bool = False) -> int:
    """Run the server until Ctrl+C. Returns a process exit code."""
    server = create_server(host, port, token)
    address, actual_port = server.server_address[:2]
    ui_url = f"http://{address}:{actual_port}/"
    logger.info("API listening on %s", ui_url)
    print(f"{APP_NAME} API + UI on {ui_url}")
    print(f"  docs:  {ui_url}openapi.json")
    print(f"  auth:  {'bearer token required' if server.token else 'loopback only'}")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(ui_url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(serve())
