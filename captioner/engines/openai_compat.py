"""Backends for OpenAI-compatible HTTP APIs.

One class does the talking; two subclasses use it. Any server that speaks the
same wire protocol works, which is the point of the registry:

  * cloud: OpenAI, Groq, Deepgram-compatible gateways, OpenRouter
  * local: faster-whisper-server, whisper.cpp's server, LM Studio, Ollama's
    OpenAI shim, vLLM

Point ``CAPTIONER_API_BASE_URL`` at one of those and set a key (some local
servers ignore the key, but the header is always sent).

Every request goes through an injectable ``transport`` so tests can exercise
the parsing and error paths without a network.
"""
import json
import logging
import os
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from ..segmenter import Word
from .registry import ASR, MT, EngineError, register

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
KEY_VARS = ("OPENAI_API_KEY", "CAPTIONER_API_KEY")
BASE_URL_VAR = "CAPTIONER_API_BASE_URL"

# A transport takes a prepared request and a timeout and returns (status, body).
Transport = Callable[[urllib.request.Request, float], tuple[int, bytes]]


def urlopen_transport(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    """Default transport: real HTTP, with the error body kept instead of raised."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]]):
    """Encode a multipart/form-data body (stdlib has no builder for this)."""
    boundary = f"----CaptionerBoundary{uuid.uuid4().hex}"
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    for name, (filename, blob) in files.items():
        out += f"--{boundary}\r\n".encode()
        out += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        out += blob + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", bytes(out)


def _error_message(raw: bytes) -> str:
    """The API's own error text, falling back to a truncated body."""
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return raw.decode("utf-8", errors="replace")[:300]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if payload.get("message"):
            return str(payload["message"])
    return json.dumps(payload)[:300]


def _first_env(names: Sequence[str]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


class _OpenAIBackend:
    """Shared configuration, request plumbing and error handling."""

    name = "openai"

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        transport: Transport = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or _first_env(KEY_VARS)
        self.base_url = (
            base_url or os.environ.get(BASE_URL_VAR) or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = float(timeout)
        self._transport = transport or urlopen_transport

    def describe(self) -> str:
        return f"OpenAI-compatible API at {self.base_url}"

    def available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, f"set {KEY_VARS[0]} to use the {self.name} backend"
        return True, self.base_url

    def _post(self, path: str, payload: dict = None, body: bytes = None,
              content_type: str = None) -> dict:
        """POST JSON or a prepared body; return the parsed JSON response."""
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key or 'none'}"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            headers["Content-Type"] = content_type or "application/octet-stream"
        request = urllib.request.Request(url, data=body or b"", headers=headers, method="POST")
        try:
            status, raw = self._transport(request, self.timeout)
        except Exception as exc:  # noqa: BLE001 - everything becomes EngineError
            raise EngineError(f"{self.name}: request to {url} failed: {exc}") from exc
        if not 200 <= status < 300:
            raise EngineError(f"{self.name}: HTTP {status} from {url}: {_error_message(raw)}")
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise EngineError(f"{self.name}: {url} did not return JSON") from exc


class OpenAIWhisperEngine(_OpenAIBackend):
    """``POST /audio/transcriptions`` with word-level timestamps."""

    name = "openai"
    DEFAULT_MODEL = "whisper-1"
    ENV_MODEL = "CAPTIONER_ASR_MODEL"
    # The endpoint rejects anything larger; say so here instead of after upload.
    MAX_UPLOAD_BYTES = 24 * 1024 * 1024

    def __init__(self, model: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model or os.environ.get(self.ENV_MODEL) or self.DEFAULT_MODEL

    def describe(self) -> str:
        return f"audio/transcriptions API ({self.model}) at {self.base_url}"

    def available(self) -> tuple[bool, str]:
        ok, why = super().available()
        if not ok:
            return ok, why
        return True, f"{self.model} at {self.base_url}"

    def transcribe_words(
        self,
        audio_path: str,
        model_size: str = None,
        on_progress=None,
        on_stage=None,
        should_cancel=None,
        language: str | None = None,
        on_language=None,
    ) -> list[Word]:
        """Timed words for one file, uploaded to the API.

        The upload is one request, so ``should_cancel`` is only honored before
        it starts and after it returns: there is no mid-request interruption.
        """
        path = Path(audio_path)
        if not path.exists():
            raise EngineError(f"{self.name}: no such file: {audio_path}")
        if should_cancel is not None and should_cancel():
            from ..transcribe import TranscriptionCancelled

            raise TranscriptionCancelled("canceled by the user")
        size = path.stat().st_size
        if size > self.MAX_UPLOAD_BYTES:
            raise EngineError(
                f"{self.name}: {path.name} is {size / 1e6:.1f} MB, the API limit is "
                f"{self.MAX_UPLOAD_BYTES / 1e6:.0f} MB; extract a smaller clip or use "
                "the local faster-whisper engine"
            )
        if on_stage is not None:
            on_stage("uploading", None)
        model = model_size or self.model
        fields = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        if language:
            # The API accepts an ISO-639-1 hint; omitting it lets the server
            # detect, which is what an unset job language means here.
            fields["language"] = language
        content_type, body = _multipart(fields, {"file": (path.name, path.read_bytes())})
        data = self._post("/audio/transcriptions", body=body, content_type=content_type)
        if on_language is not None and isinstance(data, dict) and data.get("language"):
            on_language(str(data["language"]), None)
        words = words_from_response(data)
        if on_stage is not None:
            on_stage("transcribing", 1.0)
        if on_progress is not None:
            on_progress(1.0)
        logger.info("API transcribed %s: %d words", path.name, len(words))
        return words

    def model_is_cached(self, model_size: str) -> bool:
        """Nothing to download locally, but the API needs a key."""
        return bool(self.api_key)


def words_from_response(data: dict) -> list[Word]:
    """Timed words out of a verbose_json response.

    ``words`` is preferred. Servers that only return ``segments`` still work:
    each segment's text is split into tokens with its duration apportioned by
    token length, which is approximate but keeps the segmenter's captions
    intact. A response with neither raises, because inventing one long word
    for a whole file would silently ruin every caption cue.
    """
    words: list[Word] = []
    for item in data.get("words") or []:
        text = str(item.get("word") or item.get("text") or "").strip()
        start, end = item.get("start"), item.get("end")
        if text and isinstance(start, (int, float)) and isinstance(end, (int, float)):
            words.append(Word(text=text, start=float(start), end=float(end)))
    if words:
        return words

    for seg in data.get("segments") or []:
        text = str(seg.get("text") or "").strip()
        start, end = seg.get("start"), seg.get("end")
        if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        words.extend(_apportion(text, float(start), float(end)))
    if words:
        logger.warning("API returned segments only: word timings are approximate")
        return words

    raise EngineError(
        "the API returned no word or segment timings; "
        "use response_format=verbose_json with timestamp_granularities=word"
    )


def _apportion(text: str, start: float, end: float) -> list[Word]:
    """Split one segment's text across its span, weighted by token length."""
    tokens = text.split()
    if not tokens:
        return []
    weights = [len(token) + 1 for token in tokens]
    total = sum(weights)
    span = max(0.0, end - start)
    out: list[Word] = []
    cursor = start
    for token, weight in zip(tokens, weights, strict=True):
        portion = span * weight / total
        out.append(Word(text=token, start=cursor, end=cursor + portion))
        cursor += portion
    return out


class OpenAIChatTranslator(_OpenAIBackend):
    """Translate subtitle lines with ``/chat/completions``.

    Lines are sent numbered in one request; the reply is parsed back by number
    and, if that fails (models do occasionally drop a line), the batch is
    retried a line at a time. Nothing is cached here beyond the caller's own
    cache, because the same text can legitimately translate differently.
    """

    name = "openai"
    DEFAULT_MODEL = "gpt-4o-mini"
    ENV_MODEL = "CAPTIONER_MT_MODEL"
    SYSTEM_PROMPT = (
        "You translate subtitle lines. Translate every numbered line from {source} "
        "into {language}. Keep the same line numbers, keep the same order, one "
        "translated line per input line. Output the numbered lines only, with no "
        "notes or explanations."
    )

    def __init__(self, model: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model or os.environ.get(self.ENV_MODEL) or self.DEFAULT_MODEL

    def describe(self) -> str:
        return f"chat completions API ({self.model}) at {self.base_url}"

    def available(self) -> tuple[bool, str]:
        ok, why = super().available()
        if not ok:
            return ok, why
        return True, f"{self.model} at {self.base_url}"

    def translate_many(self, texts: Sequence[str], target: str,
                       source: str | None = None) -> list[str]:
        """Translate ``texts`` into ``target``, preserving order and length.

        ``source`` is the language being read (``None`` = English): a hosted
        model usually works it out for itself, but naming it is free and stops
        a short line in a rare script from being read as the wrong language.
        """
        lines = [str(text) for text in texts]
        if not lines:
            return []
        parsed = self._translate_batch(lines, target, source)
        if parsed is not None:
            return parsed
        logger.warning("%s dropped lines in a batch reply; retrying one at a time",
                       self.name)
        return [self._translate_one(line, target, source) for line in lines]

    def _chat(self, user_content: str, target: str, source: str | None = None) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT.format(
                    language=_language_label(target),
                    source=_language_label(source or "en"))},
                {"role": "user", "content": user_content},
            ],
        }
        data = self._post("/chat/completions", payload=payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError(f"{self.name}: unexpected response shape: {data}") from exc
        return str(content or "")

    def _translate_batch(self, lines: list[str], target: str,
                         source: str | None = None) -> list[str] | None:
        numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))
        return _parse_numbered(self._chat(numbered, target, source), len(lines))

    def _translate_one(self, line: str, target: str, source: str | None = None) -> str:
        content = self._chat(f"1. {line}", target, source)
        parsed = _parse_numbered(content, 1)
        if parsed is not None:
            return parsed[0]
        return content.strip()


# "1. text", "2) text", "3: text" - models are not consistent about the mark.
_NUMBERED_LINE = re.compile(r"^\s*(\d{1,3})\s*[.):]?\s*(.+?)\s*$")


def _parse_numbered(content: str, count: int) -> list[str] | None:
    """``["line 1", "line 2"...]`` from a numbered reply, or None if incomplete.

    Returning None rather than guessing keeps a bad reply from silently
    shifting every following subtitle by one line.
    """
    found: dict[int, str] = {}
    for raw in content.splitlines():
        match = _NUMBERED_LINE.match(raw)
        if match is None:
            continue  # surrounding prose, or a line the model forgot to number
        index = int(match.group(1))
        if 1 <= index <= count:
            found[index] = match.group(2)
    if len(found) != count:
        return None
    return [found[index] for index in range(1, count + 1)]


def _language_label(code: str) -> str:
    """Display name for a language code, without importing the heavy module."""
    from ..languages import translation_name

    return translation_name(code) or code


def _register() -> None:
    register(ASR, "openai", OpenAIWhisperEngine,
             describe="audio/transcriptions API (OpenAI or compatible)")
    register(MT, "openai", OpenAIChatTranslator,
             describe="chat completions API (OpenAI or compatible)")


_register()
