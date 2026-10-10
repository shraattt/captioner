"""QtMultimedia audio playback for caption preview files.

Wraps PySide6.QtMultimedia so the dashboard can play back the audio behind a
captioned clip. The player owns its own QMediaPlayer + QAudioOutput, and
forwards Qt signals into simple int/str signals the UI can bind to.

Usable standalone for smoke tests:

    from captioner.player import Player
    p = Player()
    p.open("data/uploads/smoke_speech.wav")
    p.play()
    p.wait_until_done()     # blocks until playback finishes or an error
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtMultimedia import (
    QAudioOutput,
    QMediaFormat,
    QMediaPlayer,
)
from PySide6.QtMultimedia import (
    QMediaPlayer as QP,
)

logger = logging.getLogger(__name__)


def _error_name(code: QP.Error) -> str:
    """Human-readable name for a QMediaPlayer.Error.

    PySide6 does not expose these enums as ints, so ``int(code)`` raises
    TypeError on this build — take the enum name instead.
    """
    return getattr(code, "name", None) or str(code)

# File formats the Qt backend on this build knows how to decode. WAV (and MP3
# where present) are the common caption-source formats; the player degrades
# gracefully on anything else rather than crashing.
_DECODE_FILE_FORMATS = frozenset(
    {
        QMediaFormat.FileFormat.Wave,
        QMediaFormat.FileFormat.MP3,
        QMediaFormat.FileFormat.AAC,
        QMediaFormat.FileFormat.FLAC,
        QMediaFormat.FileFormat.Mpeg4Audio,
        QMediaFormat.FileFormat.Ogg,
        QMediaFormat.FileFormat.Matroska,
        QMediaFormat.FileFormat.MPEG4,
        QMediaFormat.FileFormat.QuickTime,
        QMediaFormat.FileFormat.WebM,
    }
)

# Audio codecs the backend can decode.
_DECODE_AUDIO_CODECS = frozenset(
    {
        QMediaFormat.AudioCodec.MP3,
        QMediaFormat.AudioCodec.AAC,
        QMediaFormat.AudioCodec.FLAC,
        QMediaFormat.AudioCodec.Wave,
        QMediaFormat.AudioCodec.WMA,
        QMediaFormat.AudioCodec.ALAC,
    }
)


class Player(QObject):
    """Thin wrapper around QMediaPlayer for captioning preview playback."""

    position_ms_changed = Signal(int)
    duration_ms_changed = Signal(int)
    player_error = Signal(str)
    playback_state_changed = Signal(str)  # "stopped" / "playing" / "paused"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._player: QMediaPlayer | None = None
        self._audio: QAudioOutput | None = None
        self._fmt: QMediaFormat | None = None
        self._position_ms: int = 0
        self._duration_ms: int = 0
        self._last_pos_update: float = 0.0
        self._position_timer_active = False

    #lifecycle

    def _ensure(self) -> None:
        if self._player is not None:
            return
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._audio.setVolume(1.0)
        self._player.setAudioOutput(self._audio)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.errorOccurred.connect(self._on_error)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.hasAudioChanged.connect(self._on_has_audio)
        self._player.bufferProgressChanged.connect(self._on_buffer_progress)
        self._player.activeTracksChanged.connect(self._on_active_tracks_changed)
        self._position_ms = 0
        self._duration_ms = 0

    #public API

    @staticmethod
    def can_decode(path: str) -> bool:
        """Return True if the Qt backend is likely to be able to decode *path*.

        This is a lightweight pre-check using the file extension against the
        formats/codecs the backend reports as supported. It is advisory only:
        ``open`` still tries the file and reports errors through the signal.
        """
        ext = Path(path).suffix.lower()
        if not ext:
            return False
        try:
            fmt = QMediaFormat()
            file_formats = fmt.supportedFileFormats(QMediaFormat.ConversionMode.Decode)
            audio_codecs = fmt.supportedAudioCodecs(QMediaFormat.ConversionMode.Decode)
        except Exception:
            return True  # conservative: assume yes if we can't query
        ext_map = {
            ".wav": QMediaFormat.FileFormat.Wave,
            ".mp3": QMediaFormat.FileFormat.MP3,
            ".m4a": QMediaFormat.FileFormat.AAC,
            ".aac": QMediaFormat.FileFormat.AAC,
            ".flac": QMediaFormat.FileFormat.FLAC,
            ".ogg": QMediaFormat.FileFormat.Ogg,
            ".opus": QMediaFormat.FileFormat.Ogg,
            ".webm": QMediaFormat.FileFormat.WebM,
            ".mp4": QMediaFormat.FileFormat.MPEG4,
            ".mov": QMediaFormat.FileFormat.QuickTime,
            ".mkv": QMediaFormat.FileFormat.Matroska,
            ".m4v": QMediaFormat.FileFormat.MPEG4,
        }
        codec_map = {
            ".mp3": QMediaFormat.AudioCodec.MP3,
            ".wav": QMediaFormat.AudioCodec.Wave,
            ".aac": QMediaFormat.AudioCodec.AAC,
            ".m4a": QMediaFormat.AudioCodec.AAC,
            ".flac": QMediaFormat.AudioCodec.FLAC,
            ".wma": QMediaFormat.AudioCodec.WMA,
            ".mp4": QMediaFormat.AudioCodec.AAC,
            ".mov": QMediaFormat.AudioCodec.AAC,
        }
        want_fmt = ext_map.get(ext)
        want_codec = codec_map.get(ext)
        if want_fmt is not None and want_fmt in file_formats:
            return True
        if want_codec is not None and want_codec in audio_codecs:
            return True
        return False

    def open(self, path: str) -> None:
        """Try to load *path* as media the Qt backend can decode.

        Arguments are any path ``QMediaPlayer`` can open (wav/mp3 most
        reliable; mp4/mkv/webm when the backend has the codecs). Errors are
        reported through ``player_error`` rather than raised.
        """
        self._ensure()
        self.stop()
        src = Path(path).resolve().as_uri()
        self._player.setSource(src)
        logger.debug("Player opened %s", src)

    @Slot()
    def play(self) -> None:
        self._ensure()
        self._player.play()

    @Slot()
    def pause(self) -> None:
        self._ensure()
        self._player.pause()

    @Slot()
    def stop(self) -> None:
        if self._player is not None:
            self._player.stop()

    def start_async(self) -> None:
        """Start playback from the current position without blocking."""
        self._ensure()
        self._player.play()

    @Slot()
    def seek(self, position_ms: int) -> None:
        """Jump to *position_ms* (clamped to [0, duration])."""
        self._ensure()
        ms = max(0, min(int(position_ms), self._duration_ms or 0))
        self._player.setPosition(ms)
        self._position_ms = ms

    #volume / mute

    @property
    def volume(self) -> int:
        """Current volume as an integer percent 0..100."""
        if self._audio is None:
            return 100
        return int(round(max(0.0, min(1.0, self._audio.volume())) * 100))

    @volume.setter
    def volume(self, percent: int) -> None:
        if self._audio is None:
            return
        self._audio.setVolume(max(0.0, min(1.0, float(percent) / 100.0)))

    @property
    def muted(self) -> bool:
        return self._audio.isMuted() if self._audio is not None else False

    @muted.setter
    def muted(self, on: bool) -> None:
        if self._audio is not None:
            self._audio.setMuted(on)

    #querying

    @property
    def position_ms(self) -> int:
        return self._position_ms

    @property
    def duration_ms(self) -> int:
        return self._duration_ms

    @property
    def is_playing(self) -> bool:
        if self._player is None:
            return False
        return self._player.playbackState() == QP.PlaybackState.PlayingState

    @property
    def is_paused(self) -> bool:
        if self._player is None:
            return False
        return self._player.playbackState() == QP.PlaybackState.PausedState

    @property
    def is_stopped(self) -> bool:
        if self._player is None:
            return True
        return self._player.playbackState() == QP.PlaybackState.StoppedState

    #blocking helper for tests

    def wait_until_done(self, timeout: float = 30.0) -> str | None:
        """Block until playback finishes, stops, or an error occurs.

        Returns ``"finished"``, ``"stopped"``, or an error string, or ``None``
        on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._player is not None:
                ms = self._player.mediaStatus()
                if ms == QP.MediaStatus.LoadedMedia and not self.is_playing and self._position_ms > 0:
                    # loaded but not playing and we already moved: treat as stopped
                    return "stopped"
                if ms == QP.MediaStatus.EndOfMedia:
                    return "finished"
                err = self._player.error()
                if err != QP.Error.NoError:
                    return self._player.errorString() or f"error({_error_name(err)})"
            time.sleep(0.05)
        return None

    #Qt slots / signal handlers

    @Slot(int)
    def _on_position(self, ms: int) -> None:
        self._position_ms = ms
        now = time.time()
        if now - self._last_pos_update > 0.05:
            self._last_pos_update = now
            self.position_ms_changed.emit(ms)

    @Slot(int)
    def _on_duration(self, ms: int) -> None:
        self._duration_ms = max(0, ms)
        self.duration_ms_changed.emit(self._duration_ms)

    @Slot(QP.MediaStatus)
    def _on_media_status(self, status: QP.MediaStatus) -> None:
        # When the backend finishes a load that failed, report it once.
        if status == QP.MediaStatus.InvalidMedia:
            err = self._player.error() if self._player is not None else QP.Error.NoError
            if err != QP.Error.NoError:
                self.player_error.emit(
                    self._player.errorString()
                    or f"unsupported media ({_error_name(err)})"
                )

    @Slot(QP.Error, str)
    def _on_error(self, error_code: QP.Error, error_string: str) -> None:
        if error_code != QP.Error.NoError:
            logger.warning("Player error %s: %s", _error_name(error_code), error_string)
            self.player_error.emit(error_string or f"error({_error_name(error_code)})")

    @Slot(QP.PlaybackState)
    def _on_state(self, state: QP.PlaybackState) -> None:
        name = {
            QP.PlaybackState.StoppedState: "stopped",
            QP.PlaybackState.PlayingState: "playing",
            QP.PlaybackState.PausedState: "paused",
        }.get(state, "unknown")
        self.playback_state_changed.emit(name)

    @Slot(bool)
    def _on_has_audio(self, has_audio: bool) -> None:  # noqa: F841 - hook registered in _ensure
        pass

    @Slot(float)
    def _on_buffer_progress(self, progress: float) -> None:  # noqa: F841
        # Qt 6 emits a single 0.0-1.0 fraction here (not the Qt 5 start/end
        # pair), so the slot must take exactly one float argument.
        pass

    @Slot()
    def _on_active_tracks_changed(self) -> None:  # noqa: F841
        pass
