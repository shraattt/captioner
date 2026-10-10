"""Tests for the QtMultimedia playback wrapper.

Two layers, deliberately separated:

* Deterministic tests that need no media backend at all — lazy construction,
  volume/mute defaults, signal forwarding, and the ``can_decode`` capability
  logic (driven against fake backends so they assert real behaviour rather
  than whatever this machine happens to support).
* One display-gated test that opens a real file and drives actual playback.
  It is skipped under the offscreen platform plugin, which never completes a
  media load (the player sits in LoadingMedia forever with no audio device),
  so real playback can only be proven in a windowed run.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtMultimedia import QMediaPlayer as QP
from PySide6.QtWidgets import QApplication

from captioner import config
from captioner import player as player_mod
from captioner.player import Player

OFFSCREEN = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"
requires_display = pytest.mark.skipif(
    OFFSCREEN,
    reason="QtMultimedia needs a real display/audio device; offscreen never finishes loading media",
)

SMOKE_WAV = config.UPLOADS_DIR / "smoke_speech.wav"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def player(qapp):
    return Player()


#can_decode


class _FakeMode:
    Decode = "decode"


class _FakeFF:
    Wave = "wave"
    MP3 = "mp3"
    AAC = "aac"
    FLAC = "flac"
    Mpeg4Audio = "m4a"
    Ogg = "ogg"
    Matroska = "mkv"
    MPEG4 = "mp4"
    QuickTime = "mov"
    WebM = "webm"


class _FakeAC:
    MP3 = "mp3"
    AAC = "aac"
    FLAC = "flac"
    Wave = "wave"
    WMA = "wma"
    ALAC = "alac"


def _fake_backend(file_formats=(), audio_codecs=(), raises=False):
    """A stand-in QMediaFormat reporting exactly the given capabilities."""

    class FakeFormat:
        ConversionMode = _FakeMode
        FileFormat = _FakeFF
        AudioCodec = _FakeAC

        def supportedFileFormats(self, mode):
            assert mode == _FakeMode.Decode
            if raises:
                raise RuntimeError("backend unavailable")
            return list(file_formats)

        def supportedAudioCodecs(self, mode):
            assert mode == _FakeMode.Decode
            if raises:
                raise RuntimeError("backend unavailable")
            return list(audio_codecs)

    return FakeFormat


def test_can_decode_rejects_non_media_paths(monkeypatch):
    monkeypatch.setattr(player_mod, "QMediaFormat", _fake_backend([_FakeFF.Wave]))
    assert Player.can_decode("notes.txt") is False
    assert Player.can_decode("archive.zip") is False
    assert Player.can_decode("no_extension") is False


def test_can_decode_accepts_format_the_backend_supports(monkeypatch):
    monkeypatch.setattr(player_mod, "QMediaFormat", _fake_backend([_FakeFF.Wave]))
    assert Player.can_decode("clip.wav") is True
    # Backend only handles WAV, so an MP3 must not be claimed as decodable.
    assert Player.can_decode("clip.mp3") is False
    assert Player.can_decode("movie.mkv") is False


def test_can_decode_accepts_codec_even_without_listed_container(monkeypatch):
    # Containers deliberately empty: only the codec table can say yes.
    monkeypatch.setattr(
        player_mod, "QMediaFormat", _fake_backend([], [_FakeAC.MP3, _FakeAC.FLAC])
    )
    assert Player.can_decode("song.mp3") is True
    assert Player.can_decode("song.flac") is True
    assert Player.can_decode("clip.wav") is False


def test_can_decode_is_conservative_when_backend_query_fails(monkeypatch):
    monkeypatch.setattr(player_mod, "QMediaFormat", _fake_backend(raises=True))
    assert Player.can_decode("clip.flac") is True


def test_can_decode_extension_matching_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(player_mod, "QMediaFormat", _fake_backend([_FakeFF.Wave]))
    assert Player.can_decode("CLIP.WAV") is True
    assert Player.can_decode("/some/dir/CLIP.WaV") is True


#lazy construction / state


def test_player_does_not_build_qt_multimedia_until_needed(player):
    """Constructing a Player must stay cheap and safe with no audio device."""
    assert player._player is None
    assert player._audio is None
    assert player.position_ms == 0
    assert player.duration_ms == 0
    assert player.is_stopped is True
    assert player.is_playing is False
    assert player.is_paused is False


def test_stop_is_a_noop_before_backend_exists(player):
    player.stop()  # must not raise, must not force construction
    assert player._player is None


def test_volume_and_mute_defaults_before_backend_exists(player):
    assert player.volume == 100
    assert player.muted is False
    # Setters are no-ops until the audio output exists, and must not raise.
    player.volume = 42
    player.muted = True
    assert player.volume == 100
    assert player.muted is False


#signal forwarding


def test_duration_signal_is_forwarded_and_clamped(player):
    seen = []
    player.duration_ms_changed.connect(seen.append)
    player._on_duration(1234)
    assert player.duration_ms == 1234
    assert seen == [1234]

    player._on_duration(-5)
    assert player.duration_ms == 0
    assert seen[-1] == 0


def test_playback_state_names_are_mapped_for_the_ui(player):
    seen = []
    player.playback_state_changed.connect(seen.append)
    player._on_state(QP.PlaybackState.StoppedState)
    player._on_state(QP.PlaybackState.PlayingState)
    player._on_state(QP.PlaybackState.PausedState)
    assert seen == ["stopped", "playing", "paused"]


def test_player_error_signal_carries_the_backend_message(player):
    seen = []
    player.player_error.connect(seen.append)
    player._on_error(QP.Error.NoError, "")
    assert seen == []
    player._on_error(QP.Error.FormatError, "cannot decode stream")
    assert seen == ["cannot decode stream"]


def test_player_error_falls_back_to_a_code_when_message_is_empty(player):
    seen = []
    player.player_error.connect(seen.append)
    player._on_error(QP.Error.FormatError, "")
    assert seen and seen[0].startswith("error(")


def test_ignored_signal_hooks_match_qt_6_arity(player):
    """Qt calls these hooks directly, so their arity must match the real signal.

    ``bufferProgressChanged`` carries a single 0.0-1.0 float in Qt 6 (not the
    Qt 5 start/end pair). A two-argument handler raised TypeError on every
    buffer update during real playback, so pin the arity here.
    """
    import inspect

    assert len(inspect.signature(player._on_buffer_progress).parameters) == 1
    player._on_buffer_progress(0.75)  # raised TypeError before the fix
    player._on_has_audio(True)
    player._on_active_tracks_changed()


#real playback (needs a display)


def _pump(app, seconds, predicate=None):
    """Spin the event loop until *predicate* is true or the deadline passes."""
    import time

    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        if predicate is not None and predicate():
            return True
        time.sleep(0.02)
    app.processEvents()
    return predicate() if predicate is not None else True


@requires_display
def test_real_file_loads_plays_and_advances_position(qapp):
    """The acceptance test for playback: open the WAV, hear it, position moves."""
    if not SMOKE_WAV.exists():
        pytest.skip(f"missing fixture {SMOKE_WAV}")

    assert Player.can_decode(str(SMOKE_WAV)) is True

    p = Player()
    errors = []
    p.player_error.connect(errors.append)
    p.open(str(SMOKE_WAV))

    loaded = _pump(qapp, 10.0, lambda: p.duration_ms > 0)
    assert loaded, f"media never finished loading (errors={errors})"
    assert p.duration_ms > 8000  # the fixture is ~14 s

    p.play()
    advanced = _pump(qapp, 10.0, lambda: p.position_ms > 200)
    assert advanced, "position never advanced after play()"
    assert p.position_ms <= p.duration_ms

    p.pause()
    assert p.is_paused is True
    p.stop()
    assert p.is_stopped is True
