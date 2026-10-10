"""Tests for the dashboard playback panel.

The panel is deliberately decoupled from QtMultimedia: it talks to whatever
``player`` object it is handed, so these tests drive it with a fake player and
assert the behaviour a user actually sees — buttons wire through, the seek bar
tracks position without fighting a drag, the clock labels read correctly, and
the caption line under the playhead gets highlighted. No audio device needed,
so everything runs under the offscreen platform plugin.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QPlainTextEdit

from captioner import config, db, worker
from captioner.segmenter import Block, Word
from ui import dashboard as dashboard_mod
from ui.dashboard import Dashboard, PlaybackControls


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class FakePlayer(QObject):
    """Stand-in for captioner.player.Player that records calls."""

    position_ms_changed = Signal(int)
    duration_ms_changed = Signal(int)
    player_error = Signal(str)
    playback_state_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.opened = []
        self.played = 0
        self.paused = 0
        self.stopped = 0
        self.seeks = []
        self.volume = 100
        self.muted = False
        self.position_ms = 0
        self.is_playing = False

    def open(self, path):
        self.opened.append(str(path))

    def play(self):
        self.played += 1
        self.is_playing = True
        self.playback_state_changed.emit("playing")

    def pause(self):
        self.paused += 1
        self.is_playing = False
        self.playback_state_changed.emit("paused")

    def stop(self):
        self.stopped += 1
        self.is_playing = False
        self.playback_state_changed.emit("stopped")

    def seek(self, ms):
        self.seeks.append(int(ms))
        self.position_ms = int(ms)


@pytest.fixture()
def controls(qapp):
    fake = FakePlayer()
    ctrl = PlaybackControls(player=fake)
    ctrl.fake = fake
    return ctrl


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ui.db")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    connection = db.connect()
    yield connection
    connection.close()


@pytest.fixture()
def make_win(qapp, conn, monkeypatch):
    """Dashboard windows whose playback backend is a recording fake.

    The real QMediaPlayer must not be dragged into an offscreen test run, and
    the fake lets us assert exactly which file each window tried to open.
    """
    created = []

    def factory():
        fake = FakePlayer()
        monkeypatch.setattr(dashboard_mod, "Player", lambda parent=None: fake)
        win = Dashboard(conn=conn, persist_settings=False)
        win.timer.stop()
        win.fake_player = fake
        created.append(win)
        return win

    yield factory
    for win in created:
        win.timer.stop()
        win.close()


# --- loading a job's media --------------------------------------------------


def test_starts_disabled_with_nothing_loaded(controls):
    assert controls.has_source is False
    assert controls.btn_play.isEnabled() is False
    assert controls.btn_stop.isEnabled() is False
    assert controls.seek_bar.isEnabled() is False
    assert controls.lbl_time.text() == "0:00 / --:--"
    # volume stays available regardless of whether a file is loaded
    assert controls.volume_slider.isEnabled() is True


def test_load_opens_the_job_file_and_enables_the_transport(controls, tmp_path):
    media = tmp_path / "lecture.mp4"
    media.write_bytes(b"fake media")

    controls.load(7, str(media), [])

    assert controls.fake.opened == [str(media)]
    assert controls.has_source is True
    assert controls.btn_play.isEnabled() is True
    assert controls.seek_bar.isEnabled() is True
    assert controls.btn_stop.isEnabled() is True
    assert "lecture.mp4" in controls.lbl_status.text()


def test_load_reports_a_missing_source_file(controls, tmp_path):
    controls.load(3, str(tmp_path / "gone.wav"), [])

    assert controls.has_source is False
    assert controls.fake.opened == []
    assert controls.btn_play.isEnabled() is False
    assert "missing" in controls.lbl_status.text().lower()


def test_load_without_a_path_keeps_controls_off(controls):
    controls.load(None, "", [])
    assert controls.has_source is False
    assert controls.btn_play.isEnabled() is False
    assert controls.fake.opened == []


def test_reselecting_the_same_job_does_not_restart_playback(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")

    controls.load(1, str(media), [])
    controls.fake.position_ms = 4000
    controls.load(1, str(media), [])  # the 1s queue refresh does this constantly

    assert controls.fake.opened == [str(media)]  # opened exactly once
    assert controls.fake.stopped == 0


def test_selecting_a_different_job_switches_the_source(controls, tmp_path):
    first = tmp_path / "one.wav"
    second = tmp_path / "two.mp3"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    controls.load(1, str(first), [])
    controls.load(2, str(second), [])

    assert controls.fake.opened == [str(first), str(second)]


# --- transport --------------------------------------------------------------


def test_play_pause_and_stop_wire_through(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])

    controls.toggle_play()
    assert controls.fake.played == 1
    assert controls.btn_play.text() == "⏸  Pause"

    controls.toggle_play()
    assert controls.fake.paused == 1
    assert controls.btn_play.text() == "▶  Play"

    controls.fake.position_ms = 3000
    controls.stop()
    assert controls.fake.stopped == 1
    assert controls.seek_bar.value() == 0
    assert controls.lbl_time.text().startswith("0:00")


def test_play_does_nothing_without_a_source(controls):
    controls.toggle_play()
    assert controls.fake.played == 0


def test_position_and_duration_drive_the_slider_and_clock(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])

    controls.fake.duration_ms_changed.emit(95_000)
    controls.fake.position_ms_changed.emit(65_000)

    assert controls.seek_bar.maximum() == 95_000
    assert controls.seek_bar.value() == 65_000
    assert controls.lbl_time.text() == "1:05 / 1:35"


def test_dragging_the_seek_bar_seeks_once_on_release(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])
    controls.fake.duration_ms_changed.emit(30_000)

    controls._begin_seek()
    controls.seek_bar.setValue(12_000)
    assert controls.fake.seeks == []  # no seek storm while dragging

    controls._end_seek()
    assert controls.fake.seeks == [12_000]


def test_position_updates_do_not_fight_an_in_progress_drag(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])
    controls.fake.duration_ms_changed.emit(30_000)

    controls._begin_seek()
    controls.seek_bar.setValue(5_000)
    controls.fake.position_ms_changed.emit(20_000)  # player keeps playing

    assert controls.seek_bar.value() == 5_000  # the user's drag wins


def test_seek_issued_before_loading_is_applied_once_the_duration_arrives(controls, tmp_path):
    """Dragging the bar while the file is still loading must not be dropped.

    Seeking a source with no known duration silently snaps to 0, so the
    request is held and replayed on ``durationChanged``.
    """
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])
    assert controls.seek_bar.maximum() == 0  # still loading

    controls.seek_to(12_000)
    assert controls.fake.seeks == []  # nothing seekable yet
    assert controls.lbl_time.text() == "0:12 / --:--"

    controls.fake.duration_ms_changed.emit(30_000)

    assert controls.fake.seeks == [12_000]
    assert controls.seek_bar.value() == 12_000
    assert controls.lbl_time.text() == "0:12 / 0:30"


def test_stale_position_events_do_not_overwrite_a_queued_seek(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])
    controls.seek_to(12_000)

    # the not-yet-seekable player reports a position from before the request
    controls.fake.position_ms_changed.emit(250)
    assert controls._pending_seek == 12_000

    controls.fake.duration_ms_changed.emit(30_000)
    assert controls.fake.seeks == [12_000]
    assert controls.lbl_time.text() == "0:12 / 0:30"


def test_stop_discards_a_queued_seek(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])
    controls.seek_to(12_000)

    controls.stop()
    controls.fake.duration_ms_changed.emit(30_000)

    assert controls.fake.seeks == []
    assert controls.seek_bar.value() == 0


def test_seek_relative_clamps_to_the_media(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])
    controls.fake.duration_ms_changed.emit(10_000)

    controls.fake.position_ms = 2_000
    controls.seek_relative(-5_000)
    controls.fake.position_ms = 9_500
    controls.seek_relative(5_000)

    assert controls.fake.seeks == [0, 10_000]


# --- volume / mute ----------------------------------------------------------


def test_volume_slider_sets_player_volume(controls):
    controls.volume_slider.setValue(35)
    assert controls.fake.volume == 35
    assert "35%" in controls.volume_slider.toolTip()


def test_mute_button_toggles_the_player(controls):
    controls.btn_mute.setChecked(True)
    assert controls.fake.muted is True
    assert controls.btn_mute.text() == "Muted"

    controls.btn_mute.setChecked(False)
    assert controls.fake.muted is False
    assert controls.btn_mute.text() == "Mute"


def test_mute_shortcut_toggle_flips_the_button(controls):
    controls.toggle_mute()
    assert controls.fake.muted is True
    controls.toggle_mute()
    assert controls.fake.muted is False


# --- errors -----------------------------------------------------------------


def test_player_error_is_shown_to_the_user(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])
    assert controls.lbl_error.isHidden() is True

    controls.fake.player_error.emit("cannot decode stream")

    assert controls.lbl_error.text() == "cannot decode stream"
    assert controls.lbl_error.isHidden() is False


def test_loading_another_job_clears_a_previous_error(controls, tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"media")
    controls.load(1, str(media), [])
    controls.fake.player_error.emit("boom")

    other = tmp_path / "other.wav"
    other.write_bytes(b"media")
    controls.load(2, str(other), [])

    assert controls.lbl_error.text() == ""
    assert controls.lbl_error.isHidden() is True


# --- caption highlighting ---------------------------------------------------


def _preview_with(ctrl, lines):
    preview = QPlainTextEdit()
    preview.setPlainText("\n".join(lines))
    ctrl.attach_preview(preview)
    return preview


def test_highlight_follows_the_playhead(controls):
    preview = _preview_with(controls, ["1", "t0", "first line", "", "2", "t1", "second line", ""])
    controls.set_spans([(0, 1000, 2), (1000, 2000, 6)])

    controls.fake.position_ms_changed.emit(500)
    selections = preview.extraSelections()
    assert len(selections) == 1
    assert selections[0].cursor.blockNumber() == 2
    assert selections[0].cursor.block().text() == "first line"

    controls.fake.position_ms_changed.emit(1500)
    selections = preview.extraSelections()
    assert selections[0].cursor.block().text() == "second line"


def test_highlight_clears_when_the_playhead_leaves_every_caption(controls):
    preview = _preview_with(controls, ["1", "t0", "only line", ""])
    controls.set_spans([(0, 1000, 2)])

    controls.fake.position_ms_changed.emit(500)
    assert preview.extraSelections() != []

    controls.fake.position_ms_changed.emit(4000)
    assert preview.extraSelections() == []


def test_highlight_survives_a_span_pointing_past_the_document(controls):
    preview = _preview_with(controls, ["1", "t0", "only line", ""])
    controls.set_spans([(0, 1000, 999)])

    controls.fake.position_ms_changed.emit(500)  # must not raise

    assert preview.extraSelections() == []


def test_stop_clears_the_highlight(controls):
    preview = _preview_with(controls, ["1", "t0", "only line", ""])
    controls.set_spans([(0, 1000, 2)])
    controls.fake.position_ms_changed.emit(500)
    assert preview.extraSelections() != []

    controls.stop()
    assert preview.extraSelections() == []


# --- dashboard integration --------------------------------------------------


def _seed_done_job(conn, tmp_path, name="clip.mp4"):
    src = tmp_path / name
    src.write_bytes(b"not real media")
    job_id = worker.enqueue(conn, str(src), model_size="tiny.en")
    conn.execute(
        "UPDATE jobs SET status = 'done', started_at = ?, finished_at = ? WHERE id = ?",
        ("2026-01-01 10:00:00", "2026-01-01 10:00:05", job_id),
    )
    conn.commit()
    worker._store_captions(
        conn,
        job_id,
        [
            Block([Word("Hello", 0.0, 0.4), Word("world.", 0.5, 1.0)]),
            Block([Word("Second", 1.5, 1.8), Word("line.", 1.9, 2.4)]),
        ],
    )
    return job_id, src


def test_dashboard_plays_the_selected_jobs_own_file(make_win, conn, tmp_path):
    job_id, src = _seed_done_job(conn, tmp_path)
    win = make_win()

    win.table.selectRow(0)
    win.update_preview()

    assert win.player_controls.has_source is True
    assert win.player_controls.source_path == str(src)
    assert win.fake_player.opened == [str(src)]


def test_dashboard_spans_line_up_with_the_rendered_captions(make_win, conn, tmp_path):
    job_id, _ = _seed_done_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()

    text, spans = win._build_preview(job_id)
    blocks = text.split("\n")
    assert len(spans) == 2
    for start, end, block_no in spans:
        # the highlighted block must be the caption body, not its index/timing
        assert blocks[block_no - 1].startswith("00:")
        assert not blocks[block_no].startswith("0")

    # first caption starts at 0 ms, so the playhead highlights it immediately
    assert spans[0][0] == 0
    win.fake_player.position_ms_changed.emit(300)
    selections = win.preview.extraSelections()
    assert len(selections) == 1
    assert selections[0].cursor.block().text() == "Hello world."

    win.fake_player.position_ms_changed.emit(2000)
    selections = win.preview.extraSelections()
    assert selections[0].cursor.block().text() == "Second line."


def test_dashboard_clears_playback_when_the_queue_empties(make_win, conn, tmp_path):
    job_id, _ = _seed_done_job(conn, tmp_path)
    win = make_win()
    win.table.selectRow(0)
    win.update_preview()
    assert win.player_controls.has_source is True

    win._remove_job(job_id)

    assert win.player_controls.has_source is False
    assert win.player_controls.btn_play.isEnabled() is False


def test_dashboard_play_shortcut_reports_when_nothing_is_selected(make_win, conn):
    win = make_win()
    win.toggle_play()  # empty queue: must not raise, must explain itself
    assert win.player_controls.has_source is False


def test_dashboard_queues_mp3_and_mp4_for_captioning(make_win, conn, tmp_path):
    win = make_win()
    audio = tmp_path / "voice note.mp3"
    video = tmp_path / "lecture.mp4"
    audio.write_bytes(b"x")
    video.write_bytes(b"x")

    assert win.add_paths([str(audio), str(video)]) == 2

    rows = conn.execute("SELECT file_name, status FROM jobs ORDER BY id").fetchall()
    assert [r["file_name"] for r in rows] == ["voice note.mp3", "lecture.mp4"]
    assert all(r["status"] == "queued" for r in rows)
