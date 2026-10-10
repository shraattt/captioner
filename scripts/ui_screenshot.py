"""Render the dashboard with demo data and save a PNG screenshot.

Used to visually verify the UI without a display: builds a **throwaway** queue
with a couple of finished jobs, constructs the Dashboard under Qt's native QPA
(window kept off-screen so real fonts are used), and grabs the window to
screenshots/dashboard.png.

The temp database matters: an earlier version connected to the real
``data/captioner.db`` and left demo rows behind in the user's queue.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.pop("QT_QPA_PLATFORM", None)  # native windows QPA: real fonts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from captioner import config, db, worker  # noqa: E402
from captioner.segmenter import Block, Word  # noqa: E402
from ui.dashboard import Dashboard  # noqa: E402

DEMO_JOBS = [
    ("interview-part1.mp4", ["Welcome back to the show.", "Tell us about the project."]),
    ("lecture-07-notes.mp4", ["Neural attention maps each query", "onto a weighted mix of values."]),
]


def main() -> None:
    # Keep the sandbox inside the project instead of %TEMP%: Windows temp paths
    # contain the user's name, and these screenshots go in the README. A
    # project-relative path shows up in the UI as "<project>/data/bench/...".
    bench = Path(__file__).resolve().parent.parent / "data" / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(bench)
    tempfile.tempdir = str(bench)

    with tempfile.TemporaryDirectory(prefix="screenshot-") as tmp:
        tmp_path = Path(tmp)
        # Point the whole app at the sandbox before anything connects.
        config.DB_PATH = tmp_path / "screenshot.db"
        config.OUTPUT_DIR = tmp_path / "outputs"
        config.UPLOADS_DIR = tmp_path / "uploads"
        config.ensure_dirs()

        conn = db.connect()
        for name, lines in DEMO_JOBS:
            src = tmp_path / name
            src.write_bytes(b"demo")
            job_id = worker.enqueue(conn, str(src), model_size="small.en")
            conn.execute(
                "UPDATE jobs SET status='done', started_at=?, finished_at=? WHERE id=?",
                ("2026-01-05 10:00:00", "2026-01-05 10:02:30", job_id),
            )
            conn.commit()
            blocks = [
                Block([
                    Word(line.split()[0], 0.0, 0.5),
                    Word(" ".join(line.split()[1:]), 0.5, len(line) / 20 + 1.0),
                ])
                for line in lines
            ]
            worker._store_captions(conn, job_id, blocks)

        app = QApplication.instance() or QApplication([])
        win = Dashboard(conn=conn, persist_settings=False)
        win.timer.stop()
        win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        win.resize(1280, 860)
        win.show()
        app.processEvents()
        time.sleep(0.3)
        app.processEvents()

        out_dir = Path(__file__).resolve().parent.parent / "screenshots"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / "dashboard.png"
        win.grab().save(str(out), "PNG")
        # Also capture the selected/playback state, like a user clicking a row.
        if win.table.rowCount() > 1:
            win.table.selectRow(0)
            app.processEvents()
            win.grab().save(str(out_dir / "dashboard-playback.png"), "PNG")
        conn.close()
        print(out)


if __name__ == "__main__":
    main()
