"""The opening card: CAPTIONER, "project by sharat", then the window.

A title card is not decoration here. Building the dashboard reads the queue,
restores the geometry and touches the disk, and on a cold machine that is a
visible pause on a blank screen. The card covers that pause and says whose app
it is; it is dismissed the moment the window is ready, after a short minimum so
it reads as an opening rather than a flicker.

Two ways to skip it, both used by the app itself: ``--self-test`` (a sanity run
should not wait for an animation) and ``CAPTIONER_NO_SPLASH=1`` (screenshots,
recordings of the window, a user who does not want it). ``--no-splash`` on the
command line does the same thing for a packaged build.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .constants import APP_NAME, APP_VERSION
from .theme import current_palette

CARD_WIDTH = 480
CARD_HEIGHT = 264
# How long the card stays up at minimum. Long enough to read, short enough that
# a fast start does not feel slower than it was.
MINIMUM_MS = 1100
FADE_MS = 240
# The sweep along the bottom: a violet segment crossing a dim track, the one
# moving thing on the card (there is no real progress to report yet).
PULSE_MS = 1200
PULSE_FRACTION = 0.3


def enabled(environ=None) -> bool:
    """Should this start show the opening card?

    ``--self-test`` and ``--no-splash`` are handled by the caller (they are
    stripped from the arguments first); this reads the environment, where
    ``CAPTIONER_NO_SPLASH`` is the switch for scripts and recordings.
    """
    environ = os.environ if environ is None else environ
    value = str(environ.get("CAPTIONER_NO_SPLASH", "")).strip().lower()
    return value not in ("1", "true", "yes", "on")


class Splash(QWidget):
    """A frameless violet-glass card: icon, CAPTIONER, project by sharat."""

    def __init__(self, *, minimum_ms: int = MINIMUM_MS, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(CARD_WIDTH, CARD_HEIGHT)
        # One name for the widget tests and screen readers; the window title is
        # never seen (no frame, no taskbar entry) but is what a screen reader
        # announces for the window.
        self.setWindowTitle(APP_NAME)
        self._pulse = 0.0
        self._started = time.monotonic()
        self._minimum_ms = max(0, int(minimum_ms))
        self._fade: QPropertyAnimation | None = None

        palette = current_palette()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 28, 36, 34)
        layout.setSpacing(0)
        layout.addStretch(1)

        icon = QLabel(self)
        icon.setObjectName("splashIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        pixmap = QPixmap(str(_icon_path()))
        if not pixmap.isNull():
            icon.setPixmap(
                pixmap.scaled(
                    54, 54,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        layout.addWidget(icon)
        layout.addSpacing(12)

        self.title = QLabel(APP_NAME.upper(), self)
        self.title.setObjectName("splashTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        font = QFont(self.title.font())
        font.setPointSizeF(26)
        font.setWeight(QFont.Weight.Bold)
        # Wide tracking is what turns a product name into a title card.
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 8)
        self.title.setFont(font)
        self.title.setStyleSheet(f"color: {palette.accent_bright}; background: transparent;")
        layout.addWidget(self.title)
        layout.addSpacing(4)

        self.subtitle = QLabel("project by sharat", self)
        self.subtitle.setObjectName("splashSubtitle")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sub_font = QFont(self.subtitle.font())
        sub_font.setPointSizeF(11.5)
        self.subtitle.setFont(sub_font)
        self.subtitle.setStyleSheet(
            f"color: {palette.text_dim}; background: transparent;"
        )
        layout.addWidget(self.subtitle)

        layout.addStretch(1)
        self.version = QLabel(f"v{APP_VERSION}", self)
        self.version.setObjectName("splashVersion")
        self.version.setAlignment(Qt.AlignmentFlag.AlignRight)
        version_font = QFont(self.version.font())
        version_font.setPointSizeF(9)
        self.version.setFont(version_font)
        self.version.setStyleSheet(
            f"color: {palette.text_faint}; background: transparent;"
        )
        layout.addWidget(self.version)

        self._sweep = QVariantAnimation(self)
        self._sweep.setStartValue(0.0)
        self._sweep.setEndValue(1.0)
        self._sweep.setDuration(PULSE_MS)
        self._sweep.setLoopCount(-1)
        self._sweep.valueChanged.connect(self._on_sweep)
        self._sweep.start()

    # -- painting ---------------------------------------------------------

    def _on_sweep(self, value) -> None:
        self._pulse = float(value)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        card = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        gradient = QLinearGradient(card.topLeft(), card.bottomRight())
        gradient.setColorAt(0.0, QColor(palette.bg))
        gradient.setColorAt(1.0, QColor(palette.surface_solid))
        border = QColor(palette.accent)
        border.setAlpha(140)
        painter.setBrush(gradient)
        painter.setPen(QPen(border, 1.2))
        painter.drawRoundedRect(card, 18.0, 18.0)

        # The sweep sits at the very bottom, inside the card's rounded corners.
        left, right = 40.0, CARD_WIDTH - 40.0
        width = right - left
        track = QColor(palette.text)
        track.setAlpha(26)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(left, CARD_HEIGHT - 24, width, 3.0), 1.5, 1.5)
        segment = width * PULSE_FRACTION
        travel = width - segment
        painter.setBrush(QColor(palette.accent_bright))
        painter.drawRoundedRect(
            QRectF(left + travel * self._pulse, CARD_HEIGHT - 24, segment, 3.0),
            1.5, 1.5,
        )
        painter.end()

    # -- lifecycle --------------------------------------------------------

    def center_on_screen(self) -> None:
        """Put the card in the middle of the primary screen."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:  # pragma: no cover - no screen to centre on
            return
        middle = screen.availableGeometry().center()
        self.move(middle.x() - self.width() // 2, middle.y() - self.height() // 2)

    def finish(self, reveal) -> None:
        """Fade out after the minimum, then call ``reveal`` (which shows home).

        The dashboard is built while the card is on screen, so by the time this
        is called the wait is usually just the minimum: the card is a title, not
        a loading spinner. ``reveal`` runs once, when the card is gone - showing
        the window underneath a fading card would flash the desktop between
        them.
        """
        waited = int((time.monotonic() - self._started) * 1000)
        QTimer.singleShot(max(0, self._minimum_ms - waited),
                          lambda: self._fade_out(reveal))

    def _fade_out(self, reveal) -> None:
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(FADE_MS)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.InCubic)

        def gone() -> None:
            self.hide()
            self.close()
            reveal()

        animation.finished.connect(gone)
        # The animation is parented, but staying referenced until it finishes
        # keeps it out of reach of an early garbage collection.
        self._fade = animation
        animation.start()


def _icon_path() -> Path:
    """captioner.ico beside the bundle - the same lookup the window does."""
    return Path(__file__).resolve().parents[2] / "captioner.ico"
