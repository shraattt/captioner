"""Live caption sessions: model choice, device choice, start and stop.

The overlay itself lives in ui/overlay.py; this mixin is the dashboard side of
it - which model to load, which device to capture, and which language (if any)
to translate the captions into.
"""


from PySide6.QtWidgets import (
    QInputDialog,
    QMessageBox,
)

from captioner import config, models
from captioner.translate import LANGUAGES, language_name


class LiveMixin:
    """Starting, configuring and stopping live caption sessions."""

    #live captions (Phase 3)

    @property
    def LIVE_MODEL_CHOICES(self):
        """Built-in sizes plus any locally fine-tuned (Phase 4/5) models."""
        return models.live_choices()

    def _refresh_live_model_label(self) -> None:
        model = (
            self.settings.value("live_model", config.LIVE_MODEL_SIZE)
            if self.settings
            else config.LIVE_MODEL_SIZE
        )
        self.act_live_model.setText(f"Live &model: {model}")

    def choose_live_model(self) -> None:
        """Pick the live caption model (remembered; takes effect next session).

        On clean audio tiny.en is already the most accurate live option
        (measured in scripts/bench_live_wer.py); base/small help when the
        source is noisy or far from the mic, at the cost of some latency.
        """
        current = (
            self.settings.value("live_model", config.LIVE_MODEL_SIZE)
            if self.settings
            else config.LIVE_MODEL_SIZE
        )
        names = [f"{m}{'  (current)' if m == current else ''}" for m in self.LIVE_MODEL_CHOICES]
        name, ok = QInputDialog.getItem(
            self,
            "Live captions model",
            "Model for live captions:\n"
            "tiny = fastest, best on clear audio · base/small = sturdier on "
            "noise, a little slower",
            names,
            max(0, list(self.LIVE_MODEL_CHOICES).index(current))
            if current in self.LIVE_MODEL_CHOICES
            else 0,
            False,
        )
        if not ok:
            return
        model = self.LIVE_MODEL_CHOICES[names.index(name)]
        if self.settings is not None:
            self.settings.setValue("live_model", model)
        self._refresh_live_model_label()
        self.statusBar().showMessage(
            f"live model: {model} (takes effect on the next live session)", 5000
        )

    LIVE_TRANSLATION_OFF = "Off (keep the spoken language)"

    def _live_translate_to(self) -> str:
        """Target language for live captions, or "" when translation is off."""
        if self.settings is None:
            return config.LIVE_TRANSLATE_TO
        return self.settings.value("live_translate_to", config.LIVE_TRANSLATE_TO) or ""

    def _refresh_live_language_label(self) -> None:
        target = self._live_translate_to()
        label = f"Live translation: {language_name(target)}" if target else "Live translation: off"
        self.act_live_translate.setText(label)

    def choose_live_language(self) -> None:
        """Pick the language live captions are translated into, or turn it off.

        Only finished sentences are translated, on their own thread, so a
        session never stalls waiting for one. The first sentence in a new
        language loads that model (one pause); afterwards the cache answers.
        """
        current = self._live_translate_to()
        codes = [""] + sorted(LANGUAGES, key=lambda code: LANGUAGES[code])
        names = [
            self.LIVE_TRANSLATION_OFF if not code else f"{LANGUAGES[code]} ({code})"
            for code in codes
        ]
        index = codes.index(current) if current in codes else 0
        name, ok = QInputDialog.getItem(
            self,
            "Live translation",
            "Translate live captions into:Finished sentences appear translated under the captions.",
            names,
            index,
            False,
        )
        if not ok:
            return
        target = codes[names.index(name)]
        if self.settings is not None:
            self.settings.setValue("live_translate_to", target)
        self._refresh_live_language_label()
        message = f"live translation: {language_name(target)}" if target else "live translation off"
        self.statusBar().showMessage(f"{message} (takes effect on the next live session)", 5000)

    def _launch_live(self, device: dict) -> None:
        """Start the overlay + worker for *device* (shared by menu and hotkey)."""
        from ui.overlay import LiveOverlay, LiveWorker

        self.stop_live()  # replace any running session
        self._overlay = LiveOverlay()
        self._overlay.set_device_name(f"● {device['name']}")
        self._overlay.stop_requested.connect(self.stop_live)
        self._overlay.show()
        model = (
            self.settings.value("live_model", config.LIVE_MODEL_SIZE)
            if self.settings
            else config.LIVE_MODEL_SIZE
        )
        try:
            model = models.resolve(model)
        except ValueError as exc:
            self.statusBar().showMessage(f"{exc} — using {config.LIVE_MODEL_SIZE}", 6000)
            model = config.LIVE_MODEL_SIZE
        self._live_worker = LiveWorker(
            device["id"], model_size=model, translate_to=self._live_translate_to()
        )
        self._live_worker.state_ready.connect(self._overlay.update_state)
        self._live_worker.live_error.connect(self._overlay.show_error)
        self._live_worker.start()
        self.act_stop_live.setEnabled(True)
        self._live_device = device
        self.statusBar().showMessage(f"live captions on: {device['name']}", 5000)

    def _pick_live_device(self) -> dict | None:
        """Ask which audio source to caption; None means the user canceled."""
        from captioner.live import list_input_devices

        try:
            devices = list_input_devices()
        except Exception as exc:
            QMessageBox.warning(self, "Live captions", f"Could not list audio devices:\n{exc}")
            return None
        if not devices:
            QMessageBox.warning(
                self,
                "Live captions",
                "No microphone or loopback device was found.\n"
                "Check Windows sound settings, then try again.",
            )
            return None
        names = [d["name"] for d in devices]
        name, ok = QInputDialog.getItem(
            self, "Live captions", "Caption audio from:", names, 0, False
        )
        if not ok:
            return None
        return devices[names.index(name)]

    def start_live(self) -> None:
        """Pick an audio source (remembered for next time), then go live."""
        device = self._pick_live_device()
        if device is not None:
            self._launch_live(device)

    def stop_live(self) -> None:
        if self._live_worker is not None and self._live_worker.isRunning():
            self._live_worker.stop()
            self._live_worker.wait(3000)
        self._live_worker = None
        if self._overlay is not None:
            self._overlay.close_transcript_window()
            self._overlay.close()
            self._overlay = None
        if getattr(self, "act_stop_live", None) is not None:
            self.act_stop_live.setEnabled(False)
            if self.isVisible():
                self.statusBar().showMessage("live captions stopped", 3000)

    def toggle_live(self) -> None:
        """Global hotkey target: start with the remembered device, or stop."""
        if self._live_worker is not None and self._live_worker.isRunning():
            self.stop_live()
            return
        device = self._live_device
        if device is None:
            device = self._pick_live_device()  # first press: ask (we have focus)
            if device is None:
                return
        self._launch_live(device)
