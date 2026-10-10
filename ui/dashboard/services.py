"""Optional background services the window can start and stop.

Today that means the HTTP API and its browser UI, which `python cli.py api`
also serves as a standalone process. Running it in a daemon thread here means
you can drive the queue from a phone or a script without keeping a console
open.
"""

import logging
import threading

from PySide6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class ServiceMixin:
    """Toggles for the long-running services the app can host."""

    def api_url(self) -> str:
        """Where the API is listening, or an empty string when it is off."""
        if self._api_server is None:
            return ""
        host, port = self._api_server.server_address[:2]
        return f"http://{host}:{port}/"

    def _refresh_api_action(self) -> None:
        running = self._api_server is not None
        self.act_api.setText("Stop web &API + UI" if running else "Toggl&e web API + UI")

    def toggle_api(self) -> None:
        """Start or stop the HTTP API on this machine."""
        if self._api_server is not None:
            self.stop_api()
            return
        from captioner import api

        try:
            server = api.create_server()
        except SystemExit as exc:
            # A non-loopback bind without CAPTIONER_API_TOKEN: the API can read
            # local files and return transcripts, so it refuses to start.
            QMessageBox.warning(self, "Web API", str(exc))
            return
        except OSError as exc:
            QMessageBox.warning(self, "Web API", f"could not bind the API port: {exc}")
            return
        thread = threading.Thread(target=server.serve_forever, name="captioner-api", daemon=True)
        thread.start()
        self._api_server, self._api_thread = server, thread
        self._refresh_api_action()
        url = self.api_url()
        logger.info("web API listening on %s", url)
        self.statusBar().showMessage(f"web API + UI at {url} (docs at {url}openapi.json)", 15000)

    def stop_api(self) -> None:
        server, self._api_server = self._api_server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        self._api_thread = None
        self._refresh_api_action()
        self.statusBar().showMessage("web API stopped", 4000)

    def close_services(self) -> None:
        """window closeEvent hook: never leave a listener behind the window."""
        if self._api_server is not None:
            self.stop_api()
