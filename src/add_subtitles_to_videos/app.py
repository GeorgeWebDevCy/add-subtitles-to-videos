from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from .config import APP_NAME
from .ui.main_window import MainWindow
from .ui.theme import application_stylesheet

_NULL_STREAMS = []


def _ensure_standard_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        if getattr(sys, stream_name) is not None:
            continue

        stream = open(
            os.devnull,
            "w",
            encoding="utf-8",
            buffering=1,
            errors="replace",
        )
        _NULL_STREAMS.append(stream)
        setattr(sys, stream_name, stream)


def run() -> int:
    _ensure_standard_streams()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Subtitle Foundry")
    app.setStyle("Fusion")
    app.setStyleSheet(application_stylesheet())

    window = MainWindow()
    window.showFullScreen()
    return app.exec()
