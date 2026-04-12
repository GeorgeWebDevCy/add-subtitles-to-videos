from __future__ import annotations

from multiprocessing import freeze_support
import os
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
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


def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _branding_icon_path() -> Path | None:
    icon_path = _runtime_root() / "assets" / "branding" / "subtitle-foundry-icon.ico"
    if icon_path.exists():
        return icon_path
    return None


def run() -> int:
    freeze_support()
    _ensure_standard_streams()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Subtitle Foundry")
    app.setStyle("Fusion")
    app.setStyleSheet(application_stylesheet())
    icon_path = _branding_icon_path()
    if icon_path is not None:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            app.setWindowIcon(icon)

    window = MainWindow()
    if not app.windowIcon().isNull():
        window.setWindowIcon(app.windowIcon())
    window.showMaximized()
    return app.exec()
