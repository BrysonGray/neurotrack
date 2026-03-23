"""
Qt bootstrap utilities for the visualization package.

Provides lazy Qt/Matplotlib imports, environment detection helpers,
and QApplication lifecycle management used by ortho_viewer.
"""

import importlib
import os

from IPython import get_ipython


_QT_APP_INSTANCE = None


def _is_jupyter_notebook() -> bool:
    ip = get_ipython()
    return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"


def _has_gui_display() -> bool:
    """Check whether a desktop GUI display is available for Qt windows."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _try_import_ui_dependencies():
    """Import Qt and Matplotlib Qt canvas dependencies.

    Raises
    ------
    RuntimeError
        If required Qt dependencies are missing or fail to import.
    """
    try:
        global _QT_APP_INSTANCE
        qt_widgets = importlib.import_module("qtpy.QtWidgets")
        qt_core = importlib.import_module("qtpy.QtCore")
        if qt_widgets.QApplication.instance() is None:
            _QT_APP_INSTANCE = qt_widgets.QApplication([])
        else:
            _QT_APP_INSTANCE = qt_widgets.QApplication.instance()
        backend = importlib.import_module("matplotlib.backends.backend_qt5agg")
        return (
            qt_widgets.QApplication,
            qt_widgets.QWidget,
            qt_widgets.QDialog,
            qt_widgets.QVBoxLayout,
            qt_widgets.QHBoxLayout,
            qt_widgets.QPushButton,
            qt_widgets.QLabel,
            qt_widgets.QSlider,
            qt_widgets.QComboBox,
            qt_core.Qt,
            backend.FigureCanvasQTAgg,
        )
    except Exception as exc:
        raise RuntimeError(
            "Qt UI dependencies are not available. Install with: "
            "pip install QtPy PyQt5 matplotlib"
        ) from exc


def _ensure_qapplication():
    """Ensure a QApplication exists before creating QWidget objects."""
    QApplication, *_ = _try_import_ui_dependencies()
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
