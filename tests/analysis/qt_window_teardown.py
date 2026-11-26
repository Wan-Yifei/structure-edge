"""Deterministic teardown for the Qt window tests.

A module rather than conftest.py because the repo root has its own
conftest.py, which sits ahead of this directory on sys.path -- so
`from conftest import destroy_window` imports the wrong file.
"""


def destroy_window(w):
    """Close a window AND destroy it, instead of leaving it to the GC.

    close() only hides. Without WA_DeleteOnClose the C++ object then lives
    until the collector takes the Python wrapper, at a moment nothing
    controls -- measured: after a plain close(), sip.isdeleted(w) is False.
    Two separate process aborts come out of that window:

      * the ViewBox inside a collected cycle is finalised before an
        InfiniteLine that points at it, and pyqtgraph's unguarded
        `self.getViewBox().size()` raises inside a C++ virtual. See
        analysis/pg_patches.py.
      * the wrapper is collected while Qt still holds a deleteLater event
        for it; delivering that event calls back into a wrapper with no
        Python thread state -- "PyEval_SaveThread ... the GIL is released".
        Already written down at test_absorption_marker.py:88, measured 3/3.

    WHAT IS NOT HERE, and why: an earlier version ended with
    `app.sendPostedEvents(None, QEvent.Type.DeferredDelete)`. Measured, it
    is both useless and harmful. Useless because WA_DeleteOnClose +
    close() + processEvents() has already destroyed this window -- the
    docstring that claimed "processEvents() alone does not deliver
    DeferredDelete" was simply wrong. Harmful because receiver=None is a
    GLOBAL flush: it was observed destroying a bystander widget this code
    never touched, which is precisely how you manufacture the second abort
    above for the sixteen files that do not use this helper.

    Do NOT move this into a qapp teardown either. By then the widgets are
    gone and delivering their queued events aborts the interpreter.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    w.close()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
