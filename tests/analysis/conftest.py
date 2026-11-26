"""Shared helpers for the Qt window tests.

Here rather than copied into each file: `_destroy` currently guards two
files, and sixteen more still call `show()` then `close()` without it.
Whoever extends it should have one place to change.
"""
import os
import pathlib
import sys

# Before any Qt import anywhere in this directory. Two window-building
# files used to lack this and passed only because collection happened to
# import a module that set it first; alone they fell through to the real
# `windows` platform and a pixel probe measured 0px.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# So `from qt_window_teardown import destroy_window` resolves to the module
# beside the tests. Without it only the repo root is on sys.path, and the
# root's own conftest.py shadows anything named conftest here -- which is
# why the helper lives in a separate, uniquely named module rather than in
# this file.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
