"""The viewer and the heatmap have to fit side by side.

Both windows floor their own width at their widest toolbar row, so that
no control can ever be pushed into an overflow popup where it looks like
it has gone missing. That floor is a real constraint on a desktop: two
windows at 1337 and 1289 px need 2626 px, and the screen this was
specified against has 2560. They overlapped.

Nothing in either window's own tests could see that, because it is a
property of the PAIR. So it is pinned here, and the three ways it has
already been broken are pinned with it:

  * a spin box padded by more than the overlap it had to clear -- 50 px
    where 22 was enough, which with six boxes on one row cost 168 px
  * a toolbar row carrying more control groups than the others
  * two checkboxes (Opt Delta-Flow, Cumulative, 93 px each) added to the
    viewer's ALREADY widest row, which took its floor to 1473 against the
    heatmap's 1140 and made side-by-side impossible at any size. Measured
    on screen on 2026-11-24: the two windows overlapped by 193 px and
    neither could be narrowed, because both were at their minimum.

PLATFORM, and why this runs in a subprocess. Offscreen has no font
directory, so Qt falls back to a face about 30% wider and every width
here is wrong -- offscreen the same two windows measure 1733 + 1488 =
3221 px. This used to skip on `QT_QPA_PLATFORM` being set, which made the
guard dead in both collection orders:

  * alphabetically, tests/analysis/test_opt_flow_view.py is imported
    first and does os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    at module scope, so this module skipped on every full run -- which is
    how the third regression above shipped past an assertion written
    exactly for it
  * forced to collect first, the env was still offscreen by the time any
    test ran (pytest imports every module before running any), so it
    failed on the inflated numbers instead

A module-scope env var set by a sibling module is not something a skip
condition can read its way out of. The measurement therefore happens in a
child process with the platform forced, where neither collection order
nor anyone's setdefault can reach it.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6.QtWidgets")

_REPO = pathlib.Path(__file__).resolve().parent.parent.parent

# The screen this requirement was written against. A machine with less
# than this genuinely cannot tile the two windows, and should say so
# rather than skip.
SCREEN_PX = 2560


def _probe() -> dict:
    """Both windows' floors, measured natively in a child process.

    Returns {"viewer": {...}, "heatmap": {...}} with floor/actual/rows.
    Skips -- rather than fails -- only when the child cannot get a native
    Qt platform at all (a headless machine), which is a different fact
    from the windows not fitting.
    """
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "windows" if sys.platform == "win32" else "xcb"
    proc = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--probe"],
        capture_output=True, text=True, env=env, cwd=str(_REPO), timeout=300)
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    if "platform plugin" in (proc.stderr or ""):
        pytest.skip(f"no native Qt platform in this environment: "
                    f"{proc.stderr.strip().splitlines()[:2]}")
    raise AssertionError(
        f"width probe produced no measurement (exit {proc.returncode}).\n"
        f"stdout tail: {(proc.stdout or '')[-800:]}\n"
        f"stderr tail: {(proc.stderr or '')[-800:]}")


@pytest.fixture(scope="module")
def floors():
    return _probe()


def test_the_two_windows_fit_side_by_side(floors):
    v = floors["viewer"]
    h = floors["heatmap"]
    assert v["floor"] == v["actual"] and h["floor"] == h["actual"], (
        "a window did not shrink to its own stated minimum, so these "
        "numbers do not describe what a user can do")
    assert v["floor"] + h["floor"] <= SCREEN_PX, (
        f"viewer {v['floor']} px + heatmap {h['floor']} px = "
        f"{v['floor'] + h['floor']} px, which does not fit {SCREEN_PX} px. "
        f"Whatever was just added to a toolbar row has raised a window's "
        f"floor; put it on a shorter row.\n"
        f"  viewer rows:  {v['rows']}\n"
        f"  heatmap rows: {h['rows']}")


def test_no_single_row_dominates_its_window(floors):
    """A floor is the widest row, so one overloaded row costs the window.

    Not a style rule: it is the failure mode both windows hit. The
    heatmap carried five control groups on one row and ten on the other,
    and the crowded row alone set a floor 165 px above what the window
    needed.

    This is the weaker of the two guards and it did NOT catch the
    2026-11-24 regression -- a row already the widest by 124 px growing to
    308 px wider is still nowhere near this ratio. The sum above is what
    catches that; this one catches the shape that makes a window wide for
    no reason.
    """
    rows = sorted(floors["heatmap"]["rows"], reverse=True)
    assert len(rows) >= 3, f"expected the toolbar to be split, got {rows}"
    assert rows[0] - rows[1] <= 0.55 * rows[0], (
        f"the widest row is {rows[0]} px against {rows[1]} px for the "
        f"next one; that row alone is setting the window's floor")


# ── the child process ────────────────────────────────────────────────────

def _measure(qapp, cls, rows_of):
    w = cls()
    w.show()
    for _ in range(40):
        qapp.processEvents()
    # Ask for less than any floor, then read back what Qt allowed: a
    # minimumWidth nobody enforces would describe a layout the user can
    # get out of, and is not the constraint being measured.
    w.resize(300, 700)
    for _ in range(15):
        qapp.processEvents()
    out = {"floor": w.minimumWidth(), "actual": w.width(),
           "rows": rows_of(w)}
    w.close()
    for _ in range(10):
        qapp.processEvents()
    return out


def _main() -> None:
    sys.path.insert(0, str(_REPO))
    from PyQt6.QtWidgets import QApplication, QToolBar

    app = QApplication.instance() or QApplication([])
    from analysis.trade_viewer_qt import TradeViewerQt
    from analysis.liq_hm_window import LiqHmWindow

    result = {
        "platform": app.platformName(),
        "viewer": _measure(
            app, TradeViewerQt,
            lambda w: [tb.sizeHint().width() for tb in w._toolbar_rows]),
        "heatmap": _measure(
            app, LiqHmWindow,
            lambda w: [tb.sizeHint().width()
                       for tb in w._toolbar.findChildren(QToolBar)]),
    }
    # flush: this is the only channel back to the test, and the exit below
    # does not run atexit handlers.
    print(json.dumps(result), flush=True)
    # The viewer opens an OpenD quote context whose threads keep a clean
    # interpreter exit waiting -- see tests/analysis/test_opt_flow_view.py.
    os._exit(0)


if __name__ == "__main__":
    _main()
