"""Every subplot's x-axis must be labelled by _set_xaxis_ticks, or blanked.

The subplots are X-linked to the main chart, so their tick POSITIONS align
for free. Their tick TEXT does not: a subplot _set_xaxis_ticks forgets to
call does not fall back to blank labels, it falls back to pyqtgraph's own
auto ticks -- which label the x-axis with the raw bar index, printing
"4320" in the place a time belongs. DCVD read that way from v0.44.0 to
v0.45.2 because it was added to the layout and not to that function.

The failure is silent and per-panel, so this pins the invariant for all of
them rather than for the one that broke.
"""

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

# plot attribute -> the indicator key that reveals it
SUBPLOTS = {
    "_plot_v":    "vol",
    "_plot_kd":   "kd",
    "_plot_cvd":  "cvd",
    "_plot_adi":  "adi",
    "_plot_dcvd": "dcvd",
}


@pytest.fixture(scope="module")
def viewer():
    from PyQt6.QtWidgets import QApplication
    from analysis.trade_viewer_qt import TradeViewerQt

    app = QApplication.instance() or QApplication([])
    v = TradeViewerQt()
    yield v
    v.close()
    app.processEvents()


@pytest.fixture
def klines():
    t = pd.date_range("2026-10-19 09:30", periods=300, freq="1min")
    return pd.DataFrame({"time_key": t.strftime("%Y-%m-%d %H:%M:%S")})


def tick_text(plot):
    """The tick labels pyqtgraph will actually paint on the bottom axis."""
    levels = plot.getAxis("bottom")._tickLevels
    if not levels:
        return None                      # never set -> auto ticks
    return [s for _, s in levels[0]]


class TestEverySubplotIsBlanked:
    @pytest.mark.parametrize("attr,key", sorted(SUBPLOTS.items()))
    def test_no_subplot_shows_bar_indices(self, viewer, klines, attr, key):
        cb = viewer._ind_checks[key]
        was = cb.isChecked()
        cb.setChecked(True)
        try:
            viewer._set_xaxis_ticks(klines)
            labels = tick_text(getattr(viewer, attr))
        finally:
            cb.setChecked(was)

        assert labels is not None, (
            f"{attr} was never given ticks, so pyqtgraph auto-labels it "
            f"with the bar index -- add it to _set_xaxis_ticks")
        assert set(labels) == {""}, f"{attr} shows text: {labels[:4]}"

    def test_positions_match_the_main_chart(self, viewer, klines):
        for key in SUBPLOTS.values():
            viewer._ind_checks[key].setChecked(True)
        viewer._set_xaxis_ticks(klines)

        want = [x for x, _ in viewer._plot_c.getAxis("bottom")._tickLevels[0]]
        for attr in SUBPLOTS:
            got = [x for x, _ in
                   getattr(viewer, attr).getAxis("bottom")._tickLevels[0]]
            assert got == want, attr


class TestTheMainChartStillShowsTimes:
    def test_intraday_labels_are_times_not_indices(self, viewer, klines):
        viewer._set_xaxis_ticks(klines)
        labels = tick_text(viewer._plot_c)
        assert labels and any(":" in s for s in labels), labels
        assert not any(s.isdigit() for s in labels), labels
