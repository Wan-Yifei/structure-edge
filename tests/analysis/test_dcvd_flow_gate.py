"""The flow gate drawn into the DCVD bars' opacity.

DCVD is a RATIO and says nothing about size: a 500-share bar and a
50,000-share bar both read 0.5 when two thirds of them is buying. The gate
is that missing dimension -- signed volume measured against the bar's own
recent normal -- and over 60 walk-forward sessions the two together scored
+0.351 a trade against +0.112 for DCVD alone.

It is encoded as opacity because colour already carries the sign and height
the ratio, leaving opacity the free channel. These tests pin the arithmetic
(which is where a lookahead would hide) and that the bars really are drawn
in two groups; the exact alphas are a style choice and are not pinned.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication   # noqa: E402

T0 = datetime(2026, 10, 19, 10, 0)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    from analysis.trade_viewer_qt import TradeViewerQt

    v = TradeViewerQt()
    yield v
    v.close()
    qapp.processEvents()


def draw(v, flows):
    """flows: list of (buy, sell). Returns the gate array."""
    from core.time_utils import candle_start

    rows, buckets = [], {}
    for i, (b, s) in enumerate(flows):
        end = T0 + timedelta(minutes=i + 1)
        rows.append((end.strftime("%Y-%m-%d %H:%M"),
                     100.0, 101.0, 99.0, 100.0, float(b + s)))
        buckets[candle_start(end - timedelta(minutes=1), 1)] = {
            100.0: {"buy": float(b), "sell": float(s), "neutral": 0.0}}
    kl = pd.DataFrame(rows, columns=["time_key", "open", "high", "low",
                                     "close", "volume"])
    v._clear_dcvd_items()
    v._draw_dcvd(kl, buckets, 1)
    return np.asarray(v._dcvd_gate, dtype=bool)


def steady(n=120, buy=3000, sell=2000):
    return [(buy, sell)] * n


class TestTheGate:
    def test_it_is_published_as_a_boolean_per_bar(self, viewer):
        g = draw(viewer, steady())
        assert g.shape == (120,)
        assert g.dtype == bool

    def test_a_flat_tape_admits_everything(self, viewer):
        """A degenerate case worth naming. Identical bars put every |x| at
        one value, the 80th percentile lands on that same value, and the
        test is >= -- so every bar clears. The opposite choice (>) would
        admit none.

        >= is kept because backtest/absorption.py uses >= and the panel has
        to draw the rule that was measured; on real data, where ties have
        measure zero, the two agree. The failure mode is therefore "the
        gate stops filtering", not "the gate hides the signal" -- and a
        tape flat to the share does not occur outside this test.
        """
        g = draw(viewer, steady())
        assert not g[:20].any(), "the warmup gates nothing"
        assert g[-20:].all(), "a tie should clear a >= gate"
        # One transition, no gaps: everything before the first True is
        # warmup, everything after it clears.
        first = int(np.argmax(g))
        assert g[first:].all(), "the gate flickered on a constant tape"

    def test_the_warmup_is_two_windows_deep(self, viewer):
        """About 40 bars, not 20. Two rolling windows chain: the volume
        median needs 20 before x exists at all, and the percentile then
        needs 20 non-NaN x. This is why the study's baseline covered 82%
        of minutes rather than all of them -- the opening stretch of each
        session has no gate."""
        g = draw(viewer, steady())
        first = int(np.argmax(g))
        assert 35 <= first <= 45, f"warmup was {first} bars"

    def test_a_burst_clears_it(self, viewer):
        flows = steady(120)
        flows[100] = (90_000, 2_000)
        g = draw(viewer, flows)
        assert g[100], "a 30x bar did not clear the gate"

    def test_the_warmup_never_clears(self, viewer):
        """Under min_periods there is no distribution to be extreme in."""
        flows = steady(120)
        flows[2] = (90_000, 2_000)
        g = draw(viewer, flows)
        assert not g[:20].any()

    def test_a_bar_does_not_raise_its_own_hurdle(self, viewer):
        """The rolling median and percentile both end at t-1. Were the
        current bar inside them, a burst would lift the threshold it is
        then asked to clear and could gate itself out."""
        flows = steady(120)
        flows[100] = (90_000, 2_000)
        g_one = draw(viewer, flows)
        # Same burst, but now the bars AFTER it differ. Nothing at or
        # before 100 changed, so the verdict at 100 must not move.
        flows2 = list(flows)
        for i in range(101, 120):
            flows2[i] = (50_000, 1_000)
        g_two = draw(viewer, flows2)
        assert g_one[100] == g_two[100]
        assert (g_one[:101] == g_two[:101]).all()

    def test_direction_is_not_part_of_the_gate(self, viewer):
        """|x| is symmetric: a burst of selling clears exactly as a burst
        of buying does. The side is decided afterwards, by the sign."""
        up = steady(120)
        up[100] = (90_000, 2_000)
        dn = steady(120)
        dn[100] = (2_000, 90_000)
        assert draw(viewer, up)[100] == draw(viewer, dn)[100]


class TestTheBarsAreDrawnInTwoGroups:
    def _bars(self, v):
        return [i for i in v._dcvd_items
                if type(i).__name__ == "BarGraphItem"]

    def test_a_mixed_tape_draws_lit_and_dim_separately(self, viewer):
        flows = steady(120)
        for i in (40, 60, 80, 100):
            flows[i] = (90_000, 2_000)
        draw(viewer, flows)
        # Two signs x two opacities, minus any combination with no bars.
        assert len(self._bars(viewer)) >= 2

    def test_nothing_is_drawn_for_an_empty_tape(self, viewer):
        draw(viewer, [(0, 0)] * 40)
        assert not self._bars(viewer)


class TestToolbarFold:
    """Six toolbar rows are about a fifth of the window, and none of it
    changes between trades. The control lives in the status bar because
    that is the one strip folding cannot take away with it."""

    def test_folding_hides_every_row_and_unfolding_restores_them(
            self, viewer, qapp):
        viewer.show()
        qapp.processEvents()
        rows = viewer._toolbar_rows
        assert rows and all(tb.isVisible() for tb in rows)

        viewer._toggle_toolbars()
        qapp.processEvents()
        assert not any(tb.isVisible() for tb in rows)

        viewer._toggle_toolbars()
        qapp.processEvents()
        assert all(tb.isVisible() for tb in rows)

    def test_the_button_says_which_way_it_goes(self, viewer, qapp):
        viewer.show()
        qapp.processEvents()
        before = viewer._fold_btn.text()
        viewer._toggle_toolbars()
        qapp.processEvents()
        assert viewer._fold_btn.text() != before
        viewer._toggle_toolbars()
        qapp.processEvents()
        assert viewer._fold_btn.text() == before


class TestTheLight:
    """Three bands on the newest bar, bottom-right of the PRICE chart.

    It was 7pt in the DCVD strip and could not be read; the panel is a few
    dozen pixels tall. It is a persistent item now rather than one rebuilt
    per draw, so these look it up by name instead of digging it out of the
    per-draw item list.

    Level only. The flip -- the rule's other half -- is already on screen:
    a red bar followed by a green one IS the flip, so a light repeating it
    would spend a colour on something the eye already has.

    Amber is not "weak green": between 0 and 0.4 measured +0.074 R a trade
    against +0.282 for the full rule, which is nearer nothing than signal.
    """

    @staticmethod
    def _label(v):
        return v._dcvd_light if v._dcvd_light.isVisible() else None

    def test_it_sits_on_the_price_chart_not_in_the_panel(self, viewer):
        """The whole point of the move. A light back in the DCVD strip is
        the unreadable one, and nothing else in these tests would notice."""
        draw(viewer, steady(40) + [(9000, 1000)])
        assert viewer._dcvd_light in viewer._plot_c.items
        assert viewer._dcvd_light not in viewer._plot_dcvd.items
        assert not any(type(i).__name__ == "TextItem"
                       for i in viewer._dcvd_items),             "a second light is still being drawn into the DCVD panel"

    def test_each_band_gets_its_own_colour(self, viewer):
        from analysis.trade_viewer_qt import DCVD_LIGHT

        seen = []
        for last in ((8000, 2000), (6000, 4000), (3500, 6500)):
            draw(viewer, steady(40) + [last])
            lbl = self._label(viewer)
            assert lbl is not None
            seen.append(lbl.textItem.defaultTextColor().name())
        assert len(set(map(str, seen))) == 3, f"bands share a colour: {seen}"
        # And the thresholds are the ones the rule was measured at.
        assert [lo for lo, _ in DCVD_LIGHT][:2] == [0.4, 0.0]

    def test_it_reads_the_newest_bar(self, viewer):
        draw(viewer, steady(40) + [(9000, 1000)])
        assert "+0.80" in self._label(viewer).toPlainText()
        draw(viewer, steady(40) + [(1000, 9000)])
        assert "-0.80" in self._label(viewer).toPlainText()

    def test_a_thin_newest_bar_falls_back_to_the_last_reading(self, viewer):
        """The floor voids a bar; the light then reports the newest bar that
        has a reading rather than vanishing or inventing a zero."""
        draw(viewer, steady(40) + [(9000, 1000)] + [(1, 1)])
        assert "+0.80" in self._label(viewer).toPlainText()

    def test_no_light_when_nothing_has_a_reading(self, viewer):
        draw(viewer, [(0, 0)] * 40)
        assert self._label(viewer) is None
