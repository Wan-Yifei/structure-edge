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

    HUE IS THE SIGN, BRIGHTNESS IS THE FLOW STATE. An earlier version
    painted the cleared gate teal whatever the sign, so a heavy SELL
    reading announced itself in the colour that means up.

    The first fix for that swapped in the gold/purple pair the heatmap
    and the delta labels use, on the reasoning that any hue a reader
    parses as direction is wrong here. That went too far: the badge is
    read as a traffic light, and gold is the amber that means "not yet",
    so a heavy +0.88 looked like a warning. Direction is what the sign
    IS -- the bug was a hue contradicting it, not the existence of a
    hue. So: candle green above zero, candle red below, and brightness
    for whether the gate cleared.
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

    def test_a_negative_reading_is_never_painted_in_the_bull_colour(
            self, viewer):
        """The reported bug, as a test.

        `DCVD -0.72  heavy flow` came out in #26a69a, the same teal as a
        bull candle.
        """
        from analysis.trade_viewer_qt import _GREEN, _RED
        base = steady(40, buy=3000, sell=2000)
        # Heavy and NEGATIVE: ten times the usual size, selling.
        g = draw(viewer, base + [(20000, 30000)])
        assert g[-1], "the fixture's heavy bar did not clear the gate"
        lbl = self._label(viewer)
        got = lbl.textItem.defaultTextColor()
        assert "-" in lbl.toPlainText(), "fixture is not a negative reading"
        assert got.name().lower() != _GREEN.lower(), \
            f"负的读数画成了多头色 {got.name()}"
        assert got.name().lower() == _RED.lower(), \
            f"heavy selling should take the sell hue, got {got.name()}"

    def test_the_hue_follows_the_sign(self, viewer):
        """Buying and selling differ in hue, not merely in the number.

        And in the pair a traffic light is read with: the second report
        on this badge was a heavy +0.88 in gold, which reads as amber --
        "strong buying" announced as a warning.
        """
        from analysis.trade_viewer_qt import _GREEN, _RED, _UP, _DOWN
        base = steady(40, buy=3000, sell=2000)
        draw(viewer, base + [(30000, 20000)])
        up = self._label(viewer).textItem.defaultTextColor().name()
        draw(viewer, base + [(20000, 30000)])
        dn = self._label(viewer).textItem.defaultTextColor().name()
        assert up.lower() == _GREEN.lower() and dn.lower() == _RED.lower()
        assert up != dn
        for amber in (_UP, _DOWN):
            assert up.lower() != amber.lower(), \
                f"大流量读数用了热力图的金/紫，在红绿灯里读成黄灯：{up}"

    def test_the_gate_shows_as_brightness_not_as_another_hue(self, viewer):
        """Green moved off the level and onto the gate, but as opacity.

        Over the study's 60 out-of-sample sessions the gate returned
        +0.363 R a trade against +0.264 for the level, and the two
        disagree precisely here: 0.4 on thin volume (+0.186) and heavy
        volume that never reached 0.4 (+0.300). So the fixtures ARE
        those two disagreements, not merely two different numbers --
        and the pair must differ in alpha while sharing a hue.
        """
        from analysis.trade_viewer_qt import DCVD_LIGHT_NONE
        base = steady(40, buy=3000, sell=2000)

        # Heavy, deliberately UNDER 0.4: dr1 = 0.2 on ten times the size.
        g = draw(viewer, base + [(30000, 20000)])
        assert g[-1], "the fixture's heavy bar did not clear the gate"
        heavy = self._label(viewer).textItem.defaultTextColor()

        # Over 0.4 on a SMALL bar -- the old green, now the dim one. On
        # this flat tape every |x| is 0.2, so the gate is 0.2: a lopsided
        # ratio on a quiet bar (ofi 900 against a scale of 5000) is thin
        # even at dr1 = 0.60.
        g = draw(viewer, base + [(1200, 300)])
        assert not g[-1], "the fixture's thin bar cleared the gate"
        thin = self._label(viewer).textItem.defaultTextColor()

        # Neither: small bar, small ratio.
        g = draw(viewer, base + [(700, 500)])
        assert not g[-1]
        none = self._label(viewer).textItem.defaultTextColor()

        assert heavy.rgb() == thin.rgb(), \
            "闸门过没过不该换色相，只该换亮度"
        assert heavy.alpha() > thin.alpha() + 40, \
            f"大流量没有更亮：{heavy.alpha()} vs {thin.alpha()}"
        assert none.name().lower() == DCVD_LIGHT_NONE.lower()

    def test_the_label_says_which_state_it_is_in_english(self, viewer):
        """The colour alone is a convention to memorise; the word is not.

        English, because every other mark on this chart is -- BOS,
        CHoCH, Put Wall. A lone Chinese phrase among them reads as a
        different system's label.
        """
        import re
        # Not isascii(): the badge opens with a bullet on purpose, and
        # that check would reject it along with the thing being banned.
        cjk = re.compile(r"[\u4e00-\u9fff]")
        for flows, word in (([(30000, 20000)], "heavy flow"),
                            ([(1200, 300)], "thin volume")):
            draw(viewer, steady(40) + flows)
            t = self._label(viewer).toPlainText()
            assert word in t, t
            assert not cjk.search(t), f"标签里还有中文：{t!r}"

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
