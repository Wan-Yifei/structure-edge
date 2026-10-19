"""The vectorised bar-start keys must equal candle_start(), exactly.

Five drawing paths used to rebuild these keys each with iterrows() plus
strptime, per bar. On 4,081 one-minute bars that measured 240 ms apiece --
about a second of a 1.6 s render spent computing one identical list five
times over. The replacement is vectorised and memoised.

A vectorised rewrite of a date rule is the kind of change that looks right
and is wrong on one timeframe. pandas' floor() is epoch-aligned while
candle_start() aligns within the hour; the two agree only while 60 is
divisible by the candle size, which every size in TIMEFRAME_MAP happens to
satisfy. So the shortcut would have passed today and broken silently the
day a 45m was added. These tests compare against candle_start() itself,
across every size the viewer offers and a few it does not.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from core.time_utils import candle_start                     # noqa: E402


def reference(stamps, cm):
    """What the five loops did, bar by bar."""
    out = []
    for s in stamps:
        end = datetime.strptime(str(s)[:16], "%Y-%m-%d %H:%M")
        out.append(candle_start(end - timedelta(minutes=cm), cm))
    return out


def stamps_for(cm, n=400):
    """n consecutive candle stamps, crossing an hour and a day boundary."""
    t0 = datetime(2026, 10, 19, 9, 31)
    return [(t0 + timedelta(minutes=cm * i)).strftime("%Y-%m-%d %H:%M")
            for i in range(n)]


class TestItMatchesCandleStart:
    @pytest.mark.parametrize("cm", [1, 3, 5, 15, 30, 60, 240, 1440])
    def test_every_timeframe_the_viewer_offers(self, cm):
        from analysis.trade_viewer_qt import bar_start_keys

        s = stamps_for(cm)
        got = list(bar_start_keys(pd.Series(s), cm))
        assert got == reference(s, cm), f"cm={cm}"

    @pytest.mark.parametrize("cm", [2, 4, 45, 90])
    def test_sizes_that_do_not_divide_an_hour(self, cm):
        """45 and 90 are where floor() would have diverged. They are not in
        TIMEFRAME_MAP today; this is here so adding one fails loudly."""
        from analysis.trade_viewer_qt import bar_start_keys

        s = stamps_for(cm)
        assert list(bar_start_keys(pd.Series(s), cm)) == reference(s, cm)

    def test_the_map_is_covered(self):
        """If a timeframe is added and not tested above, say so here rather
        than let it through untested."""
        from analysis.trade_viewer_qt import TIMEFRAME_MAP

        tested = {1, 3, 5, 15, 30, 60, 240, 1440}
        offered = {cm for _, cm in TIMEFRAME_MAP.values()}
        assert offered <= tested, f"untested timeframes: {offered - tested}"


class TestItSurvivesBadInput:
    def test_an_unparseable_stamp_becomes_nat(self):
        from analysis.trade_viewer_qt import bar_start_keys

        s = pd.Series(["2026-10-19 09:31", "not a date", "2026-10-19 09:33"])
        got = bar_start_keys(s, 1)
        assert got[0] == datetime(2026, 10, 19, 9, 30)
        assert got[1] is pd.NaT, "callers skip on NaT, so it must be NaT"
        assert got[2] == datetime(2026, 10, 19, 9, 32)

    def test_an_empty_frame_is_empty_not_an_error(self):
        from analysis.trade_viewer_qt import bar_start_keys

        assert len(bar_start_keys(pd.Series([], dtype=object), 1)) == 0


class TestTheMemo:
    """The whole point is computing it once; a memo that misses is the
    same cost as before, and one that hits when it should not is worse."""

    @pytest.fixture(scope="class")
    def viewer(self):
        from PyQt6.QtWidgets import QApplication
        from analysis.trade_viewer_qt import TradeViewerQt

        app = QApplication.instance() or QApplication([])
        v = TradeViewerQt()
        yield v
        v.close()
        app.processEvents()

    @staticmethod
    def _frame(stamps):
        return pd.DataFrame({"time_key": stamps,
                             "low": [1.0] * len(stamps)})

    def test_the_same_frame_hits(self, viewer):
        k = self._frame(stamps_for(1, 50))
        a = viewer._bucket_keys(k, 1)
        assert viewer._bucket_keys(k, 1) is a

    def test_a_new_bar_misses(self, viewer):
        s = stamps_for(1, 50)
        a = viewer._bucket_keys(self._frame(s), 1)
        b = viewer._bucket_keys(self._frame(s + stamps_for(1, 1)), 1)
        assert b is not a and len(b) == len(a) + 1

    def test_changing_the_timeframe_misses(self, viewer):
        k = self._frame(stamps_for(1, 50))
        a = viewer._bucket_keys(k, 1)
        assert viewer._bucket_keys(k, 5) is not a
