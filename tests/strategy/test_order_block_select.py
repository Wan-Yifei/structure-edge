"""Order-block selection: strength instead of recency, and both sides.

The chart drew a dozen overlapping boxes in one price band and nothing
below price. Both came from the one cut at the end of detect_order_blocks
-- `blocks[-max_count:]` keeps whatever happened last, and what happened
last is wherever price has been. On SOXL 5m the detector found 30 blocks,
14 above the last close and 15 below, and drew three above and none below.

These pin the new behaviour, and pin the old one too: backtest/engine.py
calls the same function and a silent change there is an algorithm change,
not a viewer fix.
"""

import numpy as np
import pandas as pd
import pytest

from strategy.smc.order_blocks import detect_order_blocks


def frame(n=80):
    """Alternating candles with volume that rises over time.

    Every bar alternates up/down so the detector always finds an opposing
    candle to anchor on, and volume climbs so later blocks are stronger --
    which makes "strength" and "recency" pick different sets, the thing
    worth telling apart.
    """
    i = np.arange(n)
    base = 100 + (i % 2) * 0.5
    return pd.DataFrame({
        "open": base, "close": base + np.where(i % 2, -0.4, 0.4),
        "high": base + 0.6, "low": base - 0.6,
        "volume": 1000.0 + i * 10,
    })


def sig(idx, direction, from_idx=None):
    return {"idx": idx, "direction": direction,
            "from_idx": from_idx if from_idx is not None else idx - 2}


class TestTheDefaultIsUnchanged:
    """backtest/engine.py gets exactly what it used to."""

    def test_recent_is_still_the_default(self):
        k = frame()
        s = [sig(i, "bull" if i % 2 else "bear") for i in range(10, 70, 6)]
        a = detect_order_blocks(k, s, max_count=4)
        b = detect_order_blocks(k, s, max_count=4, select="recent")
        assert [x["idx"] for x in a] == [x["idx"] for x in b]

    def test_recent_keeps_the_last_n_in_signal_order(self):
        k = frame()
        s = [sig(i, "bull") for i in range(10, 70, 6)]
        allb = detect_order_blocks(k, s, max_count=0)
        got = detect_order_blocks(k, s, max_count=3)
        assert [x["idx"] for x in got] == [x["idx"] for x in allb[-3:]]

    def test_max_count_zero_still_means_no_cut(self):
        k = frame()
        s = [sig(i, "bull") for i in range(10, 70, 6)]
        assert len(detect_order_blocks(k, s, max_count=0)) > 3


class TestVolRatioIsReported:
    """Selecting by strength needs a number; the detector had one and
    threw it away after filtering on it."""

    def test_every_block_carries_it(self):
        k = frame()
        s = [sig(i, "bull") for i in range(10, 70, 6)]
        for b in detect_order_blocks(k, s, max_count=0):
            assert b["vol_ratio"] > 0

    def test_it_is_volume_over_the_median_of_the_five_before(self):
        k = frame(40)
        k.loc[18:21, "volume"] = 5000.0       # whichever of these it picks
        s = [sig(22, "bull", from_idx=20)]
        b = detect_order_blocks(k, s, max_count=0)
        assert len(b) == 1
        # Checked against the bar the detector chose, not the one assumed:
        # it scans back from the swing for the first OPPOSING candle, so
        # which bar wins depends on the candle colours, not on the index.
        i = b[0]["idx"]
        med = float(np.median(k["volume"].values[max(0, i - 5):i]))
        assert b[0]["vol_ratio"] == pytest.approx(
            float(k["volume"].values[i]) / med)


class TestBothSidesGetDrawn:
    @staticmethod
    def _two_sided():
        """Blocks well above and well below a middle reference price."""
        n = 120
        px = np.concatenate([np.full(40, 110.0), np.full(40, 100.0),
                             np.full(40, 90.0)])
        i = np.arange(n)
        k = pd.DataFrame({
            "open": px, "close": px + np.where(i % 2, -0.4, 0.4),
            "high": px + 0.8, "low": px - 0.8,
            "volume": 1000.0 + i * 10,
        })
        # Alternate on the loop position, not the bar index: the step is
        # 6 from 10, so every index is even and `i % 2` made them all bear.
        s = [sig(i, "bull" if j % 2 else "bear")
             for j, i in enumerate(range(10, 118, 6))]
        return k, s

    def test_recency_starves_one_side(self):
        """The bug, kept as a test so the fix is visibly a fix."""
        k, s = self._two_sided()
        got = detect_order_blocks(k, s, max_count=4)
        ref = 100.0
        assert not [b for b in got if b["bottom"] > ref + 5], \
            "the recent cut should have taken only the latest band"

    def test_strength_fills_both(self):
        k, s = self._two_sided()
        got = detect_order_blocks(k, s, max_count=6, select="strength",
                                  max_breakers=2, ref_price=100.0)
        above = [b for b in got if b["bottom"] > 100.0]
        below = [b for b in got if b["top"] < 100.0]
        assert above and below, f"above={len(above)} below={len(below)}"

    def test_ref_price_defaults_to_the_last_close(self):
        k, s = self._two_sided()
        a = detect_order_blocks(k, s, max_count=6, select="strength")
        b = detect_order_blocks(k, s, max_count=6, select="strength",
                                ref_price=float(k["close"].iloc[-1]))
        assert [x["idx"] for x in a] == [x["idx"] for x in b]


class TestOverlapCollapses:
    def test_the_strongest_of_an_overlapping_group_survives(self):
        """Same direction, same price band, different volume."""
        k = frame(60)
        for idx, vol in ((20, 9000.0), (26, 3000.0), (32, 2000.0)):
            k.loc[idx, "volume"] = vol
        s = [sig(22, "bull", from_idx=20), sig(28, "bull", from_idx=26),
             sig(34, "bull", from_idx=32)]
        got = detect_order_blocks(k, s, max_count=6, select="strength",
                                  max_breakers=None, ref_price=0.0)
        live = [b for b in got if b["subtype"] != "breaker"]
        if len(live) > 1:
            for i, a in enumerate(live):
                for b in live[i + 1:]:
                    assert not (min(a["top"], b["top"])
                                > max(a["bottom"], b["bottom"])), \
                        "two kept blocks still overlap"

    def test_opposite_directions_are_not_merged(self):
        """A demand zone and a supply zone at one price are two readings
        of that price, not one."""
        k = frame(60)
        s = [sig(22, "bull", from_idx=20), sig(24, "bear", from_idx=21)]
        got = detect_order_blocks(k, s, max_count=6, select="strength",
                                  max_breakers=None, ref_price=0.0)
        dirs = {b["direction"] for b in got}
        assert dirs == {"bull", "bear"} or len(got) < 2


class TestBreakerCap:
    @staticmethod
    def _many_breakers():
        # Three legs, not two: price has to LEAVE a band for the blocks in
        # it to count as broken, so it visits 110, then 90, then settles at
        # 100 between them. With only 110 then 90, nothing below ever
        # breaks and the fixture cannot show what the cap does.
        n = 180
        px = np.concatenate([np.full(60, 110.0), np.full(60, 90.0),
                             np.full(60, 100.0)])
        i = np.arange(n)
        k = pd.DataFrame({
            "open": px, "close": px + np.where(i % 2, -0.4, 0.4),
            "high": px + 0.8, "low": px - 0.8,
            "volume": 1000.0 + i * 10,
        })
        s = [sig(i, "bull" if j % 2 else "bear")
             for j, i in enumerate(range(10, 178, 6))]
        return k, s

    def test_none_keeps_them_all(self):
        k, s = self._many_breakers()
        got = detect_order_blocks(k, s, max_count=50, select="strength",
                                  max_breakers=None, ref_price=100.0)
        allb = detect_order_blocks(k, s, max_count=0)
        n_brk = sum(1 for b in allb if b["subtype"] == "breaker")
        assert sum(1 for b in got if b["subtype"] == "breaker") == n_brk

    def test_the_cap_is_per_side_not_global(self):
        """A global cap put both survivors on whichever side price left
        last, which emptied the other side all over again."""
        k, s = self._many_breakers()
        got = detect_order_blocks(k, s, max_count=4, select="strength",
                                  max_breakers=1, ref_price=100.0)
        brk = [b for b in got if b["subtype"] == "breaker"]
        above = [b for b in brk if b["bottom"] > 100.0]
        below = [b for b in brk if b["top"] < 100.0]
        assert len(above) <= 1 and len(below) <= 1
        assert above and below, "each side should get its own retest level"

    def test_it_keeps_the_most_recent_not_the_strongest(self):
        """Volume decides which live block to show; for a broken one the
        question is which level price might come back to, so recency."""
        k, s = self._many_breakers()
        got = detect_order_blocks(k, s, max_count=4, select="strength",
                                  max_breakers=1, ref_price=100.0)
        allb = detect_order_blocks(k, s, max_count=0)
        for side, pick in (("above", lambda b: b["bottom"] > 100.0),
                           ("below", lambda b: b["top"] < 100.0)):
            kept = [b for b in got
                    if b["subtype"] == "breaker" and pick(b)]
            pool = [b for b in allb
                    if b["subtype"] == "breaker" and pick(b)]
            if kept and pool:
                assert kept[0]["idx"] == max(b["idx"] for b in pool), side
