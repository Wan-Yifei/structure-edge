"""Tests for drill_gen -- mostly that a question shows no bar past its decision.

A drill is worthless if any field leaks the future, and the leak would be
invisible: the page would just look unusually easy. So the lookahead checks
are written as the real property -- recompute the field from the truncated
series and demand the same number -- rather than checking an off-by-one.
"""
import numpy as np
import pandas as pd
import pytest

from strategy.drill_gen import (atr, call, flow, label, pivots, structure)


def _ramp(n, start=100.0, step=0.1):
    c = np.array([start + step * i for i in range(n)])
    return c + 0.05, c - 0.05, c          # high, low, close


class TestAtr:
    def test_warmup_is_nan(self):
        h, l, c = _ramp(40)
        a = atr(h, l, c, n=14)
        assert np.isnan(a[:14]).all()
        assert np.isfinite(a[14:]).all()

    def test_constant_range_converges_to_that_range(self):
        h, l, c = _ramp(80, step=0.0)
        a = atr(h, l, c, n=14)
        assert a[-1] == pytest.approx(0.10, abs=1e-6)

    def test_short_series_is_all_nan(self):
        h, l, c = _ramp(10)
        assert np.isnan(atr(h, l, c, n=14)).all()


class TestLabel:
    def test_up_barrier_first_is_a_win(self):
        H = np.array([10.0, 10.0, 13.0])
        L = np.array([10.0, 9.9, 12.0])
        C = np.array([10.0, 10.0, 13.0])
        assert label(H, L, C, 0, a=1.0, up=2.0, dn=1.0, fwd=5) == 1

    def test_down_barrier_first_is_a_loss(self):
        H = np.array([10.0, 10.1, 13.0])
        L = np.array([10.0, 8.5, 12.0])
        C = np.array([10.0, 9.0, 13.0])
        assert label(H, L, C, 0, a=1.0, up=2.0, dn=1.0, fwd=5) == 0

    def test_same_bar_touches_both_counts_as_loss(self):
        """The bar's path is unknown, so the adverse reading is the honest one."""
        H = np.array([10.0, 12.5])
        L = np.array([10.0, 8.5])
        C = np.array([10.0, 10.0])
        assert label(H, L, C, 0, a=1.0, up=2.0, dn=1.0, fwd=5) == 0

    def test_neither_barrier_in_horizon_is_none(self):
        H = np.full(10, 10.05)
        L = np.full(10, 9.95)
        C = np.full(10, 10.0)
        assert label(H, L, C, 0, a=1.0, up=2.0, dn=1.0, fwd=5) is None

    def test_horizon_is_respected(self):
        H = np.array([10.0] + [10.05] * 5 + [99.0])
        L = np.full(7, 9.95)
        C = np.full(7, 10.0)
        assert label(H, L, C, 0, a=1.0, fwd=3) is None
        assert label(H, L, C, 0, a=1.0, fwd=6) == 1


class TestPivots:
    def test_last_k_bars_are_never_confirmed(self):
        """A swing needs k bars on its right, so the newest k cannot qualify --
        the real-time cost of a pivot definition."""
        rng = np.random.default_rng(0)
        H = 100 + rng.normal(0, 1, 80).cumsum()
        L = H - 0.5
        sh, sl = pivots(H, L, 0, 79, k=3)
        assert all(j <= 79 - 3 for j in sh + sl)

    def test_a_clear_peak_is_found(self):
        H = np.array([1, 2, 3, 9, 3, 2, 1], float)
        sh, _ = pivots(H, H - 0.1, 0, 6, k=3)
        assert sh == [3]

    def test_adding_future_bars_cannot_change_past_pivots(self):
        rng = np.random.default_rng(1)
        H = 100 + rng.normal(0, 1, 120).cumsum()
        L = H - 0.4
        a, b = pivots(H, L, 0, 60, k=3)
        a2, b2 = pivots(H[:70], L[:70], 0, 60, k=3)
        assert (a, b) == (a2, b2)


class TestStructure:
    def test_too_few_swings_abstains(self):
        H, L, C = np.full(20, 10.05), np.full(20, 9.95), np.full(20, 10.0)
        sc, why, info = structure(C, H, L, C, 19, a=0.1)
        assert sc == 0 and info["trend"] == "未知"

    def test_rising_series_reads_as_uptrend(self):
        n = 70
        base = np.array([100 + 0.3 * i for i in range(n)])
        wig = np.array([2.2 * np.sin(i / 2.2) for i in range(n)])
        c = base + wig
        sc, why, info = structure(c, c + 0.3, c - 0.3, c, n - 1, a=0.4)
        assert info["trend"] == "上升"
        assert sc > 0

    def test_falling_series_reads_as_downtrend_and_scores_negative(self):
        n = 70
        base = np.array([140 - 0.3 * i for i in range(n)])
        wig = np.array([2.2 * np.sin(i / 2.2) for i in range(n)])
        c = base + wig
        sc, why, info = structure(c, c + 0.3, c - 0.3, c, n - 1, a=0.4)
        assert info["trend"] == "下降"
        assert sc < 0

    def test_reading_ignores_bars_after_the_decision(self):
        rng = np.random.default_rng(3)
        c = 100 + rng.normal(0, 0.4, 200).cumsum()
        H, L = c + 0.3, c - 0.3
        i = 120
        a = structure(c, H, L, c, i, a=0.5)
        shocked = c.copy()
        shocked[i + 1:] += 50.0            # violent, entirely in the future
        b = structure(shocked, shocked + 0.3, shocked - 0.3, shocked, i, a=0.5)
        assert a[0] == b[0]
        assert [t for _, t in a[1]] == [t for _, t in b[1]]

    def test_every_reason_carries_a_signed_weight(self):
        n = 70
        c = np.array([100 + 0.3 * i + 2.2 * np.sin(i / 2.2) for i in range(n)])
        sc, why, _ = structure(c, c + 0.3, c - 0.3, c, n - 1, a=0.4)
        assert why and all(isinstance(v, int) for v, _ in why)
        assert sc == sum(v for v, _ in why)


class TestCall:
    def test_thresholds(self):
        assert call(2) == "看涨" and call(5) == "看涨"
        assert call(1) == "中性" and call(0) == "中性"
        assert call(-1) == "看跌" and call(-4) == "看跌"


class TestFlow:
    def test_window_ends_at_the_decision_bar(self):
        buy = np.array([0.0] * 10 + [1000.0])
        sell = np.array([1000.0] * 10 + [0.0])
        assert flow(buy, sell, 10, 1) == 1.0          # only the last bar
        assert flow(buy, sell, 9, 1) == -1.0          # only bar 9

    def test_future_bars_cannot_move_the_answer(self):
        rng = np.random.default_rng(5)
        buy = rng.uniform(200, 900, 60)
        sell = rng.uniform(200, 900, 60)
        i = 40
        for n in (1, 5, 15):
            before = flow(buy, sell, i, n)
            b2, s2 = buy.copy(), sell.copy()
            b2[i + 1:] = 1e9                 # future stuffed with buying
            assert flow(b2, s2, i, n) == before

    def test_thin_tape_abstains_rather_than_reporting_noise(self):
        buy = np.full(20, 5.0)
        sell = np.full(20, 5.0)
        assert flow(buy, sell, 10, 1) is None

    def test_longer_window_smooths_a_single_bar_flip(self):
        buy = np.concatenate([np.full(14, 100.0), [5000.0]])
        sell = np.concatenate([np.full(14, 900.0), [50.0]])
        one, five = flow(buy, sell, 14, 1), flow(buy, sell, 14, 5)
        # The newest bar is overwhelmingly bought; the four before it were
        # sold. The 1m reading takes the flip at face value, the 5m keeps
        # the preceding bars in view and lands far short of it.
        assert one == pytest.approx(0.980, abs=0.01)
        assert five == pytest.approx(0.193, abs=0.01)
        assert abs(five) < abs(one)

    def test_balanced_tape_is_zero(self):
        buy = np.full(20, 500.0)
        sell = np.full(20, 500.0)
        assert flow(buy, sell, 10, 5) == 0.0


class TestHeatmap:
    """The grid is the only field built from a second-resolution source, so
    its two aggregations -- sum across price levels, average across seconds --
    are easy to get backwards and invisible once drawn."""

    @staticmethod
    def _ladder(n_sec=600, levels=(100.0, 100.5, 101.0), size=1000.0):
        t0 = pd.Timestamp("2026-10-14 10:00:00")
        rows = []
        for s in range(n_sec):
            for p in levels:
                rows.append((t0 + pd.Timedelta(seconds=s), p, size, "B"))
        return pd.DataFrame(rows, columns=["ts", "price", "size", "side"])

    def test_empty_or_missing_input_returns_none(self):
        from strategy.drill_gen import heatmap
        t = pd.Timestamp("2026-10-14 10:10:00")
        assert heatmap(None, t, 99.0, 102.0) is None
        empty = pd.DataFrame(columns=["ts", "price", "size", "side"])
        assert heatmap(empty, t, 99.0, 102.0) is None

    def test_window_excludes_anything_after_the_decision(self):
        from strategy.drill_gen import heatmap
        lad = self._ladder()
        t = pd.Timestamp("2026-10-14 10:05:00")
        a = heatmap(lad, t, 99.0, 102.0)
        future = lad.copy()
        future.loc[future.ts > t, "size"] = 9e6
        b = heatmap(future, t, 99.0, 102.0)
        assert a["db"] == b["db"] and a["da"] == b["da"]
        assert a["top"] == b["top"]

    def test_levels_in_one_bin_sum_while_seconds_average(self):
        """Three levels of 1000 in one bin, held every second, is a band of
        3000 -- not 1000 (averaged across levels) and not 3000 x seconds."""
        from strategy.drill_gen import heatmap
        # 99-102 over 48 bins is 0.0625 a bin, so all three share one.
        lad = self._ladder(levels=(100.0, 100.01, 100.02), size=1000.0)
        t = pd.Timestamp("2026-10-14 10:10:00")
        hm = heatmap(lad, t, 99.0, 102.0)
        assert hm["top"] == pytest.approx(3000, rel=0.02)

    def test_grid_shape_matches_the_declared_dimensions(self):
        import base64
        from strategy.drill_gen import heatmap
        lad = self._ladder()
        hm = heatmap(lad, pd.Timestamp("2026-10-14 10:10:00"), 99.0, 102.0)
        for key in ("db", "da"):
            assert len(base64.b64decode(hm[key])) == hm["nb"] * hm["nc"]
        assert hm["nc"] == 30 * 60 // hm["sec"]

    def test_a_wall_lands_in_its_own_price_row(self):
        import base64
        from strategy.drill_gen import heatmap
        lad = self._ladder(levels=(100.0, 101.0), size=100.0)
        lad.loc[lad.price == 101.0, "size"] = 50000.0
        lo, hi = 99.0, 102.0
        hm = heatmap(lad, pd.Timestamp("2026-10-14 10:10:00"), lo, hi)
        g = np.frombuffer(base64.b64decode(hm["db"]), np.uint8) \
              .reshape(hm["nb"], hm["nc"])
        hot = int((101.0 - lo) / (hi - lo) * hm["nb"])
        assert g[hot].max() == 255
        quiet = int((100.0 - lo) / (hi - lo) * hm["nb"])
        # Quantisation is logarithmic, so the quiet level survives the byte.
        # Linear against the maximum put 100 shares next to a 50,000 wall on
        # a code of 1 and every ordinary level collapsed into the same two or
        # three values -- gone before the page could draw it. It must still
        # be clearly dimmer than the wall; the page separates them further
        # with a gamma above 1.
        assert 20 < g[quiet].max() < 200
        assert g[quiet].max() < g[hot].max()
        assert hm["log"] == 1

    def test_a_price_range_with_no_rows_returns_none(self):
        from strategy.drill_gen import heatmap
        lad = self._ladder()
        t = pd.Timestamp("2026-10-14 10:10:00")
        assert heatmap(lad, t, 200.0, 210.0) is None


class TestReadSetRoundTrip:
    """--append reads back what the generator wrote. The two ends drifted
    once already: output moved to `window.DRILL_c1=[...];` for file:// and
    the reader was still calling json.loads, so the first scheduled append
    died on a JSONDecodeError."""

    @staticmethod
    def _rows():
        return [{"id": "a_c1", "day": "2026-10-14", "tm": "11:30"},
                {"id": "b_c1", "day": "2026-10-15", "tm": "13:05"}]

    def test_reads_plain_json(self, tmp_path):
        import json as _json
        from strategy.drill_gen import _read_set
        p = tmp_path / "q.json"
        p.write_text(_json.dumps(self._rows()), encoding="utf-8")
        assert _read_set(p) == self._rows()

    def test_reads_the_js_assignment_the_page_loads(self, tmp_path):
        import json as _json
        from strategy.drill_gen import _read_set
        p = tmp_path / "q.js"
        p.write_text(f"window.DRILL_c1={_json.dumps(self._rows())};",
                     encoding="utf-8")
        assert _read_set(p) == self._rows()

    def test_reads_confirm0_naming_too(self, tmp_path):
        import json as _json
        from strategy.drill_gen import _read_set
        p = tmp_path / "q.js"
        p.write_text(f"window.DRILL_c0={_json.dumps(self._rows())};",
                     encoding="utf-8")
        assert len(_read_set(p)) == 2

    def test_empty_file_is_an_empty_set_not_a_crash(self, tmp_path):
        from strategy.drill_gen import _read_set
        p = tmp_path / "q.js"
        p.write_text("", encoding="utf-8")
        assert _read_set(p) == []


class TestFlowStripShowsNoFuture:
    """The strip sits under a chart whose right-hand bars are hidden, so any
    leak there hands over the answer. Written as the property -- make the
    future extreme and demand the output is byte-identical -- rather than
    checking an off-by-one, because an off-by-one check passes while the
    window is wrong by an hour."""

    @staticmethod
    def _tape(n_min=60, start="2026-10-14 11:00:00", px=100.0):
        t0 = pd.Timestamp(start)
        rows = []
        for m in range(n_min):
            for s in range(0, 60, 5):
                # One side per minute and 3,000 a print, so a 10-second
                # column nets 6,000 and clears BUBBLE_MIN. Alternating every
                # print instead left every column at net zero.
                rows.append((t0 + pd.Timedelta(minutes=m, seconds=s),
                             px + 0.01 * (m % 7), 3000.0,
                             "BUY" if m % 2 else "SELL"))
        return pd.DataFrame(rows, columns=["ts", "price", "volume", "direction"])

    def test_bubbles_ignore_prints_after_the_decision(self):
        from strategy.drill_gen import bubbles
        tk = self._tape()
        t = pd.Timestamp("2026-10-14 11:40:00")
        before = bubbles(tk, t, 99.0, 101.0)
        after = tk.copy()
        m = after.ts > t
        after.loc[m, "volume"] = 5e6          # a flood, entirely in the future
        after.loc[m, "price"] = 100.9
        assert bubbles(after, t, 99.0, 101.0) == before

    def test_traded_profile_ignores_prints_after_the_decision(self):
        from strategy.drill_gen import traded_profile
        tk = self._tape()
        t = pd.Timestamp("2026-10-14 11:40:00")
        before = traded_profile(tk, t, 99.0, 101.0)
        after = tk.copy()
        after.loc[after.ts > t, "volume"] = 5e6
        assert traded_profile(after, t, 99.0, 101.0) == before

    def test_the_last_bubble_column_is_not_past_the_decision(self):
        """Column c covers (t_end-30min + c*sec, +sec], so the final column
        ends exactly at the decision bar's close -- never beyond it."""
        from strategy.drill_gen import HM_MINS, HM_SECS, bubbles
        tk = self._tape()
        t = pd.Timestamp("2026-10-14 11:40:00")
        b = bubbles(tk, t, 99.0, 101.0)
        ncol = HM_MINS * 60 // HM_SECS
        assert b, "expected some bubbles in this fixture"
        assert max(x[0] for x in b) <= ncol - 1

    def test_a_bubble_cannot_sit_outside_the_declared_price_range(self):
        from strategy.drill_gen import bubbles
        tk = self._tape()
        t = pd.Timestamp("2026-10-14 11:40:00")
        b = bubbles(tk, t, 100.02, 100.04)
        assert all(100.02 <= x[1] <= 100.04 for x in b)


class TestBreakoutIsNotAPullback:
    """`cur_leg = peak - low` goes negative when the low sits above both
    swing highs. The retracement text then read "本段下跌 -1.2 ATR，是前一段
    的 -29%，推动在减弱" -- a negative drop, and the opposite of what a low
    holding above the prior high means."""

    @staticmethod
    def _rising_then_breakout(n=70):
        c = np.array([100 + 0.3 * i + 2.2 * np.sin(i / 2.2) for i in range(n)])
        c[-6:] = c[-7] + np.arange(1, 7) * 1.5      # drive it clear of the highs
        return c

    def test_no_negative_drop_is_ever_reported(self):
        from strategy.drill_gen import structure
        c = self._rising_then_breakout()
        _, why, _ = structure(c, c + 0.3, c - 0.3, c, len(c) - 1, a=0.4)
        for _, txt in why:
            assert "-" not in txt.replace("—", ""), f"negative in: {txt}"
            assert "下跌 -" not in txt

    def test_it_says_breakout_and_scores_positive(self):
        from strategy.drill_gen import structure
        c = self._rising_then_breakout()
        sc, why, _ = structure(c, c + 0.3, c - 0.3, c, len(c) - 1, a=0.4)
        assert any("突破" in txt for _, txt in why), \
            f"expected a breakout line, got {[t for _, t in why]}"
        assert sc > 0

    def test_an_ordinary_pullback_still_gets_the_comparison(self):
        """A real pullback: the drift fixture alone ends ABOVE its own swing
        highs, so it is a breakout by this definition -- the last bars have
        to come back down for the retracement branch to apply at all."""
        from strategy.drill_gen import structure
        n = 70
        c = np.array([100 + 0.3 * i + 2.2 * np.sin(i / 2.2) for i in range(n)])
        c[-5:] = c[-6] - np.arange(1, 6) * 1.2
        _, why, _ = structure(c, c + 0.3, c - 0.3, c, n - 1, a=0.4)
        assert not any("突破" in txt for _, txt in why),             f"a pullback was read as a breakout: {[x for _, x in why]}"


class TestHeatmapSides:
    """Bid and ask are separate layers. Summed into one, a bid wall and the
    ask wall above it render identically and the panel stops answering the
    only question it exists for -- which side is holding."""

    @staticmethod
    def _two_sided():
        t0 = pd.Timestamp("2026-10-14 10:00:00")
        rows = []
        for s in range(600):
            rows.append((t0 + pd.Timedelta(seconds=s), 100.0, 4000.0, "B"))
            rows.append((t0 + pd.Timedelta(seconds=s), 101.0, 1000.0, "A"))
        return pd.DataFrame(rows, columns=["ts", "price", "size", "side"])

    @staticmethod
    def _grid(hm, key):
        import base64
        raw = base64.b64decode(hm[key])
        return np.frombuffer(raw, np.uint8).reshape(hm["nb"], hm["nc"])

    def test_each_side_lands_only_in_its_own_layer(self):
        from strategy.drill_gen import heatmap
        lo, hi = 99.0, 102.0
        hm = heatmap(self._two_sided(),
                     pd.Timestamp("2026-10-14 10:10:00"), lo, hi)
        gb, ga = self._grid(hm, "db"), self._grid(hm, "da")
        rb = int((100.0 - lo) / (hi - lo) * hm["nb"])
        ra = int((101.0 - lo) / (hi - lo) * hm["nb"])
        assert gb[rb].max() > 0 and gb[ra].max() == 0
        assert ga[ra].max() > 0 and ga[rb].max() == 0

    def test_both_layers_share_one_reference(self):
        """Scaled apart, a thin ask would render as solid as a heavy bid.
        The bid here is four times the ask and has to look it."""
        from strategy.drill_gen import heatmap
        hm = heatmap(self._two_sided(),
                     pd.Timestamp("2026-10-14 10:10:00"), 99.0, 102.0)
        gb, ga = self._grid(hm, "db"), self._grid(hm, "da")
        assert gb.max() == 255, "the larger side sets the shared reference"
        assert ga.max() < gb.max()
        assert hm["top"] == pytest.approx(4000, rel=0.02)

    def test_a_one_sided_book_still_emits_both_layers(self):
        """The page indexes both keys; a missing one would be a crash on the
        quiet days rather than an empty layer."""
        from strategy.drill_gen import heatmap
        lad = TestHeatmap._ladder()          # bids only
        hm = heatmap(lad, pd.Timestamp("2026-10-14 10:10:00"), 99.0, 102.0)
        assert "db" in hm and "da" in hm
        assert self._grid(hm, "da").max() == 0


class TestPinchKind:
    """The fast midline rolling into the slow one is a third shape. Both
    existing kinds trigger on a new low, so nothing in the set covered it."""

    @staticmethod
    def _session(n=390):
        """Up for 150 bars, then down for the rest -- long enough for the
        span-25 midline to run well clear of the span-90 one and then come
        back and cross it inside the scan window."""
        import numpy as np
        import pandas as pd

        i = np.arange(n, dtype=float)
        up = np.minimum(i, 150.0) * 0.08
        dn = np.maximum(i - 150.0, 0.0) * 0.06
        c = 100.0 + up - dn + 0.35 * np.sin(i / 3.1)
        t = pd.date_range("2026-10-14 09:31", periods=n, freq="1min")
        return pd.DataFrame(
            {"open": c, "high": c + 0.3, "low": c - 0.3, "close": c,
             "volume": np.full(n, 1000.0),
             "buy": np.full(n, 600.0), "sell": np.full(n, 400.0)},
            index=pd.DatetimeIndex(t, name="t"),
        )

    def _rows(self, kinds):
        from strategy.drill_gen import build_day
        return build_day("US.SOXL", "2026-10-14", self._session(),
                         confirm=1, kinds=kinds, light=True)

    def test_a_pinch_is_found(self):
        got = self._rows(("deep", "shallow", "pinch"))
        assert any(r["kind"] == "夹角" for r in got), \
            f"no pinch in {sorted({r['kind'] for r in got})}"

    def test_adding_it_does_not_displace_a_single_dip(self):
        """The property worth a test: dips keep their own spacing cursor, so
        the questions already on file -- and the answers keyed to their ids
        -- survive a regeneration that introduces the new kind."""
        before = [r["id"] for r in self._rows(("deep", "shallow"))]
        after = [r["id"] for r in self._rows(("deep", "shallow", "pinch"))
                 if r["kind"] != "夹角"]
        assert after == before, (
            f"{len(before) - len(after)} dip question(s) changed or vanished")

    def test_only_the_leading_edge_is_marked(self):
        """The condition holds for a run of bars; each run is one question."""
        from strategy.drill_gen import GAP
        got = [r for r in self._rows(("pinch",)) if r["kind"] == "夹角"]
        mins = sorted(int(r["tm"][:2]) * 60 + int(r["tm"][3:]) for r in got)
        assert mins, "fixture produced no pinch to space-check"
        for a, b in zip(mins, mins[1:]):
            assert b - a >= GAP, f"two pinches {b - a} bars apart"

    def test_the_kind_is_off_when_not_asked_for(self):
        got = self._rows(("deep", "shallow"))
        assert not any(r["kind"] == "夹角" for r in got)
