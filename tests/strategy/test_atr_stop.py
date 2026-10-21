"""Fixed ATR stops: the arithmetic, and the text field people type into.

The calculation is small enough that the risk is not in the formula but at
its edges -- a short direction with the sign flipped, an anchor that is not
the latest close, a multiplier list that came from a text box. Those are
what this covers.
"""

import numpy as np
import pytest

from strategy.atr_stop import fixed_stops, parse_multipliers


def bars(n=60, start=100.0, rng=1.0):
    """A deterministic series with a known, constant true range."""
    c = np.full(n, start)
    h = c + rng / 2
    l = c - rng / 2
    return h, l, c


class TestTheStopPrice:
    def test_a_long_stop_sits_below_the_anchor(self):
        h, l, c = bars()
        r = fixed_stops(h, l, c, 14, [1.0, 2.0], "bull")
        assert r["anchor"] == 100.0
        for lv in r["levels"]:
            assert lv["stop"] < r["anchor"]
        # Twice the multiple is twice the distance; nothing else should move.
        a, b = r["levels"]
        assert b["dist"] == pytest.approx(2 * a["dist"])

    def test_a_short_stop_sits_above_it(self):
        h, l, c = bars()
        r = fixed_stops(h, l, c, 14, [1.5], "bear")
        lv = r["levels"][0]
        assert lv["stop"] > r["anchor"]
        assert lv["dist"] > 0, "distance is a gap, never signed"

    def test_the_distance_is_the_atr_times_the_multiple(self):
        h, l, c = bars(rng=2.0)
        r = fixed_stops(h, l, c, 14, [1.0, 1.5], "bull")
        for lv in r["levels"]:
            assert lv["dist"] == pytest.approx(r["atr"] * lv["mult"])
            assert lv["dist_pct"] == pytest.approx(
                lv["dist"] / r["anchor"] * 100)

    def test_an_explicit_anchor_replaces_the_close(self):
        """A position already open has a fill price; the ATR is still the
        latest one, which is what keeps the distance current."""
        h, l, c = bars()
        live = fixed_stops(h, l, c, 14, [1.0], "bull")
        held = fixed_stops(h, l, c, 14, [1.0], "bull", anchor=90.0)
        assert held["anchor"] == 90.0
        assert held["atr"] == live["atr"]
        assert held["levels"][0]["dist"] == live["levels"][0]["dist"]
        assert held["levels"][0]["stop"] < live["levels"][0]["stop"]


class TestItRefusesRatherThanInvents:
    """A stop drawn at a made-up price is worse than no stop drawn."""

    def test_too_few_bars_for_the_warmup(self):
        h, l, c = bars(n=10)
        assert fixed_stops(h, l, c, 14, [1.0], "bull") is None

    def test_a_flat_series_has_no_range(self):
        c = np.full(60, 100.0)
        assert fixed_stops(c.copy(), c.copy(), c, 14, [1.0], "bull") is None

    def test_no_usable_multiplier(self):
        h, l, c = bars()
        assert fixed_stops(h, l, c, 14, [], "bull") is None
        assert fixed_stops(h, l, c, 14, [0, -1], "bull") is None

    def test_a_nonsense_anchor(self):
        h, l, c = bars()
        assert fixed_stops(h, l, c, 14, [1.0], "bull", anchor=0) is None
        assert fixed_stops(h, l, c, 14, [1.0], "bull",
                           anchor=float("nan")) is None


class TestTheTextField:
    @pytest.mark.parametrize("text,want", [
        ("1, 1.5, 2", [1.0, 1.5, 2.0]),
        ("1,1.5,2", [1.0, 1.5, 2.0]),
        (" 2 ;1 ", [1.0, 2.0]),
        ("1, 1.5, 2,", [1.0, 1.5, 2.0]),          # trailing comma
        ("2, 1, 2", [1.0, 2.0]),                   # duplicate collapses
        ("1，1.5", [1.0, 1.5]),                    # full-width comma
    ])
    def test_it_reads_what_a_person_types(self, text, want):
        assert parse_multipliers(text) == want

    def test_a_typo_costs_one_entry_not_the_overlay(self):
        assert parse_multipliers("1, abc, 2") == [1.0, 2.0]

    def test_nothing_usable_is_an_empty_list(self):
        for t in ("", None, "   ", "0", "-1", "abc"):
            assert parse_multipliers(t) == []
