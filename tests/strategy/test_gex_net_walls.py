"""Net walls rank on what is left after calls and puts cancel.

The separated Call Wall is the strike with the most positive CALL gamma,
computed as if the puts written at that strike did not exist. Dealers hedge
the residual, so when a strike's call gamma is largely offset the line drawn
there overstates what anyone has to trade.

Every test below is built so the two modes DISAGREE -- a fixture where they
happen to pick the same strike would pass against a `net` flag that was
never read.
"""
import pandas as pd
import pytest

from strategy.option.gex import top_gex_walls


def _by_strike(rows):
    """rows: (strike, call_gex, put_gex)."""
    df = pd.DataFrame(rows, columns=["option_strike_price", "call_gex", "put_gex"])
    df["net_gex"] = df["call_gex"] + df["put_gex"]
    return df


# 40 holds the biggest call gamma but is nearly cancelled by puts written
# there; 45 holds less call gamma and almost no offset. Separated mode must
# say 40, net mode must say 45.
OFFSET = _by_strike([
    (40.0, 1000.0, -950.0),     # net   +50
    (45.0,  600.0,  -50.0),     # net  +550
    (35.0,   10.0, -900.0),     # net  -890
    (30.0,   20.0, -300.0),     # net  -280
])


class TestNetAndSeparatedDisagree:

    def test_the_positive_side_moves(self):
        sep_c, _ = top_gex_walls(OFFSET, 1, net=False)
        net_c, _ = top_gex_walls(OFFSET, 1, net=True)
        assert sep_c[0]["strike"] == 40.0, "separated should follow call gamma"
        assert net_c[0]["strike"] == 45.0, "net should follow the residual"

    def test_the_reported_size_is_the_residual_not_the_gross(self):
        net_c, _ = top_gex_walls(OFFSET, 1, net=True)
        assert net_c[0]["gex"] == pytest.approx(550.0), (
            "net mode must report call+put at the strike, not the call leg")

    def test_the_negative_side_follows_the_same_rule(self):
        _, sep_p = top_gex_walls(OFFSET, 1, net=False)
        _, net_p = top_gex_walls(OFFSET, 1, net=True)
        # Separated ranks on put_gex alone: 35 (-900) beats 40 (-950)? No --
        # 40 is more negative, so separated picks 40. Net picks 35 (-890),
        # because 40's puts are cancelled by its calls.
        assert sep_p[0]["strike"] == 40.0
        assert net_p[0]["strike"] == 35.0

    def test_default_is_still_the_separated_view(self):
        """The flag has to default off -- every other GEX tool shows the
        separated walls, and disagreeing with them silently is the failure
        this toggle exists to avoid."""
        assert top_gex_walls(OFFSET, 1) == top_gex_walls(OFFSET, 1, net=False)


class TestNetKeepsTheExistingGuarantees:

    def test_a_strike_whose_net_is_zero_is_not_ranked(self):
        """_live_side drops zero-GEX strikes rather than ranking them last.
        Under net that bites harder: a strike can be busy on both sides and
        still net to nothing, and drawing a wall there would mark the one
        place with no residual to hedge."""
        df = _by_strike([(50.0, 500.0, -500.0), (55.0, 100.0, -10.0)])
        c, _ = top_gex_walls(df, 2, net=True)
        assert [w["strike"] for w in c] == [55.0], (
            "a strike that cancels exactly must not be drawn as a wall")

    def test_fewer_live_strikes_than_asked_yields_a_shorter_list(self):
        df = _by_strike([(50.0, 300.0, -10.0), (55.0, 100.0, -10.0)])
        c, p = top_gex_walls(df, 5, net=True)
        assert len(c) == 2 and len(p) == 0, (c, p)

    def test_order_is_strongest_first(self):
        c, p = top_gex_walls(OFFSET, 4, net=True)
        assert [w["gex"] for w in c] == sorted(
            [w["gex"] for w in c], reverse=True)
        assert [w["gex"] for w in p] == sorted([w["gex"] for w in p])
