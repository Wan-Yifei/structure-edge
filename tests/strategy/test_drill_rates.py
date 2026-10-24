"""The rule-light classifier that feeds the page its own base rates.

`light_of` has to agree with ruleLight in strategy/drill/index.html: the
page decides the colour a question shows, this function counts how that
colour has done, and if the two drift the page quotes one rule's record
beside another rule's light. These cases are the ones the two
implementations could plausibly disagree on.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from strategy.drill_rates import light_of  # noqa: E402


def q(series, mark=None):
    """A question stub carrying only what the classifier reads."""
    return {"drs1": series, "mark": len(series) - 1 if mark is None else mark}


class TestLightOf:
    def test_negative_is_red_whatever_came_before(self):
        assert light_of(q([0.9, 0.8, -0.01])) == "red"
        assert light_of(q([-0.5, -0.4, -0.3])) == "red"

    def test_flip_on_the_bar_is_green(self):
        assert light_of(q([0.1, -0.2, 0.55])) == "green"

    def test_flip_one_bar_back_is_green(self):
        """The widening. Reaching the threshold a bar after the turn
        measured the same as reaching it on the turn, so the bar it lands
        on does not change the colour."""
        assert light_of(q([-0.2, 0.15, 0.55])) == "green"

    def test_two_bars_back_is_not(self):
        """The allowance is one bar, not "sometime recently" -- with no
        bound this would become "dr1 >= 0.4 and a negative anywhere"."""
        assert light_of(q([-0.2, 0.15, 0.2, 0.55])) == "amber"

    def test_no_flip_behind_it_is_amber(self):
        """0.4 on its own is the set's own base rate; the flip is the part
        that carried anything."""
        assert light_of(q([0.5, 0.6, 0.55])) == "amber"

    def test_the_middle_bar_must_be_short_of_the_threshold(self):
        """If the bar after the flip already cleared 0.4 then THAT was the
        green bar; this one is a continuation, not a turn."""
        assert light_of(q([-0.2, 0.45, 0.55])) == "amber"

    def test_below_the_threshold_is_amber_even_with_a_flip(self):
        assert light_of(q([-0.2, 0.3])) == "amber"

    def test_missing_data_is_not_a_colour(self):
        assert light_of(q([0.1, 0.2, None])) is None
        assert light_of({"drs1": None, "mark": 3}) is None
        assert light_of({"mark": 3}) is None

    def test_it_does_not_run_off_the_front_of_the_series(self):
        """mark 0 and 1 have no two bars behind them; the lookback must
        return a colour rather than an IndexError."""
        assert light_of(q([0.55], mark=0)) == "amber"
        assert light_of(q([-0.2, 0.55], mark=1)) == "green"
