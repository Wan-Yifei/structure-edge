"""Cron field matching, including the step syntax that was missing.

A job configured `5 */6 * * *` never fired. `int('*/6')` raises, and the
per-minute dedup that would have quietened it sat AFTER the match, so the
scheduler logged the same ValueError once a second for twenty hours while the
archive it was meant to write went unwritten.

Both halves are covered here: the field grammar, and that an unparseable
field is complained about once a minute rather than once a tick.
"""

import os
from datetime import datetime

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

import importlib.util                              # noqa: E402
import pathlib                                     # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sched_mod", pathlib.Path(__file__).parents[2] / "analysis" / "scheduler.py")
_sched = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sched)
match = _sched.SchedulerApp._cron_field_match


class TestTheGrammarThatAlreadyWorked:
    def test_star_matches_everything(self):
        assert all(match("*", v) for v in (0, 5, 23, 59))

    def test_a_bare_number(self):
        assert match("20", 20)
        assert not match("20", 21)

    def test_a_range(self):
        assert match("1-5", 1) and match("1-5", 5)
        assert not match("1-5", 0) and not match("1-5", 6)

    def test_a_comma_list(self):
        assert match("0,6,12,18", 12)
        assert not match("0,6,12,18", 13)


class TestStepSyntax:
    def test_every_sixth_hour(self):
        """The case that silently never fired."""
        for v in (0, 6, 12, 18):
            assert match("*/6", v), v
        for v in (1, 5, 7, 11, 13, 17, 19, 23):
            assert not match("*/6", v), v

    def test_every_fifteenth_minute(self):
        assert [v for v in range(60) if match("*/15", v)] == [0, 15, 30, 45]

    def test_a_stepped_range_counts_from_its_start(self):
        assert [v for v in range(24) if match("8-20/4", v)] == [8, 12, 16, 20]

    def test_a_step_inside_a_list(self):
        assert match("3,*/6", 3) and match("3,*/6", 12)
        assert not match("3,*/6", 4)

    def test_a_zero_step_is_ignored_not_a_crash(self):
        assert not match("*/0", 5)

    def test_step_one_is_every_value(self):
        assert all(match("*/1", v) for v in (0, 1, 7, 23))


class TestItDoesNotRaise:
    def test_the_old_failure_is_a_match_now(self):
        """int('*/6') used to raise out of _check_*_cron every tick."""
        assert match("*/6", 6) is True

    def test_a_genuinely_bad_field_still_raises(self):
        """Silently returning False would hide a typo forever."""
        with pytest.raises(ValueError):
            match("every-6", 6)


class TestOnceAMinuteNotOnceASecond:
    def test_the_minute_is_claimed_before_the_match(self):
        """So a bad cron complains once a minute, not 60 times."""
        src = (pathlib.Path(__file__).parents[2]
               / "analysis" / "scheduler.py").read_text(encoding="utf-8")
        for var in ("_last_ladder_minute", "_last_classify_minute"):
            block = src[src.index(f"if cur_min == self.{var}:"):]
            claim = block.index(f"self.{var} = cur_min")
            match_at = block.index("self._cron_field_match(")
            assert claim < match_at, f"{var}: claim must precede the match"
