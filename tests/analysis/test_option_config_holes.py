"""Two holes the fourth review found in the multi-symbol config path.

Both are the SAME bug the multi-symbol work was undertaken to remove --
pressing a button silently changes which symbol gets collected -- surviving
inside the function written to remove it.

1. `_option_codes` read `code` only when `codes` was *absent*. A config
   holding both spellings with an empty list -- {"codes": [], "code":
   "US.QQQ"} -- therefore never looked at `code` and fell through to the
   default US.SOXL. A hand-edited config is exactly where both spellings
   coexist, because that is what an older file plus a newer save produces.

2. The shape check `^[A-Z]{2}\\.[A-Z0-9.]+$` accepted a symbol made of
   nothing but dots. "US.." passes, becomes a collector argument, a lock
   file named option_tick_collector_US__ and a code OpenD cannot resolve.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.scheduler import SchedulerApp                      # noqa: E402

_codes = SchedulerApp._option_codes
DEFAULT = SchedulerApp._DEFAULT_OPTION["codes"]


class TestAnEmptyCodesListDoesNotHideACode:

    def test_an_empty_list_falls_through_to_the_other_spelling(self):
        assert _codes({"codes": [], "code": "US.QQQ"}) == ["US.QQQ"]

    def test_an_empty_string_falls_through_too(self):
        assert _codes({"codes": "", "code": "US.QQQ"}) == ["US.QQQ"]

    def test_the_symbol_is_not_the_default(self):
        """The assertions above would hold if `code` were ignored and the
        default happened to be US.QQQ. It is not -- but a later change to
        _DEFAULT_OPTION could make it so, and then they would pass for the
        wrong reason. This fails the moment that happens."""
        assert DEFAULT == ["US.SOXL"], (
            "the examples above use US.QQQ precisely because it is NOT the "
            "default; pick another symbol if this changes")

    def test_a_populated_list_still_wins_over_code(self):
        """The precedence is unchanged: `codes` is the newer spelling."""
        assert _codes({"codes": ["US.QQQ"], "code": "US.SOXL"}) == ["US.QQQ"]

    def test_neither_spelling_still_gives_the_default(self):
        assert _codes({}) == DEFAULT

    def test_both_empty_still_gives_the_default(self):
        assert _codes({"codes": [], "code": ""}) == DEFAULT

    def test_the_write_side_sentinel_is_unaffected(self):
        """The save path passes fallback=None so it can REFUSE rather than
        write the default over a real config. An empty entry box must still
        come back empty there, not as US.SOXL."""
        # The sentinel is returned verbatim -- None, not []. The save
        # path tests it with `if typed:`, so both are a refusal; the point
        # here is that it is NOT the default.
        assert _codes({"codes": "", "code": ""}, fallback=None) is None
        # And with a usable `code` it is not a refusal at all.
        assert _codes({"codes": "", "code": "US.QQQ"},
                      fallback=None) == ["US.QQQ"]


class TestASymbolNeedsCharacters:

    @pytest.mark.parametrize("junk", ["US..", "US.....", "US..SOXL"])
    def test_an_all_dots_symbol_is_rejected(self, junk):
        assert _codes({"codes": junk}) == DEFAULT, junk

    def test_a_trailing_dot_is_rejected(self):
        assert _codes({"codes": "US.SOXL."}) == DEFAULT

    @pytest.mark.parametrize("ok", ["US.SOXL", "US.QQQ", "US.BRK.B",
                                    "HK.00700", "US.SPY"])
    def test_real_shapes_still_pass(self, ok):
        """The tightened pattern must not start refusing legitimate codes.
        US.BRK.B is the case that keeps the inner dot legal."""
        assert _codes({"codes": ok}) == [ok], ok

    def test_the_rejection_is_not_silent_to_the_caller(self):
        """Rejected junk falls back, so a caller that passes the write-side
        sentinel sees the refusal rather than a substituted symbol."""
        assert _codes({"codes": "US.."}, fallback=None) is None
