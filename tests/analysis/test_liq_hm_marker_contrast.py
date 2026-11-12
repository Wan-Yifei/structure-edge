"""An overlay marker has to separate from the layer it is drawn on.

The imbalance bars are painted on top of the heatmap, and each side's bar
lands on its own side's colour ramp. The bullish lime bar always worked
because lime and teal are far apart. The bearish bar did not: Pink A200
(#ff4081) and the ask layer (#ef5350) are the same red family at a similar
lightness, so where the band was brightest -- which is exactly where a thick
resting level is, and where an imbalance is most likely to be flagged -- the
bar scored OKLab dE 15.7 against its background while the lime bar scored
37.4 against its own. It was reported as "the wine red is hard to make out",
and the measurement agreed.

Distance here is OKLab dE x100, the scale the dataviz validator uses: >= 8 is
the floor for two things that must be told apart, and a pair that must never
be confused wants >= 15. These tests pin three separations and the agreement
between what is painted and what the legend claims.

No Qt widget is constructed; only module-level colour constants are read.
"""
from __future__ import annotations

import math
import os
import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import analysis.liq_hm_window as lhw   # noqa: E402

BG = (13, 17, 23)
MARKER_ALPHA = 220      # _draw_simb_markers paints at this
BAND_ALPHA = 180        # _single_rgba caps alpha here


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _linear(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _oklab(rgb) -> tuple[float, float, float]:
    r, g, b = (_linear(x) for x in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def de(a, b) -> float:
    la, aa, ba = _oklab(a)
    lb, ab, bb = _oklab(b)
    return math.hypot(la - lb, aa - ab, ba - bb) * 100


def over(fg, alpha, bg):
    t = alpha / 255.0
    return tuple(bg[i] + (fg[i] - bg[i]) * t for i in range(3))


def _marker_cols() -> tuple[tuple, tuple]:
    """The two constants out of _draw_simb_markers' own source.

    They are locals, not module attributes, so there is nothing to import.
    Reading them from the source is what keeps this test measuring the
    colours that are actually painted rather than a copy that can drift.
    """
    src = pathlib.Path(lhw.__file__).read_text(encoding="utf-8")
    out = {}
    for name in ("_BID_COL", "_ASK_COL"):
        m = re.search(rf"{name}\s*=\s*\((\s*\d+,\s*\d+,\s*\d+)\s*\)", src)
        assert m, f"{name} not found -- did _draw_simb_markers change shape?"
        out[name] = tuple(int(x) for x in m.group(1).split(","))
    return out["_BID_COL"], out["_ASK_COL"]


BID_COL, ASK_COL = _marker_cols()
TEAL_BAND = over(_hex(lhw._TEAL), BAND_ALPHA, BG)
RED_BAND  = over(_hex(lhw._RED),  BAND_ALPHA, BG)


class TestEachBarSeparatesFromItsOwnBand(unittest.TestCase):

    FLOOR = 20.0   # comfortably above the dE 8 "tellable apart" floor

    def test_the_bullish_bar_stands_out_against_the_bid_layer(self):
        d = de(over(BID_COL, MARKER_ALPHA, TEAL_BAND), TEAL_BAND)
        self.assertGreater(d, self.FLOOR, f"lime on teal is only dE {d:.1f}")

    def test_the_bearish_bar_stands_out_against_the_ask_layer(self):
        """The one that failed: Pink A200 scored 15.7 here."""
        d = de(over(ASK_COL, MARKER_ALPHA, RED_BAND), RED_BAND)
        self.assertGreater(d, self.FLOOR, f"the bear bar is only dE {d:.1f}")

    def test_neither_bar_is_much_weaker_than_the_other(self):
        """The defect was an asymmetry, so the asymmetry is what is pinned:
        one side being readable does not excuse the other."""
        bull = de(over(BID_COL, MARKER_ALPHA, TEAL_BAND), TEAL_BAND)
        bear = de(over(ASK_COL, MARKER_ALPHA, RED_BAND), RED_BAND)
        self.assertGreater(min(bull, bear) / max(bull, bear), 0.5,
                           f"lopsided: bull dE {bull:.1f}, bear dE {bear:.1f}")


class TestTheTwoBarsCannotBeConfused(unittest.TestCase):

    def test_bull_and_bear_are_far_apart_as_rendered(self):
        """Direction is the whole content of this marker, so this pair gets
        the stricter floor: a reader who mixes them up reads the book
        backwards."""
        d = de(over(BID_COL, MARKER_ALPHA, TEAL_BAND),
               over(ASK_COL, MARKER_ALPHA, RED_BAND))
        self.assertGreater(d, 15.0, f"bull vs bear is only dE {d:.1f}")


class TestNeitherBarClashesWithSomethingElseOnTheChart(unittest.TestCase):
    """White is the price path, the crosshair and the wall text; purple is
    the iceberg line. A marker that lands on one of those reads as that
    thing, however well it separates from the heatmap."""

    TAKEN = {"price path / crosshair / wall text": (255, 255, 255),
             "iceberg": (224, 64, 251)}

    def test_no_marker_is_mistakable_for_another_overlay(self):
        for what, rgb in self.TAKEN.items():
            for name, col in (("bull", BID_COL), ("bear", ASK_COL)):
                d = de(col, rgb)
                self.assertGreater(d, 15.0,
                                   f"{name} bar is dE {d:.1f} from {what}")


class TestTheLegendSaysWhatIsPainted(unittest.TestCase):
    """Legend drift is a real failure mode here -- the guide described these
    bars as blue and orange for months while the code drew lime and pink."""

    def _legend_hexes(self) -> dict[str, str]:
        src = pathlib.Path(lhw.__file__).read_text(encoding="utf-8")
        out = {}
        for kind in ("Bullish", "Bearish"):
            m = re.search(rf'color="(#[0-9a-fA-F]{{6}})">\|\s*{kind} imbalance',
                          src)
            assert m, f"{kind} legend entry not found"
            out[kind] = m.group(1)
        return out

    def test_the_bullish_swatch_matches_the_bullish_bar(self):
        self.assertEqual(_hex(self._legend_hexes()["Bullish"]), BID_COL)

    def test_the_bearish_swatch_matches_the_bearish_bar(self):
        self.assertEqual(_hex(self._legend_hexes()["Bearish"]), ASK_COL)


if __name__ == "__main__":
    unittest.main()
