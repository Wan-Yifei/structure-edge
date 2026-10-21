"""Fixed ATR-multiple stops, as a plain calculation.

The chandelier exit hangs its stop off the highest high of the last `period`
bars and ratchets as that high advances. This is the other kind: a stop a
fixed multiple of ATR away from one anchor price, which does not move.

They answer different questions. The chandelier asks "where should the stop
be NOW, given how far the trade has run"; this asks "if I enter here, where
does the stop go", which is the number needed before the trade exists and
the one a position size is computed from.

wilder_atr is imported from the chandelier package rather than reimplemented
here. The point of having both methods is to compare them, and two ATRs that
differ in their warmup would make the comparison about the ATR instead of
about the stop.

    >>> r = fixed_stops(h, l, c, 14, [1.0, 1.5, 2.0], "bull")
    >>> r["levels"][0]["stop"]
"""
from __future__ import annotations

import numpy as np

from strategy.chandelier_exit.atr import wilder_atr


def fixed_stops(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    period: int, multipliers: list[float], direction: str = "bull",
    anchor: float | None = None,
) -> dict | None:
    """Stop prices at each ATR multiple from `anchor`.

    anchor defaults to the latest close -- "if I entered right now". Pass the
    real fill price to see where the stop belongs for a position already
    open; the ATR still comes from the latest bar, which is what makes the
    distance current rather than historical.

    direction "bull" puts the stop below the anchor, "bear" above it.

    Returns None when there are too few bars for the ATR warmup, so a caller
    can say so instead of drawing a line at an invented price. Otherwise:

        {"atr": float, "anchor": float, "period": int, "direction": str,
         "levels": [{"mult", "stop", "dist", "dist_pct"}, ...]}

    `dist` is always positive -- the gap between anchor and stop. `dist_pct`
    is that gap as a percentage of the anchor, which is the number that says
    how much of the account one R costs at a given position size.
    """
    if len(highs) < period or period < 2:
        return None
    atr = wilder_atr(highs, lows, closes, period)
    if not len(atr) or not np.isfinite(atr[-1]) or atr[-1] <= 0:
        return None

    atr_v = float(atr[-1])
    base = float(closes[-1]) if anchor is None else float(anchor)
    if not np.isfinite(base) or base <= 0:
        return None
    bull = direction == "bull"

    levels = []
    for m in multipliers:
        m = float(m)
        if m <= 0:
            continue
        off = atr_v * m
        stop = base - off if bull else base + off
        levels.append({"mult": m, "stop": stop, "dist": off,
                       "dist_pct": off / base * 100.0})
    if not levels:
        return None
    return {"atr": atr_v, "anchor": base, "period": period,
            "direction": direction, "levels": levels}


def parse_multipliers(text: str) -> list[float]:
    """Read "1, 1.5, 2" into [1.0, 1.5, 2.0].

    Tolerant on purpose: this parses a text field someone types into, so
    stray spaces, a trailing comma, full-width commas pasted from Chinese
    input, and duplicates all have to survive. Anything that is not a
    positive number is dropped rather than raising -- a typo should cost one
    line, not the whole overlay.
    """
    out: list[float] = []
    for part in (text or "").replace("，", ",").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            continue
        if v > 0 and v not in out:
            out.append(v)
    return sorted(out)
