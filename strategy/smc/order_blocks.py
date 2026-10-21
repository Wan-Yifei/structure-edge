"""SMC Order Block detection.

An Order Block (OB) is the last opposing candle immediately before a
significant move (identified by a BOS/CHoCH signal).

Subtypes:
    regular    — price has not returned to the zone
    mitigation — price re-entered the zone but has not closed through it
    breaker    — price closed through the zone (former support → resistance)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _overlaps(a: dict, b: dict) -> bool:
    """Do two blocks share any price?"""
    return min(a["top"], b["top"]) > max(a["bottom"], b["bottom"])


def _merge_overlapping(blocks: list[dict]) -> list[dict]:
    """Collapse overlapping same-direction blocks down to the strongest.

    Greedy from the strongest: a block is kept unless it overlaps one
    already kept in its own direction. Bull and bear blocks are never
    merged into each other -- a demand zone and a supply zone at the same
    price are two readings of that price, not one.
    """
    out: list[dict] = []
    for b in sorted(blocks, key=lambda x: -x.get("vol_ratio", 1.0)):
        if any(k["direction"] == b["direction"] and _overlaps(k, b)
               for k in out):
            continue
        out.append(b)
    return out


def _side(b: dict, ref: float) -> str:
    """Which side of the reference price this block sits on."""
    if b["bottom"] > ref:
        return "above"
    if b["top"] < ref:
        return "below"
    return "inside"


def _by_strength(blocks: list[dict], max_count: int, max_breakers: int | None,
                 ref_price: float) -> list[dict]:
    """Quota per side of ref_price, each with its own breaker allowance.

    A single global quota is what emptied the lower half: whichever side
    price last visited takes every slot. The same happens to breakers if
    they are capped globally -- on a chart where most blocks are broken,
    the two most recent are both on one side and the other side goes
    empty again. So both quotas are per side.

    Blocks price is currently inside are always kept: those are the ones
    being tested right now, and there are never many.
    """
    live = _merge_overlapping(
        [b for b in blocks if b["subtype"] != "breaker"])
    broken = [b for b in blocks if b["subtype"] == "breaker"]

    half = max(1, max_count // 2)
    by_strength = lambda b: -b.get("vol_ratio", 1.0)       # noqa: E731
    by_recency = lambda b: b["idx"]                        # noqa: E731

    keep = [b for b in live if _side(b, ref_price) == "inside"]
    for side in ("above", "below"):
        keep += sorted([b for b in live if _side(b, ref_price) == side],
                       key=by_strength)[:half]
        side_broken = [b for b in broken if _side(b, ref_price) == side]
        if max_breakers is None:
            keep += side_broken
        elif max_breakers > 0:
            keep += sorted(side_broken, key=by_recency)[-max_breakers:]
    # Breakers price sits inside are the live retest, so they are never cut.
    keep += [b for b in broken if _side(b, ref_price) == "inside"]
    return sorted(keep, key=lambda b: b["idx"])


def detect_order_blocks(
    klines: pd.DataFrame,
    bos_signals: list[dict],
    max_count: int = 4,
    select: str = "recent",
    max_breakers: int | None = None,
    ref_price: float | None = None,
) -> list[dict]:
    """Return Order Blocks derived from BOS/CHoCH signals.

    Each block dict:
        {direction: 'bull'|'bear',
         subtype:   'regular'|'mitigation'|'breaker',
         top:    float, bottom: float,
         vol_ratio: float,  # OB candle volume / median of the 5 before it
         idx:    int,   # candle index of the OB
         bos_idx: int}  # candle index of the triggering BOS/CHoCH

    select "recent" keeps the last `max_count` in signal order, which is
    the original behaviour and what backtest/engine.py gets. "strength"
    merges overlapping same-direction blocks to the strongest by
    vol_ratio, then takes half the quota from above `ref_price` and half
    from below, so neither side can be starved by the other.

    max_breakers caps blocks price has already closed through, PER SIDE of
    ref_price. None keeps them all; a small number keeps only the most
    recent on each side, which is enough to watch a retest from either
    direction without burying the untested blocks under them.

    ref_price defaults to the last close, i.e. "which side of now".
    """
    opens   = klines["open"].values
    closes  = klines["close"].values
    highs   = klines["high"].values
    lows    = klines["low"].values
    volumes = klines["volume"].values
    n       = len(klines)
    blocks  = []

    for sig in bos_signals:
        bos_idx = sig["idx"]
        bull    = sig["direction"] == "bull"

        # OB is anchored at the reference swing (from_idx), not the break bar.
        # Scan backwards from the swing point to find the last opposing candle
        # in the structure that created that swing.
        anchor = min(sig.get("from_idx", bos_idx - 1), n - 1)
        ob_idx = None
        for k in range(anchor, max(-1, anchor - 8), -1):
            is_bear_candle = closes[k] < opens[k]
            is_bull_candle = closes[k] > opens[k]
            if bull and is_bear_candle:
                ob_idx = k
                break
            if not bull and is_bull_candle:
                ob_idx = k
                break
        # fallback: scan forward from swing toward break bar
        if ob_idx is None:
            for k in range(anchor + 1, bos_idx):
                is_bear_candle = closes[k] < opens[k]
                is_bull_candle = closes[k] > opens[k]
                if bull and is_bear_candle:
                    ob_idx = k
                    break
                if not bull and is_bull_candle:
                    ob_idx = k
                    break

        if ob_idx is None:
            continue

        # Volume filter: OB candle must exceed the median of the 5 preceding bars.
        # Low-volume OBs are more likely to be noise rather than institutional flow.
        vol_start  = max(0, ob_idx - 5)
        prior_vols = volumes[vol_start:ob_idx]
        med = float(np.median(prior_vols)) if len(prior_vols) else 0.0
        if med > 0 and float(volumes[ob_idx]) < med:
            continue
        # Kept, not just tested against: this ratio is the only measure of
        # an OB's significance the detector has, and selecting by strength
        # needs it downstream.
        vol_ratio = (float(volumes[ob_idx]) / med) if med > 0 else 1.0

        top    = float(highs[ob_idx])
        bottom = float(lows[ob_idx])

        # classify subtype based on price action after BOS
        subtype = "regular"
        for j in range(bos_idx + 1, n):
            c = closes[j]
            if bull:
                if c < bottom:          # closed below OB → breaker
                    subtype = "breaker"
                    break
                if bottom <= c <= top:  # entered zone → mitigation
                    subtype = "mitigation"
            else:
                if c > top:             # closed above OB → breaker
                    subtype = "breaker"
                    break
                if bottom <= c <= top:
                    subtype = "mitigation"

        blocks.append({
            "direction": sig["direction"],
            "subtype":   subtype,
            "top":       top,
            "bottom":    bottom,
            "vol_ratio": vol_ratio,
            "idx":       ob_idx,
            "bos_idx":   bos_idx,
        })

    if max_count <= 0:
        return blocks
    if select == "strength":
        if ref_price is None:
            ref_price = float(closes[-1]) if n else 0.0
        return _by_strength(blocks, max_count, max_breakers, ref_price)
    # "recent": the original rule -- last N in signal order.
    return blocks[-max_count:]
