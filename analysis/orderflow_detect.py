"""Order flow detection algorithms — iceberg and spoofing.

Pure Python + NumPy, no Qt dependency.  Imported by trade_viewer_qt.py and
directly testable without a display.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np

from core.time_utils import candle_start


def detect_icebergs(
    snaps: list[dict],
    bucket_to_idx: dict,
    bin_size: float,
    price_min: float,
    N_PRICE: int,
    cm: int,
    min_refreshes: int,
    vol_threshold: float,
) -> list[tuple]:
    """Detect iceberg orders from order book snapshot series.

    A price bin is flagged as an iceberg when its resting volume repeatedly
    drops (depleted by aggressive orders) then refreshes — the hallmark of a
    hidden large order refilling at a fixed price.

    Args:
        snaps:          List of order book snapshots {ts, side, price, volume}.
        bucket_to_idx:  Mapping of bar-start datetime -> bar index.
        bin_size:       Price bin width in dollars.
        price_min:      Bottom of the price range.
        N_PRICE:        Number of price bins.
        cm:             Candle duration in minutes.
        min_refreshes:  Minimum refresh count to classify as iceberg.
        vol_threshold:  Minimum volume per snapshot to include.

    Returns:
        List of (first_bar_idx, last_bar_idx, price, refresh_count).
    """
    DROP_FRAC    = 0.25
    RECOVER_FRAC = 0.40

    groups: dict[int, list] = defaultdict(list)
    for s in snaps:
        if s["volume"] < vol_threshold:
            continue
        bar_idx = bucket_to_idx.get(candle_start(s["ts"], cm), -1)
        if bar_idx < 0:
            continue
        p_bin = int((s["price"] - price_min) / bin_size)
        if not (0 <= p_bin < N_PRICE):
            continue
        groups[p_bin].append((s["ts"], bar_idx, float(s["volume"])))

    icebergs = []
    for p_bin, entries in groups.items():
        if len(entries) < min_refreshes * 2 + 1:
            continue
        entries.sort()
        bar_idxs = [e[1] for e in entries]
        vols     = np.array([e[2] for e in entries], dtype=np.float64)

        running_peak = 0.0
        depleted     = False
        refreshes: list[int] = []

        for i, v in enumerate(vols):
            running_peak = max(running_peak, v)
            if running_peak <= 0:
                continue
            if not depleted and v < DROP_FRAC * running_peak:
                depleted = True
            elif depleted and v > RECOVER_FRAC * running_peak:
                depleted = False
                refreshes.append(i)

        if len(refreshes) < min_refreshes:
            continue

        price     = price_min + (p_bin + 0.5) * bin_size
        first_bar = bar_idxs[refreshes[0]]
        last_bar  = bar_idxs[refreshes[-1]]
        icebergs.append((first_bar, last_bar, price, len(refreshes)))

    return icebergs


def detect_spoofs(
    ob_data: list[dict],
    raw_ticks: list[dict],
    bucket_to_idx: dict,
    bin_size: float,
    price_min: float,
    N_PRICE: int,
    cm: int,
    min_vol: float,
    max_duration_secs: float,
) -> list[tuple]:
    """Detect spoofing events from order book snapshots cross-referenced with ticks.

    A spoof is a large order that appears at a price level then vanishes within
    max_duration_secs with little or no execution — creating false price pressure.

    Args:
        ob_data:            Order book snapshots {ts, side, price, volume}.
        raw_ticks:          Executed tick records {ts, price, volume}.
        bucket_to_idx:      Bar-start datetime -> bar index.
        bin_size:           Price bin width.
        price_min:          Bottom of price range.
        N_PRICE:            Number of price bins.
        cm:                 Candle duration in minutes.
        min_vol:            Minimum volume for an order to be considered large.
        max_duration_secs:  Max seconds between appearance and cancellation.

    Returns:
        List of (appear_bar, disappear_bar, price, side) where side is
        'BID' (spoofer pushing price up) or 'ASK' (pushing price down).
    """
    DISAPPEAR_FRAC = 0.20
    MAX_EXEC_RATIO = 0.30

    groups: dict[tuple, list] = defaultdict(list)
    for s in ob_data:
        bar_idx = bucket_to_idx.get(candle_start(s["ts"], cm), -1)
        if bar_idx < 0:
            continue
        p_bin = int((s["price"] - price_min) / bin_size)
        if not (0 <= p_bin < N_PRICE):
            continue
        groups[(s["side"], p_bin)].append((s["ts"], bar_idx, float(s["volume"])))

    sorted_ticks = sorted(raw_ticks, key=lambda t: t["ts"])

    spoofs = []
    for (side, p_bin), entries in groups.items():
        if len(entries) < 2:
            continue
        entries.sort()

        i = 0
        while i < len(entries):
            ts_i, bar_i, vol_i = entries[i]

            prev_vol = entries[i - 1][2] if i > 0 else 0.0
            if vol_i < min_vol or prev_vol >= min_vol * 0.5:
                i += 1
                continue

            appear_ts  = ts_i
            appear_vol = vol_i
            appear_bar = bar_i

            j = i + 1
            disappear_ts  = None
            disappear_bar = None
            while j < len(entries):
                ts_j, bar_j, vol_j = entries[j]
                if (ts_j - appear_ts).total_seconds() > max_duration_secs:
                    break
                if vol_j < appear_vol * DISAPPEAR_FRAC:
                    disappear_ts  = ts_j
                    disappear_bar = bar_j
                    break
                j += 1

            if disappear_ts is None:
                i += 1
                continue

            level_price = price_min + (p_bin + 0.5) * bin_size
            executed = sum(
                t["volume"] for t in sorted_ticks
                if (appear_ts <= t["ts"] <= disappear_ts
                    and abs(float(t["price"]) - level_price) < bin_size)
            )
            if executed > MAX_EXEC_RATIO * appear_vol:
                i = j + 1
                continue

            spoofs.append((appear_bar, disappear_bar, level_price, side))
            i = j + 1

    return spoofs
