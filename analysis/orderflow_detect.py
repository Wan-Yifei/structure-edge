"""Order flow detection algorithms — iceberg, spoofing, and absorption.

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
    best_bid: float | None = None,
    best_ask: float | None = None,
    max_spread_bins: int = 2,
    col_secs: int = 30,
) -> list[tuple]:
    """Detect iceberg orders from order book snapshot series.

    A price level is flagged as an iceberg when resting volume at the TOP OF
    BOOK repeatedly drops (depleted by aggressive executions) then refreshes.

    Only levels within max_spread_bins bins of the best bid (BID) or best ask
    (ASK) are considered — passive orders deep in the book are never consumed.

    Each contiguous appearance of a price level is treated as an independent
    candidate.  A gap > 1.5 × col_secs between consecutive snapshots means the
    level vanished from the book (breakthrough or cancellation); the state
    machine resets so any later reappearance starts a fresh iceberg search.

    Args:
        snaps:           Order book snapshots {ts, side, price, volume}.
        bucket_to_idx:   bar-start datetime -> column index.
        bin_size:        Price bin width in dollars.
        price_min:       Bottom of the price range.
        N_PRICE:         Number of price bins.
        cm:              Candle duration in minutes.
        min_refreshes:   Minimum refresh count to classify as iceberg.
        vol_threshold:   Minimum volume per snapshot to include.
        best_bid:        Current best bid price.
        best_ask:        Current best ask price.
        max_spread_bins: Bins away from best bid/ask allowed.
        col_secs:        Polling interval in seconds (used to detect gaps).

    Returns:
        List of (first_bar_idx, last_bar_idx, price, refresh_count).
        Multiple tuples per price level are possible if the level reappeared
        after a gap.
    """
    DROP_FRAC    = 0.25
    RECOVER_FRAC = 0.40
    GAP_SECS     = col_secs * 1.5   # gap larger than this → level vanished

    # Group by (side, price_bin) — sides must not be mixed
    groups: dict[tuple, list] = defaultdict(list)
    for s in snaps:
        if s["volume"] < vol_threshold:
            continue
        bar_idx = bucket_to_idx.get(candle_start(s["ts"], cm), -1)
        if bar_idx < 0:
            continue
        p_bin = int((s["price"] - price_min) / bin_size)
        if not (0 <= p_bin < N_PRICE):
            continue
        groups[(s["side"], p_bin)].append((s["ts"], bar_idx, float(s["volume"])))

    bid_bin = int((best_bid - price_min) / bin_size) if best_bid is not None else None
    ask_bin = int((best_ask - price_min) / bin_size) if best_ask is not None else None

    icebergs = []
    for (side, p_bin), entries in groups.items():
        # Only top-of-book levels — executions only happen there
        if side == "BID" and bid_bin is not None:
            if p_bin < bid_bin - max_spread_bins or p_bin > bid_bin + max_spread_bins:
                continue
        elif side == "ASK" and ask_bin is not None:
            if p_bin < ask_bin - max_spread_bins or p_bin > ask_bin + max_spread_bins:
                continue

        entries.sort()

        # Split into contiguous segments; a time gap means the level disappeared
        segments: list[list] = []
        seg: list = [entries[0]]
        for k in range(1, len(entries)):
            gap = (entries[k][0] - entries[k - 1][0]).total_seconds()
            if gap > GAP_SECS:
                segments.append(seg)
                seg = [entries[k]]
            else:
                seg.append(entries[k])
        segments.append(seg)

        price = price_min + (p_bin + 0.5) * bin_size

        # Each contiguous segment is an independent iceberg candidate
        for seg_entries in segments:
            if len(seg_entries) < min_refreshes * 2 + 1:
                continue
            bar_idxs = [e[1] for e in seg_entries]
            vols     = np.array([e[2] for e in seg_entries], dtype=np.float64)

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

            first_bar = bar_idxs[refreshes[0]]
            last_bar  = bar_idxs[refreshes[-1]]
            icebergs.append((first_bar, last_bar, price, len(refreshes)))

    return icebergs


def detect_spoofs(
    ob_data: list[dict],
    bucket_to_idx: dict,
    bin_size: float,
    price_min: float,
    N_PRICE: int,
    cm: int,
    min_vol: float,
    max_duration_secs: float,
) -> list[tuple]:
    """Detect spoofing events from order book snapshots.

    A spoof is a large order that appears at a price level then vanishes within
    max_duration_secs without being executed — creating false price pressure.

    Execution is inferred from spread movement: if the best bid (for a BID
    order) or best ask (for an ASK order) moved through the price level between
    appearance and disappearance, the order was likely filled, not cancelled.

    Args:
        ob_data:           Order book snapshots {ts, side, price, volume}.
        bucket_to_idx:     Bar-start datetime -> column index.
        bin_size:          Price bin width.
        price_min:         Bottom of price range.
        N_PRICE:           Number of price bins.
        cm:                Candle duration in minutes.
        min_vol:           Min volume to be considered a large order.
                           Pass 0 to auto-set to median of the latest snapshot.
        max_duration_secs: Max lifetime (seconds) of a genuine order; faster
                           disappearance flags as a spoof candidate.

    Returns:
        List of (appear_bar, disappear_bar, price, side).
        side='BID': spoofer pushing price up; side='ASK': pushing price down.
    """
    DISAPPEAR_FRAC = 0.20

    if not ob_data:
        return []

    # Auto min_vol: median volume of the most recent snapshot's levels
    if min_vol <= 0:
        latest_ts   = max(s["ts"] for s in ob_data)
        latest_vols = [s["volume"] for s in ob_data if s["ts"] == latest_ts]
        min_vol     = float(np.median(latest_vols)) if latest_vols else 1.0

    # Build best bid/ask per timestamp for execution-proxy check
    ts_bids: dict = {}
    ts_asks: dict = {}
    for s in ob_data:
        ts = s["ts"]
        p  = float(s["price"])
        if s["side"] == "BID":
            if ts not in ts_bids or p > ts_bids[ts]:
                ts_bids[ts] = p
        else:
            if ts not in ts_asks or p < ts_asks[ts]:
                ts_asks[ts] = p
    sorted_ts = sorted(set(ts_bids) | set(ts_asks))

    def _spread_moved_through(side: str, price: float,
                              t_start: datetime, t_end: datetime) -> bool:
        """True if the spread crossed *price* between t_start and t_end."""
        for ts in sorted_ts:
            if ts <= t_start:
                continue
            if ts > t_end:
                break
            if side == "BID":
                bb = ts_bids.get(ts)
                # Best bid fell below the level → order was consumed by sellers
                if bb is not None and bb < price - bin_size:
                    return True
            else:
                ba = ts_asks.get(ts)
                # Best ask rose above the level → order was consumed by buyers
                if ba is not None and ba > price + bin_size:
                    return True
        return False

    # Group by (side, price_bin)
    groups: dict[tuple, list] = defaultdict(list)
    for s in ob_data:
        bar_idx = bucket_to_idx.get(candle_start(s["ts"], cm), -1)
        if bar_idx < 0:
            continue
        p_bin = int((s["price"] - price_min) / bin_size)
        if not (0 <= p_bin < N_PRICE):
            continue
        groups[(s["side"], p_bin)].append((s["ts"], bar_idx, float(s["volume"])))

    spoofs = []
    for (side, p_bin), entries in groups.items():
        if len(entries) < 2:
            continue
        entries.sort()
        level_price = price_min + (p_bin + 0.5) * bin_size

        i = 0
        while i < len(entries):
            ts_i, bar_i, vol_i = entries[i]
            prev_vol = entries[i - 1][2] if i > 0 else 0.0

            # Order must appear suddenly (prev small) and be large
            if vol_i < min_vol or prev_vol >= min_vol * 0.5:
                i += 1
                continue

            appear_ts  = ts_i
            appear_vol = vol_i
            appear_bar = bar_i

            # Find disappearance within max_duration_secs
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

            # If the spread moved through the level the order was executed, not spoofed
            if _spread_moved_through(side, level_price, appear_ts, disappear_ts):
                i = j + 1
                continue

            spoofs.append((appear_bar, disappear_bar, level_price, side))
            i = j + 1

    return spoofs


def detect_absorption(
    ob_data: list[dict],
    raw_ticks: list[dict],
    bin_size: float,
    price_min: float,
    N_PRICE: int,
    passive_k: float,
    active_k: float,
    hit_ratio: float,
) -> list[tuple]:
    """Detect absorption: passive orders holding a price level against significant aggression.

    A price level is flagged when ALL three conditions hold within the window:
      1. Resting volume >= avg_tick_vol * passive_k  (wall is meaningfully large)
      2. Aggressive volume at that level >= avg_tick_vol * active_k  (real pressure applied)
      3. agg_vol / pass_vol >= hit_ratio  (a meaningful fraction of the wall was attempted)
      4. Resting volume at window end > 0  (implicit: level still present = not broken through)

    Aggressive volume accumulates across all ticks at the price bin — small split orders
    are counted together, preventing detection evasion through order fragmentation.

    Args:
        ob_data:    Order book snapshots {ts, side, price, volume} pre-filtered to window,
                    sorted ascending by ts.
        raw_ticks:  Tick records {ts, price, volume, direction} pre-filtered to window.
        bin_size:   Price bin width (use tick size of the instrument).
        price_min:  Lower bound of the price range grid.
        N_PRICE:    Number of price bins.
        passive_k:  Resting volume must be >= avg_tick_vol * passive_k.
        active_k:   Aggressive volume must be >= avg_tick_vol * active_k.
        hit_ratio:  agg_vol / pass_vol must be >= this value (0–1).

    Returns:
        List of (price, side, agg_vol, pass_vol, ratio).
        side='ASK': sellers absorbing buyers (ask wall held).
        side='BID': buyers absorbing sellers (bid wall held).
    """
    if not ob_data or not raw_ticks:
        return []

    # Average volume per tick — instrument/session-adaptive baseline for thresholds
    total_vol = sum(t["volume"] for t in raw_ticks)
    n_ticks   = len(raw_ticks)
    if n_ticks == 0:
        return []
    avg_tick_vol      = total_vol / n_ticks
    passive_threshold = avg_tick_vol * passive_k
    active_threshold  = avg_tick_vol * active_k

    # Last-seen resting volume per (side, price_bin) within the window.
    # ob_data must be time-sorted so the final value reflects end-of-window state.
    last_pass: dict[tuple, float] = {}
    for s in ob_data:
        p_bin = int((s["price"] - price_min) / bin_size)
        if 0 <= p_bin < N_PRICE:
            last_pass[(s["side"], p_bin)] = float(s["volume"])

    # Accumulate aggressive tick volume per price bin (split orders sum naturally)
    agg_buy:  dict[int, float] = defaultdict(float)
    agg_sell: dict[int, float] = defaultdict(float)
    for t in raw_ticks:
        p_bin = int((t["price"] - price_min) / bin_size)
        if not (0 <= p_bin < N_PRICE):
            continue
        if t["direction"] == "BUY":
            agg_buy[p_bin]  += t["volume"]
        elif t["direction"] == "SELL":
            agg_sell[p_bin] += t["volume"]

    results = []

    # ASK absorption: buyers hitting a sell wall that held
    for p_bin, agg in agg_buy.items():
        pass_vol = last_pass.get(("ASK", p_bin), 0.0)
        if pass_vol < passive_threshold:
            continue
        if agg < active_threshold:
            continue
        ratio = agg / pass_vol
        if ratio < hit_ratio:
            continue
        price = price_min + (p_bin + 0.5) * bin_size
        results.append((price, "ASK", agg, pass_vol, ratio))

    # BID absorption: sellers hitting a buy wall that held
    for p_bin, agg in agg_sell.items():
        pass_vol = last_pass.get(("BID", p_bin), 0.0)
        if pass_vol < passive_threshold:
            continue
        if agg < active_threshold:
            continue
        ratio = agg / pass_vol
        if ratio < hit_ratio:
            continue
        price = price_min + (p_bin + 0.5) * bin_size
        results.append((price, "BID", agg, pass_vol, ratio))

    return results
