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


def detect_stacked_imbalance(
    snaps: list[dict],
    bucket_to_idx: dict,
    bin_size: float,
    price_min: float,
    N_PRICE: int,
    cm: int,
    min_levels: int = 3,
    imbalance_ratio: float = 3.0,
    min_vol: float = 0.0,
    max_depth: int = 10,
) -> list[tuple]:
    """Detect stacked imbalance from order book snapshots.

    Bid and ask levels in each snapshot are paired by depth rank (rank 0 =
    best bid / best ask, rank 1 = second-best, etc.).  A run of min_levels
    consecutive ranks where bid_vol / ask_vol >= imbalance_ratio is a bullish
    stacked imbalance; ask_vol / bid_vol >= imbalance_ratio is bearish.

    Only the top max_depth ranks are considered.  Levels beyond max_depth are
    ignored because deep-book orders are far from the current price and do not
    reflect near-term market pressure.

    Missing levels on either side within max_depth are treated as 0 volume
    rather than truncating the rank list — an unpaired level registers as an
    infinite ratio and counts as fully imbalanced.

    Each snapshot is evaluated independently; persistence across snapshots is
    a viewer-layer concern.

    Args:
        snaps:            Order book snapshots {ts, side, price, volume}.
        bucket_to_idx:    Bar-start datetime -> column index.
        bin_size:         Price bin width.
        price_min:        Bottom of price range.
        N_PRICE:          Number of price bins.
        cm:               Candle duration in minutes.
        min_levels:       Minimum consecutive imbalanced depth levels.
        imbalance_ratio:  bid/ask (or ask/bid) threshold per level.
        min_vol:          Levels below this volume are treated as absent (0).
        max_depth:        Maximum depth ranks to analyse (top-of-book only).

    Returns:
        List of (bar_idx, price_lo, price_hi, direction, mean_ratio).
        direction='BID': buyers dominating — zone spans bid-side prices.
        direction='ASK': sellers dominating — zone spans ask-side prices.
    """
    if not snaps:
        return []

    EPS = 1e-9

    by_ts: dict = defaultdict(lambda: {"BID": [], "ASK": []})
    for s in snaps:
        vol = float(s["volume"])
        if vol < min_vol:
            continue
        by_ts[s["ts"]][s["side"]].append((float(s["price"]), vol))

    results = []
    for ts, sides in by_ts.items():
        bar_idx = bucket_to_idx.get(candle_start(ts, cm), -1)
        if bar_idx < 0:
            continue

        bids = sorted(sides["BID"], key=lambda x: -x[0])   # best bid first
        asks = sorted(sides["ASK"], key=lambda x:  x[0])   # best ask first

        n_ranks = min(max(len(bids), len(asks)), max_depth)
        if n_ranks == 0:
            continue

        bid_vols   = [bids[i][1] if i < len(bids) else 0.0 for i in range(n_ranks)]
        ask_vols   = [asks[i][1] if i < len(asks) else 0.0 for i in range(n_ranks)]
        bid_prices = [bids[i][0] if i < len(bids) else None for i in range(n_ranks)]
        ask_prices = [asks[i][0] if i < len(asks) else None for i in range(n_ranks)]

        rank_dir: list[str | None] = []
        rank_ratio: list[float]    = []
        for bv, av in zip(bid_vols, ask_vols):
            if bv >= av * imbalance_ratio and bv > EPS:
                rank_dir.append("BID")
                rank_ratio.append(bv / (av + EPS))
            elif av >= bv * imbalance_ratio and av > EPS:
                rank_dir.append("ASK")
                rank_ratio.append(av / (bv + EPS))
            else:
                rank_dir.append(None)
                rank_ratio.append(1.0)

        i = 0
        while i < n_ranks:
            d = rank_dir[i]
            if d is None:
                i += 1
                continue

            j = i + 1
            while j < n_ranks and rank_dir[j] == d:
                j += 1

            if j - i >= min_levels:
                mean_ratio = float(np.mean(rank_ratio[i:j]))
                prices = (
                    [bid_prices[k] for k in range(i, j) if bid_prices[k] is not None]
                    if d == "BID"
                    else [ask_prices[k] for k in range(i, j) if ask_prices[k] is not None]
                )
                if prices:
                    results.append((bar_idx, min(prices), max(prices), d, mean_ratio))

            i = j

    return results


def detect_absorption_bubbles(
    ticks: list[dict],
    col_ts: list[datetime],
    mid_prices: list[float | None],
    col_secs: int,
    min_delta_vol: float = 500.0,
) -> list[tuple]:
    """Detect columns where aggressive order flow is absorbed by passive orders.

    For each column time window the net delta (buy_vol − sell_vol) is compared
    against the mid-price movement to that column:

    - delta >= +min_delta_vol AND price_change <= 0  →  'BUY' absorbed
      (aggressive buyers absorbed by passive sell wall — bearish)
    - delta <= -min_delta_vol AND price_change >= 0  →  'SELL' absorbed
      (aggressive sellers absorbed by passive buy wall — bullish)

    Args:
        ticks:          Tick records {ts, price, volume, direction}.
                        direction must be 'BUY', 'SELL', or 'NEUTRAL'.
        col_ts:         Per-column timestamps (col_ts[i] = snapshot time of column i).
        mid_prices:     Per-column mid-price (None if unavailable for that column).
        col_secs:       Column duration in seconds; determines bucket half-width.
        min_delta_vol:  Minimum |delta| required to flag as absorption.

    Returns:
        List of (col_idx, mid_price, direction, delta_vol).
        direction: 'BUY'  = aggressive buyers absorbed (passive sellers won).
                   'SELL' = aggressive sellers absorbed (passive buyers won).
    """
    if not ticks or not col_ts:
        return []

    # Build a sorted list of col_ts for binary search
    ts_list = list(col_ts)

    # Bucket each tick into the nearest column (bisect_right gives col after tick;
    # subtract 1 to get the column whose ts <= tick.ts)
    col_buy: dict[int, float] = defaultdict(float)
    col_sell: dict[int, float] = defaultdict(float)
    half = col_secs / 2.0

    for tk in ticks:
        direction = tk.get("direction", "NEUTRAL")
        if direction not in ("BUY", "SELL"):
            continue
        tt = tk["ts"]
        # Find the column whose window contains this tick
        # Window for column i: [col_ts[i] - half, col_ts[i] + half)
        import bisect as _bisect
        idx = _bisect.bisect_right(ts_list, tt) - 1
        if idx < 0:
            # Tick predates all columns — try column 0 if within half-window
            if (ts_list[0] - tt).total_seconds() <= half:
                idx = 0
            else:
                continue
        elif idx >= len(ts_list):
            idx = len(ts_list) - 1

        # Verify tick falls within the half-window of that column
        dt = abs((tt - ts_list[idx]).total_seconds())
        if dt > half:
            continue

        vol = float(tk["volume"])
        if direction == "BUY":
            col_buy[idx] += vol
        else:
            col_sell[idx] += vol

    results = []
    all_cols = set(col_buy) | set(col_sell)
    for i in sorted(all_cols):
        if i >= len(col_ts) or i >= len(mid_prices):
            continue
        mid = mid_prices[i]
        if mid is None:
            continue

        buy_vol  = col_buy.get(i, 0.0)
        sell_vol = col_sell.get(i, 0.0)
        delta    = buy_vol - sell_vol

        # Price change vs previous column with valid mid
        price_change: float | None = None
        for j in range(i - 1, -1, -1):
            if mid_prices[j] is not None:
                price_change = mid - mid_prices[j]
                break

        if abs(delta) < min_delta_vol:
            continue

        if delta > 0 and (price_change is None or price_change <= 0):
            results.append((i, mid, "BUY", delta))
        elif delta < 0 and (price_change is None or price_change >= 0):
            results.append((i, mid, "SELL", abs(delta)))

    return results
