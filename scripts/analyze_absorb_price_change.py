"""Analyze price_change distribution in absorption bubble detection.

Runs detect_absorption_bubbles over all available data in order_book.db
and ticks.db, then prints statistics on price_change values at fired events.
"""

import sqlite3
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timedelta
import bisect as _bisect

ROOT = pathlib.Path(__file__).parent.parent
OB_DB   = ROOT / "db" / "order_book.db"
TICK_DB = ROOT / "db" / "ticks.db"

# ── Replicate detection logic inline ──────────────────────────────────────────

def detect_absorption_bubbles(ticks, col_ts, mid_prices, col_secs, min_delta_vol=100.0):
    if not ticks or not col_ts:
        return []
    ts_list = list(col_ts)
    col_buy  = defaultdict(float)
    col_sell = defaultdict(float)
    for tk in ticks:
        direction = tk.get("direction", "NEUTRAL")
        if direction not in ("BUY", "SELL"):
            continue
        tt  = tk["ts"]
        idx = _bisect.bisect_right(ts_list, tt) - 1
        if idx < 0:
            continue
        vol = float(tk["volume"])
        if direction == "BUY":
            col_buy[idx]  += vol
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

        price_change = None
        for j in range(i - 1, -1, -1):
            if mid_prices[j] is not None:
                price_change = mid - mid_prices[j]
                break

        if abs(delta) < min_delta_vol:
            continue

        if delta > 0 and (price_change is None or price_change <= 0):
            results.append((i, mid, "BUY", delta, price_change))
        elif delta < 0 and (price_change is None or price_change >= 0):
            results.append((i, mid, "SELL", abs(delta), price_change))

    return results


# ── Query helpers ──────────────────────────────────────────────────────────────

def query_codes(ob_db):
    con = sqlite3.connect(str(ob_db))
    rows = con.execute("SELECT DISTINCT code FROM order_book_snapshots").fetchall()
    con.close()
    return [r[0] for r in rows]


def query_ob_window(code, start, end, ob_db):
    con = sqlite3.connect(str(ob_db))
    cur = con.execute(
        "SELECT ts, side, price, volume FROM order_book_snapshots "
        "WHERE code=? AND ts>=? AND ts<=? ORDER BY ts",
        [code, start.isoformat(sep=" "), end.isoformat(sep=" ")],
    )
    rows = [{"ts": datetime.fromisoformat(r[0]), "side": r[1],
             "price": r[2], "volume": r[3]} for r in cur.fetchall()]
    con.close()
    return rows


def query_ticks_window(code, start, end, tick_db):
    con = sqlite3.connect(str(tick_db))
    cur = con.execute(
        "SELECT ts, price, volume, direction FROM ticks "
        "WHERE code=? AND ts>=? AND ts<=? ORDER BY ts",
        [code, start.isoformat(sep=" "), end.isoformat(sep=" ")],
    )
    rows = [{"ts": datetime.fromisoformat(r[0]), "price": r[1],
             "volume": r[2], "direction": r[3]} for r in cur.fetchall()]
    con.close()
    return rows


def build_columns(ob_rows, col_secs=30):
    """Group OB snapshots into columns, return (col_ts, mid_prices)."""
    if not ob_rows:
        return [], []
    # Find unique snapshot timestamps
    snap_times = sorted(set(r["ts"] for r in ob_rows))
    col_ts     = snap_times  # one column per snapshot

    mid_prices = []
    for t in col_ts:
        snap = [r for r in ob_rows if r["ts"] == t]
        bids = [r["price"] for r in snap if r["side"] == "BID"]
        asks = [r["price"] for r in snap if r["side"] == "ASK"]
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None
        if best_bid and best_ask:
            mid_prices.append((best_bid + best_ask) / 2)
        elif best_bid:
            mid_prices.append(best_bid)
        elif best_ask:
            mid_prices.append(best_ask)
        else:
            mid_prices.append(None)

    return col_ts, mid_prices


# ── Main analysis ──────────────────────────────────────────────────────────────

def main():
    col_secs = 30
    min_delta = 100.0

    # Get date range from OB DB
    con = sqlite3.connect(str(OB_DB))
    row = con.execute("SELECT MIN(ts), MAX(ts) FROM order_book_snapshots").fetchone()
    con.close()
    if not row or not row[0]:
        print("No data in order_book.db")
        sys.exit(1)

    db_start = datetime.fromisoformat(row[0])
    db_end   = datetime.fromisoformat(row[1])
    print(f"OB data range: {db_start} → {db_end}")

    codes = query_codes(OB_DB)
    print(f"Codes: {codes}")

    all_buy_pc  = []  # price_change values for BUY-absorbed events
    all_sell_pc = []  # price_change values for SELL-absorbed events

    for code in codes:
        print(f"\n=== {code} ===")
        ob_rows    = query_ob_window(code, db_start, db_end, OB_DB)
        tick_rows  = query_ticks_window(code, db_start, db_end, TICK_DB)
        print(f"  OB rows: {len(ob_rows)}, tick rows: {len(tick_rows)}")

        col_ts, mid_prices = build_columns(ob_rows, col_secs)
        print(f"  Columns: {len(col_ts)}")

        bubbles = detect_absorption_bubbles(
            tick_rows, col_ts, mid_prices, col_secs, min_delta_vol=min_delta
        )
        print(f"  Absorption events (min_delta={min_delta}): {len(bubbles)}")

        buy_abs  = [(b[4], b[3]) for b in bubbles if b[2] == "BUY"]
        sell_abs = [(b[4], b[3]) for b in bubbles if b[2] == "SELL"]
        print(f"  BUY absorbed: {len(buy_abs)}, SELL absorbed: {len(sell_abs)}")

        # price_change stats for each direction
        for label, events in [("BUY absorbed (gold)", buy_abs), ("SELL absorbed (purple)", sell_abs)]:
            pcs = [e[0] for e in events if e[0] is not None]
            if not pcs:
                continue
            pcs_sorted = sorted(pcs)
            n = len(pcs_sorted)
            mean_ = sum(pcs_sorted) / n
            med_  = pcs_sorted[n // 2]
            p10   = pcs_sorted[int(n * 0.10)]
            p25   = pcs_sorted[int(n * 0.25)]
            p75   = pcs_sorted[int(n * 0.75)]
            p90   = pcs_sorted[int(n * 0.90)]
            print(f"\n  {label} (n={n}):")
            print(f"    price_change  min={pcs_sorted[0]:.4f}  p10={p10:.4f}  p25={p25:.4f}")
            print(f"                 median={med_:.4f}  mean={mean_:.4f}")
            print(f"                 p75={p75:.4f}  p90={p90:.4f}  max={pcs_sorted[-1]:.4f}")

            # Bucket: how many are exactly 0 vs slightly off vs large
            zero    = sum(1 for p in pcs if p == 0.0)
            tiny    = sum(1 for p in pcs if 0.0 < abs(p) <= 0.05)
            small   = sum(1 for p in pcs if 0.05 < abs(p) <= 0.20)
            medium  = sum(1 for p in pcs if 0.20 < abs(p) <= 0.50)
            large   = sum(1 for p in pcs if abs(p) > 0.50)
            print(f"    Buckets:  ==0: {zero}  |Δ|≤0.05: {tiny}  0.05<|Δ|≤0.20: {small}"
                  f"  0.20<|Δ|≤0.50: {medium}  |Δ|>0.50: {large}")

            all_buy_pc.extend(pcs) if "BUY" in label else all_sell_pc.extend(pcs)

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    for label, pcs in [("BUY absorbed (gold)", all_buy_pc), ("SELL absorbed (purple)", all_sell_pc)]:
        if not pcs:
            continue
        pcs_sorted = sorted(pcs)
        n = len(pcs_sorted)
        zero   = sum(1 for p in pcs if p == 0.0)
        tiny   = sum(1 for p in pcs if 0.0 < abs(p) <= 0.05)
        small  = sum(1 for p in pcs if 0.05 < abs(p) <= 0.20)
        medium = sum(1 for p in pcs if 0.20 < abs(p) <= 0.50)
        large  = sum(1 for p in pcs if abs(p) > 0.50)
        print(f"\n{label} (n={n}):")
        print(f"  min={pcs_sorted[0]:.4f}  median={pcs_sorted[n//2]:.4f}  max={pcs_sorted[-1]:.4f}")
        print(f"  ==0: {zero} ({zero/n*100:.1f}%)  "
              f"|Δ|≤0.05: {tiny} ({tiny/n*100:.1f}%)  "
              f"0.05<|Δ|≤0.20: {small} ({small/n*100:.1f}%)  "
              f"0.20<|Δ|≤0.50: {medium} ({medium/n*100:.1f}%)  "
              f"|Δ|>0.50: {large} ({large/n*100:.1f}%)")


if __name__ == "__main__":
    main()
