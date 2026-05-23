# SMC Strategy — Version 1 (`smc_v1`)

**Git tag:** `v0.0.1`  
**Algo version constant:** `ALGO_VERSION = "smc_v1"` in `backtest/engine.py`  
**Engine file:** `backtest/engine.py`  
**Strategy modules:** `strategy/smc/`

---

## Overview

Multi-timeframe Smart Money Concepts (SMC) strategy. A higher timeframe (HTF)
defines trend and marks Fair Value Gaps (FVGs); a lower timeframe (LTF) detects
when price pulls back into an FVG and triggers entry.

---

## Execution Flow

### 1. Initialisation (once per backtest run)

- Parse `ltf_per_htf = htf_minutes / ltf_minutes` from the TF pair (e.g. 15m/1m → 15).
- If `require_ltf_confirmation=True`, pre-compute LTF BOS/CHoCH signals for the
  entire LTF series using `detect_bos_choch(ltf, lookback=1, trend_window=ltf_per_htf)`.
  Signals are sorted by bar index so later window-slicing is O(log n).
- If `intraday_only=True`, pre-compute the last bar index of each calendar day.

### 2. HTF Analysis (re-run every new HTF bar)

Triggered when `searchsorted(htf_times, current_ltf_time)` returns a new HTF bar index.

| Step | Function | Output |
|------|----------|--------|
| Slice window | `htf.iloc[htf_pos+1-htf_window_bars : htf_pos+1]` | `htf_view` (≤ `htf_window_bars` rows) |
| Swing points | `find_swings(htf_view, swing_lookback)` | alternating highs / lows |
| Structure breaks | `detect_bos_choch(htf_view, swing_lookback, trend_window=htf_window_bars)` | BOS / CHoCH list |
| Trend | `determine_trend(htf_bos, bos_count)` | `"bull"` / `"bear"` / `None` |
| FVGs | `detect_fvg(htf_view, fvg_min_width_pct)` | list of open FVG zones |
| Volume profile | `compute_volume_profile(htf_view)` | bin edges + volumes (for LVN filter) |

**Key design choice (v1):** `trend_window` passed to `detect_bos_choch` equals
`htf_window_bars`, so the local-trend classifier that labels each break as BOS
or CHoCH looks back exactly as far as the structural window — no longer, no shorter.

### 3. LTF Bar Loop

For each LTF bar `i` (starting after `_WARMUP = 40` bars):

#### 3a. Manage open trade
If a trade is active, call `_find_exit()` — a numpy vectorised scan that finds
the first SL or TP hit without a Python loop.  Jump `i` forward to `exit_bar + 1`.
With `intraday_only=True`, the search is capped at the last bar of the entry day.

#### 3b. FVG touch detection
Compute `wick = bar_low` (bull) or `bar_high` (bear).  
Scan `htf_fvgs` for zones that:
- Are not filled
- Match trend direction
- Are not older than `fvg_max_age_bars` HTF bars
- Are touched by the wick
- Have not already been used today (`used_fvg_keys` set, reset daily)

Pick the FVG whose midpoint is nearest to the wick.

#### 3c. Entry depth check
`fvg_entry_depth(fvg, wick) >= fvg_entry_depth_pct`  
(0 = wick just touched the near edge; 1 = wick reached the far edge)

#### 3d. Optional filters (evaluated in order, skip bar if any fails)
1. **LVN overlap** (`require_lvn_overlap`): FVG must sit inside a Low Volume Node.
2. **Displacement candle** (`displacement_required`): the FVG-forming candle must
   be a displacement bar (large body, small wicks relative to recent ATR).

#### 3e. LTF confirmation (if `require_ltf_confirmation=True`)
Slice pre-computed LTF signals to the window `[i - ltf_per_htf, i]` (one HTF
candle's worth of LTF bars).  Require a CHoCH followed by a BOS in the trend
direction within that window, after the bar when the wick first touched the FVG.

**Key design choice (v1):** confirmation window = `ltf_per_htf` bars (one HTF
candle), not a fixed 120-bar constant.  The local-trend classifier inside
`detect_bos_choch` also uses `trend_window=ltf_per_htf` for consistency.

#### 3f. SL / TP levels
- **Bull:** SL = last HTF swing low below entry × (1 − `sl_buffer_pct`);
  TP = nearest HTF swing high above entry.
- **Bear:** SL = last HTF swing high above entry × (1 + `sl_buffer_pct`);
  TP = nearest HTF swing low below entry.

#### 3g. Risk filters
- `sl_dist / entry_price > max_sl_pct` → skip (SL too wide)
- `tp_dist / sl_dist < min_rr` → skip (R:R too low)

#### 3h. Open trade
Record entry, mark FVG as used for today, advance `i`.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trend_tf` | `"60m"` | HTF timeframe |
| `entry_tf` | `"15m"` | LTF timeframe |
| `htf_window_bars` | 20 | HTF bars for trend/structure (~5 h at 15 m) |
| `swing_lookback` | 2 | Bars each side for swing detection |
| `bos_count` | 1 | Consecutive BOS required to confirm trend |
| `fvg_min_width_pct` | 0.002 | Minimum FVG width as fraction of price |
| `fvg_entry_depth_pct` | 0.10 | Required wick penetration into FVG (0–1) |
| `fvg_max_age_bars` | 50 | Invalidate FVGs older than this many HTF bars |
| `displacement_required` | False | FVG candle must be a displacement bar |
| `require_ltf_confirmation` | False | Require CHoCH + BOS on LTF before entry |
| `require_lvn_overlap` | False | FVG must overlap a Low Volume Node |
| `sl_buffer_pct` | 0.001 | Extra buffer beyond the swing level |
| `max_sl_pct` | 0.005 | Skip trade if SL > this fraction of price |
| `min_rr` | 2.0 | Minimum risk : reward ratio |
| `allow_short` | True | Enable bear setups |
| `intraday_only` | False | Force-close at end of trading day |

---

## Differences from Previous (Pre-v1) Behaviour

| Aspect | Pre-v1 | v1 |
|--------|--------|----|
| HTF window | Hardcoded 200 bars | `htf_window_bars` param (default 20) |
| HTF `detect_bos_choch` trend_window | Hardcoded 20 (independent of window) | `= htf_window_bars` |
| LTF confirmation window | Fixed 120 LTF bars | `ltf_per_htf` (one HTF candle) |
| LTF `detect_bos_choch` trend_window | Fixed 20 | `= ltf_per_htf` |

---

## Files to Review When Updating This Strategy

- `backtest/engine.py` — main loop, `_find_exit`, `BacktestParams`, `ALGO_VERSION`
- `strategy/smc/market_structure.py` — `find_swings`, `detect_bos_choch`, `determine_trend`
- `strategy/smc/fvg.py` — `detect_fvg`, `fvg_entry_depth`
- `strategy/smc/confirmation.py` — `check_ltf_confirmation`

When changing the strategy logic, bump `ALGO_VERSION` in `engine.py`, create a new
`doc/smc_vN_strategy.md` for the new version, and add a git tag matching the version.
