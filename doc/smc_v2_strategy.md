# SMC Strategy — Version 2 (`smc_v2` … `smc_v2.4`)

**Git tags:** `smc_v2`, `smc_v2.1`, `smc_v2.2`, `smc_v2.3`, `smc_v2.4`, `smc_v2.5`  
**Algo version constant:** `ALGO_VERSION` (auto-derived from the most recent `smc_v*` tag)  
**Engine file:** `backtest/engine.py`  
**Parameter reference:** [`strategy/smc/STRATEGY.md`](../strategy/smc/STRATEGY.md)  

---

## Overview

v2 extends the v1 SMC engine with two major additions:

1. **KD channel trend detector** — a second, independent trend signal based on the
   spread between a fast and slow EMA channel.  Can run alongside `bos_choch`
   (consensus mode) or replace it entirely.
2. **Over-refill guard** — rejects entries where the bar close has already punched
   through the far side of the FVG, indicating a failed retest rather than a bounce.

v2.1 further adds **adaptive KD segmentation** (zero-crossing based, scale-invariant),
an **ATR-normalised flat filter**, `fvg_inspect` diagnostic tooling, and `kd_sl_fallback`.

The core pipeline structure (HTF structural analysis → FVG touch → depth check →
filters → SL/TP → trade management) is unchanged from v1.

---

## Version Timeline

| Tag | Key changes |
|-----|-------------|
| `smc_v1` | Baseline: BOS/CHoCH trend, FVG touch/depth, LTF confirmation, LVN + displacement filters, swing-based SL/TP |
| `smc_v2` | + KD channel trend detector; `htf_trend_methods` / `htf_trend_params` replace direct fields; consensus mode (bos_choch + kd); full HTF history EMA warmup |
| `smc_v2.1` | + Over-refill guard; adaptive KD segmentation; ATR-normalised flat filter (`kd_atr_threshold`); `ALGO_VERSION` from git tag; `fvg_inspect` tool; `audit.py` (replaces `review.py`); versioned trade IDs |
| `smc_v2.2` | + `kd_sl_fallback`; `direction_mismatch` rejection logging; screener dollar-volume filter; backtest configs moved to `config/backtest/` |
| `smc_v2.3` | **BOS scan fix** — scan starts at swing bar itself, stops before next same-kind swing (prevents BOS crossing over intermediate highs). **`determine_trend` veto** — CHoCH alone confirms immediately; any subsequent reverse BOS cancels the trend. Per-stock output subdirs; self-contained HTML reports (Plotly JS inline); UTF-8 config loading fix on Windows. |
| `smc_v2.4` | + `require_ltf_trend_bar` — new optional entry filter (Step 6b): entry bar close must move in trend direction (`close > open` for bull; `close < open` for bear). Looser than `require_ltf_confirmation`; independent and combinable with it. |
| `smc_v2.5` | + Gap-fill filter (Step 5c): rejects FVG touches preceded by an opening gap in the fill direction within a configurable lookback window. Params: `gap_fill_lookback` (bars, default 0 = off), `gap_fill_min_pct` (min gap size, default 0.001). Supports both bull and bear setups. Window is inclusive of the first touch bar. |

---

## Changes from v1

### 1. KD Channel Trend Method

**v1:** Only `bos_choch` was available.  
**v2:** `htf_trend_methods` is a tuple of method names.  Multiple methods must
**all agree** on the same non-None direction before an LTF bar is eligible for
entry.  The available methods are `"bos_choch"` and `"kd"`.

```python
# v1 equivalent (unchanged default)
BacktestParams(htf_trend_methods=("bos_choch",))

# v2: KD only
BacktestParams(htf_trend_methods=("kd",))

# v2: consensus — both must agree
BacktestParams(htf_trend_methods=("bos_choch", "kd"))
```

**KD channel:** Two EMA pairs define a fast channel (EMA of highs / EMA of lows
with span `kd_fast`) and a slow channel (span `kd_slow`).  The **width** is the
midline spread between the two channels.  A positive width → bull; negative → bear.

**Adaptive mode** (`kd_smooth > 0`, default): segments the width series at
zero-crossings of the smoothed spread.  For each segment the engine checks whether
`|avg_width| / avg_ATR ≥ kd_atr_threshold` (scale-invariant directional strength
test).  Segments below the threshold are classified as flat.

**Legacy mode** (`kd_smooth = 0`): averages width over a fixed `kd_window`-bar
lookback.  Simpler but less robust to volatility changes across symbols.

### 2. Over-Refill Guard (v2.1)

**v1:** Once the FVG depth threshold was reached, entry was not blocked by how far
the close moved into or through the zone.  
**v2.1:** After depth is confirmed, the engine checks the **close** of the trigger
bar.  If close has punched through the *far* side of the FVG, the zone has been
over-refilled — price action rejected the level rather than bouncing.  The bar is
skipped and the event is logged as `"over_refill"` in `fvg_inspect`.

- Bull: `close < fvg.bottom` → over-refill
- Bear: `close > fvg.top` → over-refill

### 3. KD SL/TP Fallback (post-v2.1, `kd_sl_fallback`)

When `kd_sl_fallback = True`, the KD slow-channel boundaries act as a fallback
SL/TP anchor when swing-based levels are missing or too wide:

- **Bull SL fallback:** `lo2` (lower band of slow channel) × (1 − `sl_buffer_pct`)
- **Bear SL fallback:** `up2` (upper band) × (1 + `sl_buffer_pct`)
- Accepted only if the resulting SL satisfies `max_sl_pct` and `min_rr`.

### 4. Direction Mismatch Logging (post-v2.1)

**v1/v2:** FVGs whose direction did not match the trend were silently skipped.  
**Post-v2.1:** These events are now logged as `"direction_mismatch"` in the
`fvg_inspect` rejection log, making them visible in the HTML diagnostic report.

### 5. Adaptive HTF Computation

**v1:** KD was not used; HTF window was sliced once per new HTF bar.  
**v2:** When KD is enabled, the full HTF series up to the current bar is passed to
`compute_kd()` for EMA warmup — avoiding edge artefacts from a short window.  The
sliced `htf_view` (for swings and FVG detection) and the full-history KD are
computed from the same HTF series, deduplicating work via a shared `_htf_full`
intermediate.

### 6. Tool Chain

| Tool | v1 | v2 / v2.1 |
|------|----|-----------|
| Grid search | `run.py` | Same + `--kd`, `--combined`, `--grid`, `--mu` flags |
| Trade audit | `review.py` (renamed to `audit.py` in v2.1) | `audit.py` — now also supports `--from-csv` |
| FVG diagnostics | None | `fvg_inspect.py` — shows per-event rejection reasons including `direction_mismatch` and `over_refill` |
| Symbol screener | None | `screener.py` — FVG quality + KD clarity + ATR + daily dollar volume scoring |
| Trade IDs | Based on entry time + price only | Prefixed with `ALGO_VERSION` to prevent cross-version DB collisions |

---

## Pipeline (v2.x, changes from v1 highlighted)

### Step 1 — Initialisation

Same as v1.  If `kd_sl_fallback = True`, the engine also pre-arranges to compute
KD for each HTF update.

### Step 2 — HTF Structure Update

Same slice-and-recompute as v1.  **New in v2:** when `"kd"` is in
`htf_trend_methods` or `kd_sl_fallback = True`, additionally computes:

```
_htf_full = htf.iloc[:htf_pos+1]   # full history for EMA warmup
kd_df     = compute_kd(_htf_full, fast, slow, atr_period)
```

`kd_df` columns used:
- `width`     — fast−slow midline spread (trend signal)
- `lo2`, `up2` — slow channel lower/upper bands (SL/TP fallback)
- `atr`       — ATR for scale-invariant flat filter

### Step 3 — Trend Determination

**v1:** `determine_trend(bos_signals, bos_count)` only.  
**v2:** For each method in `htf_trend_methods`:
- `"bos_choch"` → same as v1
- `"kd"` → `kd_trend(kd_df, kd_smooth, kd_window, kd_min_bars, kd_atr_threshold, kd_flat_threshold)`

`trend` is set only when **all** methods return the same non-None value.

### Step 4 — Direction Mismatch Logging *(new in post-v2.1)*

Before scanning for in-zone touches, the engine iterates over all active FVGs whose
direction **differs** from `trend`.  Each such FVG whose wick would have entered
the zone is logged as `"direction_mismatch"` in the rejection log.

This step has no effect on trade logic — it is purely diagnostic.

### Step 5 — FVG Touch Detection

Same as v1: wick vs. zone edge, direction == trend, age ≤ `fvg_max_age_bars`,
not already used today, nearest-midpoint selection.

### Step 6 — Entry Depth Check

Same as v1: `depth = (fvg.top − wick) / (fvg.top − fvg.bottom) ≥ fvg_entry_depth_pct`.

### Step 6a — Over-Refill Guard *(new in v2.1)*

Reject entry if the bar close has punched through the far side of the FVG:
- Bull: `close < fvg.bottom` → skip, log `"over_refill"`
- Bear: `close > fvg.top` → skip, log `"over_refill"`

### Step 7 — Optional Filters

Same as v1: LVN overlap (`require_lvn_overlap`) and displacement candle
(`displacement_required`).

### Step 8 — LTF Confirmation

Same as v1 (`require_ltf_confirmation`).

### Step 9 — SL / TP Levels

Same primary logic as v1 (swing-based).  **New in post-v2.1:**

If `kd_sl_fallback = True`:
- When no swing SL is found → try `lo2` / `up2` from the KD slow channel.
- When swing SL exists but exceeds `max_sl_pct` → override with KD boundary.
- Accepted only if the KD-derived SL still satisfies `max_sl_pct` and `min_rr`.

### Step 10 — Trade Management

Same as v1.

---

## New Parameters (v2 vs v1)

### Trend method control

| Parameter | Default | Description |
|-----------|---------|-------------|
| `htf_trend_methods` | `("bos_choch",)` | Ordered tuple of trend methods; all must agree |
| `htf_trend_params` | `{}` | Per-method config passed by prefix (see below) |

### KD method params (via `htf_trend_params`)

| Key | Default | Description |
|-----|---------|-------------|
| `kd_fast` | 25 | Fast EMA span |
| `kd_slow` | 90 | Slow EMA span |
| `kd_smooth` | 3 | Pre-smoothing window for zero-crossing detection; `0` = legacy fixed-window mode |
| `kd_min_bars` | 3 | Adaptive mode: merge segments shorter than this into the preceding one |
| `kd_atr_threshold` | 0.036 | Flat-segment filter: `\|avg_width\| / avg_ATR < threshold` → flat (scale-invariant) |
| `kd_window` | 10 | Legacy mode only: bars to average width over |
| `kd_flat_threshold` | 0.0 | Legacy mode only: price-unit flat filter (kept at 0; use `kd_atr_threshold` instead) |
| `kd_atr_period` | 14 | ATR rolling period |

### Risk management (new)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kd_sl_fallback` | `False` | Use KD slow-channel `lo2`/`up2` as fallback SL/TP anchor |

---

## v1 vs v2 Quick-Reference

| Aspect | v1 | v2 | v2.1 | v2.2 | v2.3 | v2.4 | v2.5 |
|--------|----|----|------|------|------|------|------|
| Trend method | `bos_choch` only | + `kd`; multi-method consensus | Same | Same | Same + veto rule | Same | Same |
| `determine_trend` | CHoCH → trend | Same | Same | Requires BOS after CHoCH | CHoCH alone confirms; reverse BOS vetoes | Same | Same |
| BOS scan | Start after swing bar | Same | Same | Same | Starts at swing bar (inclusive); stops at next same-kind swing | Same | Same |
| KD mode | — | Legacy (fixed window) | + Adaptive (zero-crossing + ATR filter) | Same | Same | Same | Same |
| EMA warmup | — | Full HTF history | Same | Same | Same | Same | Same |
| Over-refill guard | None | None | Added | Same | Same | Same | Same |
| Trade IDs | Time + price hash | Same | + `ALGO_VERSION` prefix | Same | Same | Same | Same |
| SL/TP fallback | Swing only | Swing only | + `kd_sl_fallback` | Same | Same | Same | Same |
| Direction mismatch | Silent | Silent | Logged in `fvg_inspect` | Same | Same | Same | Same |
| LTF trend-bar filter | None | None | None | None | None | Added (Step 6b) | Same |
| Gap-fill filter | None | None | None | None | None | None | Added (Step 5c) |
| Output layout | — | — | — | Flat run dir | Per-stock subdirs; self-contained HTML | Same | Same |
| Tooling | `run.py`, `review.py` | + flags | + `fvg_inspect.py`, `audit.py`, `screener.py` | + `screener` dollar vol | Same | Same | Same |

---

## Files to Review When Updating This Strategy

- `backtest/engine.py` — main loop, `BacktestParams`, `ALGO_VERSION`, KD integration
- `strategy/smc/kd_trend.py` — `compute_kd()`, `kd_trend()`, adaptive/legacy modes
- `strategy/smc/market_structure.py` — `find_swings`, `detect_bos_choch`, `determine_trend`
- `strategy/smc/fvg.py` — `detect_fvg`, `fvg_entry_depth`
- `strategy/smc/confirmation.py` — `check_ltf_confirmation`
- `backtest/fvg_inspect.py` — rejection log rendering, `_OUTCOME_META`

When changing strategy logic, bump `ALGO_VERSION` by creating a new annotated git
tag (`git tag smc_vN`), create a new `doc/smc_vN_strategy.md`, and update
`strategy/smc/STRATEGY.md` to reflect the current parameter set.
