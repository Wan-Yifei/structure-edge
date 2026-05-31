# smc_v2.5 — Gap-Fill Filter

**Tag:** `smc_v2.5`  
**Date:** 2026-05-31  
**Engine:** `backtest/engine.py`  

---

## Change

Adds a gap-fill filter (Step 5c) that rejects FVG touches where an opening gap
in the fill direction occurred within the configurable lookback window.

Two new `BacktestParams` fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `gap_fill_lookback` | `0` | Bars to scan (window ends at and includes the first touch bar). `0` = filter disabled. |
| `gap_fill_min_pct` | `0.001` | Minimum gap as a fraction of previous close (0.001 = 0.1%). |

`label()` emits `gf{N}` when `gap_fill_lookback > 0`.  
Rejection log reason: `gap_fill_filter`.

## Motivation

A "fill-direction gap" signals that the market opened aggressively toward the
FVG zone rather than approaching it gradually.  This typically occurs at session
open after an overnight gap.  When this happens, price often blasts through the
zone rather than reversing — the FVG level loses its significance as a
reversal point.

Real examples from `US.NVDA` (15m/3m, `bos_choch` trend, 2025):

| Trade | Date | Gap | Outcome |
|-------|------|-----|---------|
| `e186dbcc` | 2025-06-16 09:33 | Bear FVG; open gapped up +0.97% at session open | Loss (-1 R) |
| `8227aabd` | 2025-09-18 09:39 | Bear FVG; open gapped up +2.16% at session open | Loss (-1 R) |

Both trades were bear FVGs touched immediately after a large upward opening gap,
indicating bullish momentum rather than a weak retest.

## Logic

```
win_start = max(1, in_fvg_since - gap_fill_lookback + 1)
win_end   = in_fvg_since + 1   # inclusive of the touch bar

for j in [win_start, win_end):
    gap_up   = open[j] > close[j-1] * (1 + gap_fill_min_pct)
    gap_down = open[j] < close[j-1] * (1 - gap_fill_min_pct)
    if (bear FVG and gap_up) or (bull FVG and gap_down):
        reject → "gap_fill_filter"
```

The window is anchored to `in_fvg_since` (the LTF bar index of the first FVG
touch) and extends `gap_fill_lookback` bars backward, inclusive of the touch
bar itself.  This ensures that a gap at session open — which is typically bar 0
of the session and coincides with the first FVG touch — is always captured.

## Interaction with other filters

| Combination | Effect |
|-------------|--------|
| `gap_fill_lookback=0` (default) | No change from prior behaviour |
| `gap_fill_lookback=3` | Scans opening bar + 2 bars before it |
| `gap_fill_lookback=5` | Scans opening bar + 4 bars before it (recommended for daily open gaps) |
| Combined with `require_ltf_trend_bar` | Independent; both must pass |
| Combined with `displacement_required` | Independent; step 5b runs before 5c |

## Backward compatibility

`gap_fill_lookback` defaults to `0` (disabled).  All existing backtests and
parameter sets are unaffected unless the new field is explicitly set.
