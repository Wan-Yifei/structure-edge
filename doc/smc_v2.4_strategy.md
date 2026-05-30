# smc_v2.4 — LTF Trend-Bar Confirmation

**Tag:** `smc_v2.4`  
**Date:** 2026-05-30  
**Engine:** `backtest/engine.py`  

---

## Change

Adds `require_ltf_trend_bar` (default `False`) as an optional entry confirmation filter (Step 6b).

When enabled, the LTF bar that triggers the entry must close in the trend direction:
- Bull setup: `close > open` (green candle)
- Bear setup: `close < open` (red candle)

A doji (`close == open`) is treated as counter-trend and the bar is skipped.

## Motivation

The existing `require_ltf_confirmation` (Step 6) requires a CHoCH + BOS sequence on the LTF
within the current HTF bar after the FVG touch.  In practice this fires rarely because the full
structure sequence seldom completes within one HTF bar.

`require_ltf_trend_bar` provides a much looser bar-level momentum check: it verifies only that
the entry candle itself is moving with the trend, without requiring any structural pattern.
The intent is to filter out entries where price is actively retracing against the trend at
the moment of execution.

## Implementation

- `BacktestParams.require_ltf_trend_bar: bool = False` — new field (Step 6b in engine loop)
- `label()` — emits `mb` tag when active; `ltf+mb` when combined with `require_ltf_confirmation`
- Rejection log reason: `ltf_trend_bar`

## Interaction with other filters

| Combination | Effect |
|-------------|--------|
| `require_ltf_trend_bar=False` (default) | No change from prior behaviour |
| `require_ltf_trend_bar=True` alone | Filters entries where the entry bar is counter-trend |
| `require_ltf_confirmation=True` alone | Unchanged: requires CHoCH+BOS on LTF |
| Both `True` | Both filters must pass; CHoCH+BOS requirement usually dominates |

## Backward compatibility

`require_ltf_trend_bar` defaults to `False`.  All existing backtests and parameter sets
are unaffected unless the new field is explicitly enabled.
