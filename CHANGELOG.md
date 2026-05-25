# Changelog

## v0.2.0 (2026-05-25)

### SMC Detection Fixes

- **CHoCH detection bug fix** (`market_structure.py`): `trend_started_at` now resets to the
  reference-swing index (`from_idx`) instead of the break bar. Swings formed between the
  reference swing and the break bar remain valid reference points in the new trend context.
  This fixes missed CHoCH signals such as SNDK 15m 2026-05-22 16:00.
- **Session-aware BOS/CHoCH filtering**: added `_session_key()` helper that classifies each
  bar into `pre / regular / post` within the same calendar date. `max_session_gap=0` now
  correctly prevents cross-session breaks even when pre-market and regular bars share the
  same date string.
- **`max_session_gap` stored in `config/chart.json`**: per-TF defaults (`0` for intraday
  TFs ≤ 30m, `null` for 1h+) so the parameter is not forgotten between sessions.
- **Natural-time calibration** for `_BOS_MAX_SPAN` and `_TREND_WINDOW` in `trade_viewer.py`:
  mapped to trading-equivalent bar counts (1m/3m/5m → 1 h, 15m → 1 trading day, etc.)
  instead of a fixed bar count.

### Trade Viewer — Profile Range Selector

- **New `Range` radio buttons** in the toolbar (`1D / 3D / 1W`): select the date range
  over which the POC and VA are computed, independently of the viewport scroll position.
- Profile anchors to the **rightmost visible bar's date** and updates automatically on scroll.
- The session filter (Regular / Pre / Post / Night) stacks on top of the range filter.
- **Fetch window extended to 8 calendar days** so `3D` and `1W` ranges always have
  sufficient data. (For 1m charts the API `max_count=2000` cap still applies; the anchor
  date's candles are always included.)

---

## v0.1.0 (baseline)

### Trade Viewer

- Hybrid volume profile (tick data + OHLCV estimate) with POC and Value Area overlay.
- Session filter checkboxes (Regular / Pre / Post / Night) with zoom-responsive rebuild.
- Neutral-order checkbox; adaptive tight y-axis on profile and tick panels.
- BOS / CHoCH overlay with `max_span_bars` and `trend_window` tuning.
- FVG and Order Block overlays.
- KD channel trend overlay.
- Trade Review mode: load a backtest or live trade by UUID; renders HTF context + LTF entry.

### Backtest Engine

- Multi-process runner with checkpoint/resume and DuckDB persistence.
- Random parameter search and exhaustive grid search (`--grid`).
- Sharpe / Sortino / trade-count stats; versioned results folders.
- `ALGO_VERSION` derived from git tag; `migrate_algo_version.py` for re-tagging.
- FVG stored in trades DB; `fvg_inspect` audit tool.
- Adaptive KD segmentation, ATR-normalised momentum filter.

### Strategy (`strategy/smc/`)

- `market_structure.py`: swing detection, BOS/CHoCH, CHoCH displacement filter.
- `fvg.py`: Fair Value Gap detection.
- `order_blocks.py`: Order Block detection.
- `kd_trend.py`: KD channel trend detector.
