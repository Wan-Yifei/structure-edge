# Changelog

## Trade Viewer Qt (2026-05-29)

### New viewer: `analysis/trade_viewer_qt.py`

Full rewrite of the chart tool using **PyQt6 + PyQtGraph**, replacing the
Matplotlib/Tkinter-based `trade_viewer.py`.  Key improvements: native GPU
rendering, smooth zoom/pan, and a richer panel layout.

**Launch:**
```bash
uv run main.py trade_viewer_qt                              # Live mode
uv run main.py trade_viewer_qt --code US.NVDA --tf 15m
uv run main.py trade_viewer_qt --mode Historical --date 2026-05-15
```

**Panels:**
- Candlestick with per-bin tick heatmap colouring (gold = buy pressure,
  purple = sell pressure) and Delta Δ annotations per candle.
- EMA overlays (20 / 50 / 200), each independently toggleable.
- BOS / CHoCH structure markers + FVG zone overlays + Order Block overlays
  (regular / mitigation / breaker subtypes).
- MAVOL subplot (volume bars + 20-period volume MA).
- KD channel spread subplot (bull / bear / flat colour-coded).
- Session Vol Profile panel (right side): POC + Value Area; 1D / 3D / 1W
  date-range selector anchored to the rightmost visible bar.
- Single-candle Tick Profile (hover to reveal, S/M/L size filter).
- Full-panel crosshair sync + non-clipping OHLCV tooltip.
- Session filter checkboxes: Pre / Regular / Post / Night.
- Trade Review mode: enter a Trade ID → jump to entry bar, show HTF
  FVG + BOS context, entry/exit/SL/TP markers.
- Colour scheme toggle: 🔴涨🟢跌 (CN) / 🟢涨🔴跌 (US).

**`main.py` dispatch fix:** `main()` and `_parse_args()` now accept an
optional `argv` parameter so the unified `main.py` entry point can forward
leftover CLI args correctly (`uv run main.py trade_viewer_qt --code US.NVDA`
no longer raises `TypeError`).

The old `trade_viewer.py` (Matplotlib) is kept as legacy and remains
accessible via `uv run main.py trade_viewer`.

---

## smc_v2.3 (2026-05-29) — patched 2026-05-29

### Bug fixes (post-release, tag moved to HEAD)

- **`detect_fvg` default `require_displacement=True` → `False`**: The parameter
  was accidentally left as `True`, silently filtering all FVGs unless a
  displacement candle was present — independent of `params.displacement_required`.
  Backtest engine now passes `require_displacement=params.displacement_required`
  explicitly.  Previously all v2.3 backtests produced 0 trades due to this bug.

- **`compare_versions._run_one` used `result.to_summary_dict()`**: Correct
  method is `result.summary_dict()`.  Caused every worker to fail silently with
  AttributeError and report 0 for all metrics.

- **`trade_viewer_qt` overlays (FVG / OB / BOS) invisible**: Three separate bugs:
  (1) BOS/OB signals used warmup-relative indices but the chart x-axis uses
  full-df indices — items were drawn off-screen to the far left.
  (2) `detect_fvg` called with `require_displacement=True`, hiding most FVGs.
  (3) BOS rendering capped at the last 8 signals — increased to show all warmup
  signals.  Also increased `_BOS_MAX_SPAN` / `_TREND_WINDOW` to viewer-appropriate
  values so structure spanning the full visible window is detected.

---

## smc_v2.3 (2026-05-29)

### Market Structure — `determine_trend` veto rule

**Old behaviour** (smc_v2.2): A CHoCH set `consecutive = 0`, requiring at least
one same-direction BOS in the rolling window before the trend was confirmed.
This was too strict and suppressed valid entries when no confirming BOS had yet
appeared.

**New behaviour**: CHoCH immediately confirms the trend (`consecutive = 1`), but
any BOS in the *opposite* direction that appears **after** the last CHoCH vetoes
the trend → `determine_trend` returns `None`.  This correctly rejects the
`[CHoCH bear, BOS bull]` pattern (market reclaimed the structural level) while
allowing clean CHoCH-only setups.

### Market Structure — BOS scan boundary fix (`detect_bos_choch`)

The inner break-scan now starts at the current swing bar itself (`sw["idx"]`,
inclusive) and stops before the next swing of the same kind.  Previously the scan
started at `sw["idx"] + 1`, which caused the scan to skip the swing bar even when
its close already exceeded the prior-swing wick — resulting in a BOS line drawn to
a lower, later bar that visually crossed over the obvious structural high.

### Backtest — per-stock output subdirectories

Each stock's log, CSV, PNG and HTML report are now written to a dedicated
subdirectory `<run_dir>/<CODE_slug>/` (e.g. `results/…/US_NVDA/`) instead of all
files landing flat in the run root.

### Backtest — self-contained HTML reports

`report.py` now embeds the full Plotly JS bundle inline (`get_plotlyjs()`) instead
of a CDN `<script>` tag.  Reports open correctly in air-gapped / restricted
networks and do not require an internet connection.  File size increases by ~3 MB.

### Backtest — UTF-8 config loading on Windows

`_load_json_config` in `run.py` now opens JSON files with `encoding="utf-8"`,
fixing a `UnicodeDecodeError` on Windows (GBK default locale) when config files
contain non-ASCII characters.

### Backtest — infinite `profit_factor` guard

`_cap_inf_pf()` in `report.py` replaces `float("inf")` in the
`profit_factor` column with the largest finite value before passing the DataFrame
to stats functions, preventing `RuntimeWarning: invalid value encountered in double_scalars`.

---

## smc_v2.2 (2026-05-25)

### Backtest — config refactor

- Backtest configs moved to `config/backtest/` with version-suffix filenames.
- Per-stock parameter grids extracted from `run.py` into config JSON
  (`param_grid` field); `--mu` flag removed.

### Strategy — `kd_sl_fallback`, direction-mismatch logging, screener dollar volume

- `kd_sl_fallback`: when `True`, the KD slow-channel `lo2`/`up2` boundary is used
  as a fallback SL/TP anchor when swing-based levels are missing or exceed `max_sl_pct`.
- FVG events whose direction differs from the current trend are now logged as
  `"direction_mismatch"` in `fvg_inspect` rejection output.
- `screener.py`: dollar volume filter added to surface only liquid symbols.

---

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
