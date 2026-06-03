# Changelog

## v0.10.0 — Liquidity Heatmap enhancements: contrast, price path, depth tooltip, absorption bubbles (2026-06-03)

### New: Gamma contrast spinbox (`analysis/liq_hm_window.py`)

- `Gamma:` spinbox (range 0.2–5.0, default 1.0) added to toolbar next to the `Bid/Ask` toggle.
- Applied after log-normalisation in `_hot_rgba()` and `_single_rgba()` via `norm^gamma`.
- gamma > 1 suppresses sparse zones — only the densest order clusters remain bright.
- gamma < 1 boosts dim zones — reveals weaker order concentration.
- `_on_gamma_changed` only calls `_render()`, skipping the heavier overlay recalculation.

### New: Mid-price path line (`analysis/liq_hm_window.py`)

- `Price` checkbox (default on) draws a white `PlotCurveItem` connecting `(bid+ask)/2`
  for every column (ZValue=8 — above heatmap, below detection markers).
- Per-column mid-price stored in `_mid_prices` list; synced with grid rolling and price
  range resets.  `None` entries render as `np.nan` (line breaks at missing data).
- `_calc_col_mid(snap)` extracted as a module-level pure function for testability.

### New: Depth-to-cursor annotation (`analysis/liq_hm_window.py`)

- Price label now shows the cumulative resting volume between the current spread and
  the cursor position:
  - `eat↑ N` — N shares of ask liquidity to consume to push price to cursor.
  - `eat↓ N` — N shares of bid liquidity to consume to push price to cursor.
  - `[spread]` — cursor is inside the bid-ask spread.
- Uses the most recent OB snapshot cached in `_latest_snap` (updated every tick).
- `_calc_depth_label(snap, best_bid, best_ask, target)` extracted as a pure function.

### New: Absorption bubble overlay (`analysis/liq_hm_window.py`, `analysis/orderflow_detect.py`)

- `Absorb` checkbox + `MinΔ:` spinbox (default 500) added to toolbar.
- `detect_absorption_bubbles(ticks, col_ts, mid_prices, col_secs, min_delta_vol)`:
  for each column, computes `delta = buy_vol − sell_vol` from tick data and compares
  the delta direction against the mid-price movement:
  - **Gold bubble** — aggressive buyers absorbed (delta > 0, price flat/down).
  - **Purple bubble** — aggressive sellers absorbed (delta < 0, price flat/up).
  - Bubble size (8–30 px) encodes the absorbed delta volume.
- `_AbsorbTickWorker(QThread)` loads `ticks.db` in the background; triggered every
  time a new column arrives.  Main thread is never blocked.

### Tests

- `tests/analysis/test_liq_hm.py` (new, 23 tests): gamma correction, mid-price
  calculation, and depth-to-cursor annotation — no Qt dependency.
- `tests/analysis/test_orderflow_detect.py` (13 new tests): `detect_absorption_bubbles`
  covering absorption/non-absorption cases, threshold filtering, None mid-price, multiple columns.
- Total: 92 tests, all passing.

---

## v0.9.1 — Stacked imbalance detection + overlay; scheduler/viewer reliability fixes (2026-06-02)

### New: Stacked imbalance detection (`analysis/orderflow_detect.py`)

- `detect_stacked_imbalance()`: flags N consecutive depth ranks where bid/ask volume ratio ≥
  threshold as a bullish zone, or ask/bid ≥ threshold as a bearish zone.
- Bid/ask levels are paired by depth rank (rank 0 = best bid vs best ask).  Missing side
  within `max_depth` ranks counts as 0 (infinite ratio), not truncated.
- `max_depth` parameter restricts analysis to top-of-book levels (default 10); deep-book
  orders beyond this rank are ignored as noise.
- 53 unit tests total (17 new covering stacked imbalance edge cases).

### New: Stacked imbalance overlay (`analysis/liq_hm_window.py`)

- **Blue vertical bar** = bullish stacked imbalance (bid dominates N consecutive depth levels).
- **Orange vertical bar** = bearish stacked imbalance (ask dominates N consecutive depth levels).
- Toolbar controls: **Imbalance** checkbox + **Lvl** (min consecutive levels, default 3) /
  **Ratio** (bid/ask threshold, default 3.0) / **Depth** (max ranks analysed, default 10) spinboxes.

### Enhancement: 3 m timeframe (`analysis/trade_viewer_qt.py`)

- `3m` added to `TIMEFRAME_MAP` and the TF combo; now available alongside 1m / 5m / 15m / 30m / 1h / 4h.

### Fix: LITE account spread lines (`analysis/trade_viewer_qt.py`)

- `_trigger_fetch` now polls `get_market_snapshot()` for real NBBO bid/ask prices; previously
  relied on OB snapshot data which is unavailable on LITE accounts.

### Fix: Viewer close hang (`analysis/trade_viewer_qt.py`, `analysis/liq_hm_window.py`)

- `closeEvent` now explicitly closes `_liq_hm_window` and `_dom_window` before the main
  window exits, preventing Qt from destroying widgets while their threads are still running.
- `LiqHmWindow.closeEvent` stops the per-tick timer and drains both `_SnapshotWorker` and
  `_BulkSnapshotWorker` QThreads before returning.

### Fix: Scheduler watchdog OB restart (`analysis/scheduler.py`)

- `elif → if` fix: OB collector restart was silently skipped whenever the tick collector
  was alive, because both checks shared the same `elif` chain.
- **Zombie OB detection**: if the OB process is alive but the DB has received no new writes
  for > 15 min, the watchdog force-restarts it.

### New: Scheduler autostart & Windows startup toggle (`analysis/scheduler.py`)

- On launch, if the market session is ACTIVE or STARTING SOON, collectors auto-start after a
  200 ms delay — no manual click needed.
- Windows startup toggle: registers / removes a `pythonw.exe` entry in `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`
  so the scheduler starts with Windows.

### Fix: DOM staleness indicator (`analysis/dom_window.py`)

- `_ts_lbl` turns red with `"STALE (Xm)"` when live data is > 60 s old, making stale feeds
  immediately visible.

---

## v0.9.0 — Heatmap pre-fill, profile Y-sync, OB stability fixes (2026-06-01)

### New: Heatmap historical pre-fill (`analysis/liq_hm_window.py`)

- On startup (or code switch), `LiqHmWindow` fetches the **last 5 distinct OB
  snapshots** from SQLite in a background `_BulkSnapshotWorker` thread before
  starting the normal per-tick timer.  Heatmap is no longer blank at open.
- Historical columns use the actual DB timestamps so the time axis is accurate.
- `_push_column()` accepts an optional `ts` parameter for pre-fill vs live use.
- `_reset_grid()` cancels any in-flight bulk worker and stops the timer; the
  `_needs_init` flag gates bulk-vs-single fetch in `set_live()`.
- `_on_max_cols_changed()` re-triggers `set_live(True)` after the grid reset so
  the new wider/narrower grid is also pre-filled.

### Fix: Spoof marker triangle orientation (`analysis/liq_hm_window.py`)

- Replaced PyQtGraph string symbols (`"t"`, `"t2"`) with explicit `QPainterPath`
  triangles — unambiguous across all PyQtGraph versions.
- **Bid spoof ▲** (apex at y = −0.5 = screen top): false buy pressure lures
  longs → marker points up.
- **Ask spoof ▼** (apex at y = +0.5 = screen bottom): false sell pressure lures
  shorts → marker points down.

### Enhancement: Crosshair labels brighter (`analysis/liq_hm_window.py`)

- Price and time labels changed from muted `#b0bec5` to white (`#ffffff`) with a
  semi-transparent dark fill so they read clearly over the heatmap image.
- Crosshair lines also brightened to white.

### Fix: Session volume profile Y-range drift (`analysis/trade_viewer_qt.py`)

- `_on_main_range_changed` now syncs `_profile_widget` Y range to the main
  chart's visible Y range on every pan/zoom — profile never drifts out of view.
- `_rebuild_session_profile()` also sets the profile Y range immediately from the
  main chart's current range (covers the first render before `_reset_view` fires).
- `poc_line` and `va_line` (`InfiniteLine` decorators) marked `ignoreBounds=True`
  so they no longer skew PyQtGraph's auto-range calculation.

### Fix: Chart view drift after code switch (`analysis/trade_viewer_qt.py`)

- `_reset_view` deferred via `QTimer.singleShot(0, ...)` so it fires after
  PyQtGraph's own deferred `updateAutoRange` calls (triggered by
  `prepareGeometryChange()` in each `set_data()`), guaranteeing our explicit
  range overrides auto-range instead of the reverse.
- `_reset_view` moved after `_rebuild_session_profile()` in `_render`.

### Fix: Live code switch / Stop→Connect reliability (`analysis/trade_viewer_qt.py`)

- `_live_code` tracks the actually subscribed code; `_stop_live` unsubscribes
  the correct code instead of the new (already updated) code field value.
- `_connect_opend` and the disconnect branch of `_on_connect_toggle` both discard
  any stuck `DataFetcher` (disconnect its signals, set to `None`) so
  `isRunning()` never blocks a fresh fetch after reconnect.
- `_last_chart_key` cleared on reconnect to force a view reset on next render.
- Code-change branch of `_trigger_fetch` returns after `_start_live()` to avoid
  concurrent `ctx.subscribe()` + `ctx.request_history_kline()` deadlock.

### Fix: OB collector WAL runaway & no-data recovery (`analysis/order_book_collector.py`, `feeds/order_book_store.py`)

- **Write rate-limit**: minimum 2 s gap per code (`_MIN_WRITE_INTERVAL`) prevents
  high-frequency pushes from growing the WAL unboundedly.
- **Prune**: `OrderBookStore.prune(keep=1000)` deletes old rows, keeping the most
  recent 1 000 rows per code; called every watchdog tick.
- **WAL checkpoint**: `PRAGMA wal_checkpoint(PASSIVE)` run every 10 watchdog
  ticks (`_CHECKPOINT_INTERVAL`) to keep the WAL file small.
- **Re-subscribe**: watchdog automatically calls `ctx.subscribe()` with
  `subscribe_push=True` after `timeout_minutes` of no data, then resets the
  `warned` flag so the next timeout triggers another retry.

### Minor: Logging cleanup & read_only support

- `tick_collector.py`: removed verbose total-row-count log at startup and exit.
- `tick_store.py`: `TickStore(read_only=True)` skips DDL to avoid RESERVED lock
  contention with the writer process.
- `config/schedule.json`: added `US.SOXS` to targets.

## v0.8.1 — Iceberg & spoof detection fixes (2026-06-01)

### Fix: Iceberg segment reset on level disappearance (`orderflow_detect.py`)

- A time gap > 1.5 × col_secs between consecutive snapshots at a price level
  now splits the history into independent segments; each segment runs a fresh
  state machine (running_peak / depleted reset).
- Breakthrough → reappearance correctly starts a new iceberg search rather than
  continuing the previous one.  Multiple segments at the same level each produce
  their own `(first_bar, last_bar, price, n_ref)` tuple → separate cyan segments.
- `col_secs` added as a parameter to `detect_icebergs`; passed from
  `_col_secs_spin` in `LiqHmWindow`.

### Fix: Spoof detection — execution proxy replaces missing tick data (`orderflow_detect.py`)

- Removed `raw_ticks` parameter (was always `[]` in the HM context, making the
  execution filter completely ineffective).
- Execution is now inferred from **spread movement**: if the best bid (BID order)
  fell below the price level, or the best ask (ASK order) rose above it, between
  appearance and disappearance, the order was consumed — not a spoof.
- `min_vol=0` auto-computes the threshold as the **median volume** of the most
  recent OB snapshot, adapting to each ticker's liquidity automatically.
- Each reappearance of a large order after a prior spoof/disappearance is treated
  as an independent new event with its own ▲/▼ marker.

## v0.8.0 — Liquidity Heatmap floating window (2026-06-01)

### New: Standalone Liquidity Heatmap window (`analysis/liq_hm_window.py`)

- `LiqHmWindow(QWidget)` — independent floating window showing resting order book
  depth as a price × time heatmap (X = wall-clock time, Y = price).
- Each column = one OB snapshot; columns scroll left as new data arrives.
- **Combined mode** (default): black → purple → amber → yellow hot colormap
  encodes total (bid + ask) resting volume per price level.
- **Bid/Ask mode**: teal = bid depth, red = ask depth, shown separately.
- **Best bid / ask lines**: teal and red dashed horizontal lines mark the current
  top-of-book spread (盘口) and update every tick.
- **Iceberg overlay**: cyan `--` segments mark price levels where resting volume
  repeatedly drops then refreshes — now correctly restricted to within 2 price
  bins of the best bid (bid side) or best ask (ask side).
- **Spoof overlay**: orange ▲/▼ triangles (size 15, white outline) mark large
  orders that appear and vanish without execution.
- **Crosshair**: local mouse hline + vline with price and time labels; vline also
  synced from main chart crosshair via `pin_timestamp()` in historical mode.
- **Legend bar**: dedicated row below toolbar showing colormap stages, best
  bid/ask line colors, and active overlay indicators.
- **⟲ Reset button**: restores zoom to full data range; double-click also resets.
- **Background polling**: `_SnapshotWorker(QThread)` fetches latest snapshot
  asynchronously — never blocks the UI thread.
- Immediate first tick on `set_live(True)` — heatmap populates within seconds.
- Standalone usage: `uv run analysis/liq_hm_window.py`

### Integration with `analysis/trade_viewer_qt.py`

- Replaced embedded heatmap ImageItems + iceberg/spoof overlays on the main chart
  with a single **Liquidity Heatmap** toggle button that opens/closes `LiqHmWindow`.
- Removed OB data loading from `DataFetcher.run()` — the 55 M-row DB query was
  blocking the fetcher thread and preventing chart rendering and stock switching.
- Crosshair sync: in historical mode, moving the cursor pins the heatmap vline
  to the matching column via `liq_hm_window.pin_timestamp(bar_ts)`.

### Fix: Iceberg detection proximity filter (`analysis/orderflow_detect.py`)

- **Bug**: previously detected icebergs at any depth in the book, including levels
  far from the spread where no executions occur — producing many false positives.
- **Fix 1**: group snapshots by `(side, price_bin)` instead of `price_bin` alone;
  mixing BID and ASK volumes at the same level was generating phantom drop/recover
  signals.
- **Fix 2**: only flag levels within `max_spread_bins=2` bins of the best bid
  (BID side) or best ask (ASK side) — passive orders deep in the book are never
  consumed and cannot produce genuine iceberg signals.

### Other

- `config/schedule.json`: added `order_book_enabled` flag and `remote_backup`
  config block (S3/Wasabi weekly snapshot).
- `scripts/check_db.py`: quick DB diagnostic (row count, codes, latest ts per code).

## v0.7.0 — DOM (Depth of Market) window (2026-05-31)

### New: Depth of Market window (`analysis/dom_window.py`)

- `DomWindow(QWidget)` — independent floating window showing resting order book
  bid/ask depth as vertical bar chart (price on X, volume on Y).
- Teal bars = bids (resting buy orders); red bars = asks (resting sell orders).
- Dashed vertical lines mark the best bid and best ask prices.
- **Depth selector**: 10 / 20 / 30 / 50 levels per side (combo box).
- **Live mode**: refreshes every 1 s from `db/order_book.db` (latest snapshot).
- **Historical mode**: crosshair-synced — moving the cursor in the main chart
  updates the DOM window to the order book snapshot closest to that bar's time.
- **Hover tooltip**: hover over any bar to see:
  - Side (BID / ASK) and price
  - Volume at that level
  - Cumulative volume from the best price to this level ("how much to eat through")
  - Number of levels from best to hovered price
- Status bar: live best bid/ask, total depth, and spread.
- Standalone usage: `uv run analysis/dom_window.py --code US.SNDK`

### Integration with `analysis/trade_viewer_qt.py`

- New **DOM** toggle button in the indicators toolbar (tb2, after Spoof controls).
- Clicking DOM opens/closes `DomWindow` as a separate floating window.
- Closing the DOM window unchecks the toolbar button automatically.
- Code, live/historical mode, and candle timeframe stay in sync with the main viewer
  on each data load (`set_code`, `set_live`, `set_timeframe`).

### New: Absorption detection (`analysis/orderflow_detect.py` + DOM window)

Detects price levels where large passive orders held against significant aggressive
flow within a single bar window.  Three conditions must all pass:

1. **Passive wall** ≥ `avg_tick_vol × Pass` (resting order large enough to matter)
2. **Aggressive volume** ≥ `avg_tick_vol × Act` (real pressure applied — split orders
   accumulate naturally, so fragmented aggression is captured equally)
3. **Hit ratio** = `agg_vol / pass_vol` ≥ `Hit%` (meaningful fraction was attempted)
4. Resting volume at window end > 0 (level still present = not broken through)

Window is bar-aligned: `[candle_start(now, tf), now]` in live mode;
`[time_key − tf, time_key]` in historical mode (moomoo time_key = bar end).

`avg_tick_vol` = session average volume per trade — adaptive to instrument
liquidity without manual calibration.

DOM toolbar controls: `[✓ Absorb]  Pass:[3.0]×  Act:[1.0]×  Hit:[30]%`

Display: gold outline = ASK absorption (sell wall held); blue outline = BID
absorption (buy wall held).  Hover tooltip shows aggressive vol, passive vol,
and hit ratio for flagged levels.

`tests/analysis/test_orderflow_detect.py`: 17 new tests for `detect_absorption`
covering edge cases, each threshold condition, split-order accumulation,
direction filtering, and simultaneous bid/ask detection (36 total, all pass).

---

## v0.6.0 — Scheduler: order book collector + remote backup (2026-05-31)

### New: Order Book Collector integration

- `analysis/scheduler.py`: new **Collectors** panel with an Order Book Collector
  toggle (default enabled). When the scheduler is running, `order_book_collector.py`
  starts and stops alongside `tick_collector.py` at session boundaries.
- Setting persisted to `config/schedule.json` as `order_book_enabled`.

### New: Remote Backup panel

- Cron-based upload of `ticks.db` and `order_book.db` to S3/Wasabi.
- UI fields: enable toggle, cron expression (default `0 20 * * 1-5`), S3 path,
  AWS profile, endpoint URL (blank = standard AWS S3).
- **Backup Now** button for immediate manual trigger.
- Uses `aws s3 sync` — skips files unchanged since last upload.
- Streams `aws s3 cp/sync` output to the log panel in real time; shows file size
  before upload starts.
- Profile `"default"` or blank omits `--profile` flag, allowing env-var auth.
- 600 s per-file timeout with background kill timer.
- Settings persisted to `config/schedule.json` under `remote_backup` key.

---

## v0.5.0 — Gap-fill filter (smc_v2.5) (2026-05-31)

### smc_v2.5 — Gap-fill filter (Step 5c)

- New `gap_fill_lookback` parameter (default 0 = off): scans the N LTF bars ending
  at (and including) the first FVG touch for an opening gap in the fill direction.
  Bear FVG: rejects if `open > prev_close × (1 + gap_fill_min_pct)`.
  Bull FVG: rejects if `open < prev_close × (1 - gap_fill_min_pct)`.
- New `gap_fill_min_pct` parameter (default 0.001): minimum gap size threshold.
- `label()` emits `gf{N}` tag when active.
- Rejection log reason: `gap_fill_filter` (added to `fvg_inspect._OUTCOME_META`).
- Validated against two NVDA losses (`e186dbcc` 2025-06-16, `8227aabd` 2025-09-18)
  — both bear FVGs opened with upward session gaps (+0.97% / +2.16%) and were losses.
- `fvg_inspect.py`: added `--gap-fill-lookback` and `--gap-fill-min-pct` CLI args.
- `config/backtest/cross_stock_grid_v3.json` + `soxl_grid_v2.json`: added
  `gap_fill_lookback: [0, 3, 5]` and `gap_fill_min_pct: [0.001]` to `param_grid`.

### Docs

- `doc/smc_v2.5_strategy.md`: new change note.
- `doc/smc_v2_strategy.md`: version timeline and quick-reference table updated.
- `strategy/smc/STRATEGY.md`: Step 5c and parameter reference added.
- `doc/FILE_INDEX.md`: new — index of all important files and update checklist.

---

## v0.4.1 — Docker HPC pipeline fixes (2026-05-31)

- `docker/entrypoint.sh`: results-dir detection now filters on `YYYYMMDD_*` pattern,
  preventing `backtest/results/checkpoints/` from being mistaken as the run output dir
  (caused uploads to go to `s3://.../results/checkpoints/` and DB named `backtest_checkpoints.duckdb`)
- `docker/entrypoint.sh`: always create a run-tagged local copy
  `db/backtest_<run_tag>.duckdb` on the bind-mounted volume after each run
- `backtest/engine.py`: `_algo_version()` checks `ALGO_VERSION` env var before
  `git describe`; fixes `smc_unknown` stamping in Docker where `.git/` is absent
- `docker/backtest.env.example`: document `ALGO_VERSION` setting

---

## smc_v2.4 — LTF trend-bar confirmation (2026-05-30)

- New `require_ltf_trend_bar` parameter (Step 6b): entry bar close must move in
  trend direction (`close > open` bull; `close < open` bear).  Looser alternative
  to `require_ltf_confirmation`; independent and combinable with it.
- `label()` emits `mb` tag when active, `ltf+mb` when combined with CHoCH+BOS filter.
- Rejection log reason: `ltf_trend_bar`.

---

## v0.4.0 — Order Flow Overlays (2026-05-30)

### New: Order book data pipeline
- `feeds/order_book_store.py` — SQLite WAL storage for resting order book snapshots
- `analysis/order_book_collector.py` — push-based `OrderBookHandlerBase` subscriber;
  writes every depth change to `order_book.db`; watchdog thread mirrors tick_collector design

### New: Liquidity Heatmap (`analysis/trade_viewer_qt.py`)
- Resting limit order concentration displayed as background image (z=-10, behind candles)
- Combined mode: black → purple → amber → yellow colormap
- Bid/Ask split mode: teal (bids) / red (asks) separately
- Toolbar controls: `Liq.HM: Show`, `Bid/Ask`, `Min.Vol:` spinbox

### New: Iceberg order detection
- `analysis/orderflow_detect.py::detect_icebergs()` — detects hidden large orders by
  scanning for repeated volume drop→recover cycles at the same price level
- Cyan horizontal line segments on the heatmap; brightness encodes refresh count
- Toolbar: `Iceberg` checkbox + `Min.Ref:` spinbox (default 3)

### New: Spoofing detection
- `analysis/orderflow_detect.py::detect_spoofs()` — detects large orders that appear
  then vanish without execution; cross-references `ticks.db` to confirm non-fill
- BID spoof → orange ▲ (pushing price up); ASK spoof → orange ▼ (pushing down)
- Dotted orange duration line from order appearance to cancellation bar
- Toolbar: `Spoof` checkbox + `Max.Dur:` spinbox (default 30 s)

### Refactor + tests
- Detection logic extracted to `analysis/orderflow_detect.py` (no Qt dependency)
- `tests/analysis/test_orderflow_detect.py` — 19 unit tests covering both detectors

---

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
