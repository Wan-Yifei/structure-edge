# Changelog

## v0.13.7 — live mode fixes + aggressor bubble performance + WAL stability (2026-06-15)

### Fix: Range profile missing lower half when tick coverage is partial (`analysis/trade_viewer_qt.py`)

- `_compute_profile_bins` used all-or-nothing tick logic: if any bar had tick data
  all bars without tick coverage were silently dropped, biasing the profile toward
  the most recently loaded dates.
- Fixed: per-bar hybrid approach — tick data used where available, OHLCV proportional
  fill applied for bars with no tick coverage, so the full price range always contributes.

### Fix: Volume profile disappeared after get_cur_kline supplement (`analysis/trade_viewer_qt.py`)

- `get_cur_kline` returns fewer columns than `request_history_kline` (missing `change_rate`).
- `pd.concat` produced NaN columns, breaking downstream profile processing.
- Fixed: `cur_new.reindex(columns=df.columns)` aligns columns before concat.

### Fix: Aggressor bubbles lag 5-6 minutes during active markets (`analysis/liq_hm_window.py`)

- Worker re-queried the entire 2-hour tick window on every heatmap update, causing
  slow DB scans during high-volume sessions (e.g. 6M+ ticks in ticks.db).
- Fixed: incremental loading — full query on first load, then only fetch ticks after
  `_absorb_last_ts` on subsequent calls; cache trimmed to visible window.
- Added `_absorb_reload_pending` flag: if `_on_tick` is blocked by a running worker,
  re-triggers immediately after the worker finishes so bubbles stay current.
- `_query_ticks` now uses `TickStore` (WAL-aware) instead of raw `sqlite3.connect`
  to avoid silent failures under write pressure.

### Fix: WAL file grows unboundedly and degrades read performance (`analysis/tick_collector.py`)

- tick_collector had no periodic WAL checkpoint, unlike ob_collector.
- Fixed: watchdog runs `PRAGMA wal_checkpoint(PASSIVE)` every 10 ticks (~10 min).

### Fix: OB timestamps could be wrong if system timezone is not ET (`analysis/order_book_collector.py`, `analysis/liq_hm_window.py`)

- `datetime.now()` depended on system timezone; changed to explicit
  `datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)` so OB
  timestamps are always in US Eastern time regardless of where the code runs.

### Improvement: Tick profile delta moved to panel title (`analysis/trade_viewer_qt.py`)

- Delta total `Δ ±NNK` label was placed as a floating `TextItem` at the top of the
  buy bars, causing it to be obscured by profile bars during busy sessions.
- Moved into the panel's top label alongside the header (e.g. `09:35  Δ +12K`);
  floating TextItem removed.

## v0.13.6 — tick collector auto-reconnect + imbalance color improvements (2026-06-11)

### Fix: Tick collector silently drops subscription (`analysis/tick_collector.py`)

- Added two-tier watchdog auto-reconnect:
  1. Re-subscribe on existing ctx (handles silent subscription drop).
  2. If that fails (e.g. `WinError 10054` / `_init_connect_sync` timeout),
     close the dead ctx and rebuild a fresh `OpenQuoteContext`, re-set handler,
     and re-subscribe. `ctx` is held in a `ctx_holder` list so the watchdog
     thread can replace it without breaking main()'s shutdown path.
- Watchdog check interval reduced to 60 s (was `timeout_sec`), so reconnect
  fires faster after detection.
- Fixed watchdog loop stall: after re-subscribe, `last_resub_time` used as
  elapsed clock instead of `last_tick_time = None` silently skipping checks.

### Fix: Imbalance bars hard to see against heatmap background (`analysis/liq_hm_window.py`)

- Bullish bar: Material Blue 300 → **Lime A200** `(178,255,89)`.
- Bearish bar: Deep-Orange 400 → **Pink A200** `(255,64,129)`.
- Line width 4 → 5; opacity 200 → 220.
- Added missing Imbalance entry to the legend bar.

## v0.13.5 — fix aggressor bubbles disappearing after first tick update (2026-06-11)

### Fix: Aggressor bubbles vanish after first live data update (`analysis/liq_hm_window.py`)

- `_on_absorb_ready` was unconditionally overwriting `_absorb_ticks` with the
  worker's result, including empty lists when the tick collector has no data in
  the current window.  Bubbles disappeared after the first reload cycle.
- Fixed: only replace `_absorb_ticks` when the worker returns a non-empty result;
  stale/empty reloads keep the last good cache so existing bubbles stay visible.
- `_reset_grid()` now also disconnects and clears `_absorb_worker` (mirrors the
  existing `_bulk_worker` pattern) and explicitly clears `_absorb_ticks`, so
  switching symbols can't leave a stale worker calling back with the old code's ticks.

## v0.13.4 — fix bid/ask label anchor order (2026-06-11)

### Fix: A/B price label vertical order was swapped (`analysis/liq_hm_window.py`)

- `_bid_label` anchor was `(0, 1.0)` (bottom) → text floated upward into the spread.
- `_ask_label` anchor was `(0, 0.0)` (top) → text floated downward into the spread.
- With a tight spread the text heights exceeded the gap, making "B" appear above "A".
- Fixed: bid label anchor → `(0, 0.0)` (floats below bid line);
  ask label anchor → `(0, 1.0)` (floats above ask line).
- A is now always above B regardless of spread width; dashed line colors unaffected.

## v0.13.3 — LiqHm aggressor overlay + bid/ask line fixes (2026-06-11)

### Refactor: Absorb → Aggressor detection (`analysis/liq_hm_window.py`, `orderflow_detect.py`)

- Overlay renamed from "Absorb" to "Aggressor"; checkbox tooltip updated.
- New `detect_aggressor_bubbles()` — same bucketing as absorption but no
  price-movement filter; direction interpretation left to user.
- Gold = net BUY aggression; purple = net SELL aggression.
- MaxΔP spinbox removed; MinΔ threshold retained.
- `_on_bulk_ready` now calls `_load_absorb_ticks()` so the overlay populates
  after prefill (previously never triggered after bulk load).

### Fix: Bid/ask spread lines now visible (`analysis/liq_hm_window.py`)

- Lines were invisible (teal/red blended into Bid/Ask heatmap colormap).
- Changed to colored dashed lines + `TextItem` price labels ("B …" / "A …")
  pinned to the left edge of the view via `_set_quote_line()`.
- Default colors match heatmap Bid/Ask colormap: ask=RED, bid=TEAL.
- Colors automatically follow the main viewer's "Red Up" toggle via
  `set_red_up()`: red-up=True → ask=RED bid=TEAL; red-up=False → ask=TEAL bid=RED.

### Feat: Default symbol from config (`analysis/trade_viewer_qt.py`)

- `_default_code()` reads `default_code` from `config/schedule.json` instead
  of hardcoding "US.SNDK".

### Fix: Tick collector 24H coverage (`analysis/tick_collector.py`)

- Added `session=Session.ALL` alongside `extended_time=True` to the TICKER
  subscription — required for overnight US ticks (OpenD ≥ 9.2.4207).
  Without it the collector silently collected zero ticks outside regular hours.

## v0.13.2 — LiqHm absorb + price path fixes (2026-06-08)

### Fix: Absorption bubble color mapping inverted (`analysis/liq_hm_window.py`)

- BUY-absorbed events now render purple (bearish); SELL-absorbed render gold
  (bullish) — was previously swapped.
- Tooltip and legend text updated to match corrected semantics.

### Fix: Price path (Lo→Hi) lagging behind best bid/ask dashed lines

- `update_quote()` now sets `_live_mid = (bid+ask)/2` and immediately appends
  it as a trailing point at `x = n` via `_update_price_path()`.
- The price path now updates at quote-tick frequency, in sync with the bid/ask
  spread lines, instead of waiting for the next column push (every `col_secs` s).

### Fix: `ModuleNotFoundError: No module named 'PyQt5'` in absorb hover tooltip

- `_on_absorb_hovered()` was importing from PyQt5; changed to PyQt6.

### Improve: MaxΔP filter for absorption detection

- `detect_absorption_bubbles()`: new `max_price_move` parameter — caps the
  allowed mid-price move per column for an event to qualify as true absorption.
  Filters out false positives (e.g. spike-down with dip buyers).
- LiqHmWindow: new MaxΔP spinbox (0–9.99, default 0.10; 0 = ∞).

## v0.13.1 — LiqHm UX + DuckDB fixes (2026-06-08)

### Fix: DuckDB 1.5.2 internal assertion in signal scanner (`feeds/kline_store.py`)

- `KlineStore.date_range()`: replaced `SELECT MIN/MAX … fetchone()` with two
  `ORDER BY … LIMIT 1` queries — avoids the DuckDB 1.5.2 internal assertion that
  fires on aggregate functions applied to empty filtered result sets.
- `KlineStore.has_data()`: replaced `SELECT COUNT(*) … fetchone()` with
  `SELECT 1 … LIMIT 1` for the same reason.
- Root cause: DuckDB 1.5.2 throws `INTERNAL Error: Attempted to access index 0
  within vector of size 0` inside `.execute()` itself (not `.fetchone()`) when
  `MIN`/`MAX`/`COUNT` are applied to a WHERE-filtered empty result set.

### Improve: Liquidity Heatmap UX (`analysis/liq_hm_window.py`)

- **Two-row toolbar**: core display controls (Bid/Ask, Gamma, Price, Min.Vol,
  Col(s), History, Reset) on row 1; detection overlays (Iceberg, Spoof,
  Imbalance, Absorb) on row 2 — no more horizontal scrolling to reach controls.
- **Gamma range** extended from 5.0 to 10.0 for stronger contrast suppression.
- **Col(s) minimum** reduced from 5 s to 1 s; step changed to 1 s for
  fine-grained refresh-rate control.
- Default window size adjusted to 1000 × 460 px.

### Fix: Scanner error traceback logging (`analysis/signal_scanner.py`)

- Per-symbol scan errors now log the full Python traceback (not just the
  exception message) to aid debugging.

## v0.13.0 — SMC Signal Scanner (2026-06-07)

### New: Signal Scanner (`analysis/signal_scanner.py`, `uv run main.py scanner`)

- Standalone PyQt6 app that monitors a configurable watchlist and detects
  SMC entry setups using the same strategy logic as the backtest engine.
- `SignalDetector`: pure (no Qt) class — directly unit-testable; detects BOS/CHoCH
  trend, unfilled FVGs, swing-based SL/TP, RR filter.
- `ScanWorker(QThread)`: per-symbol bar cache (skips unchanged bars), deduplication
  against open signals in DB, `QApplication.beep()` alert on new signal.
- `ParamsDialog`: Auto mode (queries BacktestDB for highest-PF run) or Manual mode
  (key BacktestParams fields as a form); "Preview match" button shows live DB result.
- `SignalScanner` main window: watchlist table, recent-signals table (last 50),
  log area, status bar, toolbar (Connect / Scan / Interval / Sound / Add / Remove /
  Edit Params).

### New: System tray notifications + click-to-open viewer (`analysis/signal_scanner.py`)

- `QSystemTrayIcon` with teal icon; each new signal shows an 8-second OS-level
  balloon notification with symbol, direction, entry zone, and RR.
- Clicking the balloon (or double-clicking a row in the signals table) launches a
  new `trade_viewer_qt` process in Historical mode for that symbol and trend TF,
  positioned at the signal date.

### New: Signals persistence layer (`db/signals.py`)

- `SignalsDB`: SQLite WAL-mode database at `db/signals.db`.
- Schema: signal_id, symbol, direction, signal_time, trend/entry TF, entry zone,
  SL, TP, RR, BOS price, strategy, params_json, algo_version, source, status,
  closed_at, created_at.
- `insert_signal`, `update_status`, `query_signals`, `get_open_signals`,
  `get_all_open_signals`; context-manager support.

### New: BacktestDB.get_best_params() (`backtest/db.py`)

- Queries DuckDB for the highest-PF completed run for a symbol within a
  configurable lookback window (default 3 months, min 5 trades, min PF 1.5).
- Used by `ScanWorker` auto-params mode and `ParamsDialog` preview.

### New: Scanner Signals overlay (`analysis/trade_viewer_qt.py`)

- "Scanner Signals" toggle button in Row 3 toolbar.
- Reads open signals from `db/signals.db` for the current symbol; overlays
  entry zone band (teal/red, α=35), SL dashed line, TP dashed line, and
  direction + RR label on the main chart.

### Fix: Viewer — Enter key in code field (`analysis/trade_viewer_qt.py`)

- `installEventFilter` on `_code_edit` intercepts `Key_Return / Key_Enter`
  before `QToolBar` can consume them on Windows (where `returnPressed` alone
  is unreliable inside a toolbar).
- Added log messages for two previously silent-return paths in `_trigger_fetch`:
  "Fetch in progress" and "Live: switched to X".

### Fix: Viewer — subplot titles show current symbol (`analysis/trade_viewer_qt.py`)

- Vol and KD subplots now display `"{symbol}  Vol"` / `"{symbol}  KD"` as titles,
  updated on every render, so multiple open viewer windows are easy to distinguish.

### New: Entry point (`main.py`)

- `uv run main.py scanner` launches the signal scanner.

### Tests

- `tests/db/test_signals_db.py`: 9 unit tests for SignalsDB CRUD.
- `tests/analysis/test_signal_detector.py`: 13 unit tests for SignalDetector
  (guards, allow_short filter, min_rr filter, signal dict structure, mock-based).

---

## v0.12.0 — Viewer: Range Volume Profile + daily TF + per-TF historical lookback (2026-06-07)

### New: Range Volume Profile (`analysis/trade_viewer_qt.py`)

- Drag-selectable region on the candle chart renders a tick/OHLCV volume profile
  in the right panel with POC / VAH / VAL lines and an inline overlay on the chart.
- 31 unit tests for `_compute_profile_bins` and `_compute_poc_vah_val`.

### New: Daily timeframe (`analysis/trade_viewer_qt.py`, `core/time_utils.py`)

- Added `"1d"` (K_DAY) to `TIMEFRAME_MAP` — now available in the TF dropdown.
- Historical lookback: 2000 calendar days; Live lookback: 730 days.
- K_DAY `time_key` normalised to `"YYYY-MM-DD 00:00:00"` on fetch so all
  downstream `[:16]` slices and `strptime("%Y-%m-%d %H:%M")` calls work uniformly.
- X-axis labels show full `"YYYY-MM-DD"` for daily (vs `"MM-DD HH:MM"` for intraday).
- `candle_start` extended to handle 240-min (4h) and 1440-min (1d) candles correctly.

### New: Per-TF historical lookback (`analysis/trade_viewer_qt.py`)

- Replaced fixed 8-day lookback with `_HIST_LOOKBACK_DAYS` dict:
  `1m→3d, 3m→5d, 5m→10d, 15m→20d, 30m→30d, 1h→90d, 4h→500d, 1d→2000d`.
- 4h now fetches ~570 bars; 1h ~440 bars instead of the previous ~13 / ~56 bars.

---

## v0.11.0 — Absorption bubble fixes + hover tooltip + scheduler zombie watchdog (2026-06-05)

### Fix: Absorption bubble tick bucketing (`analysis/orderflow_detect.py`)

- Replaced half-window bisect logic with direct `bisect_right - 1` bucketing.
  Each column owns `[col_ts[i], col_ts[i+1])`; no half-window clipping needed.
  Fixes tick drops on column boundaries that caused bubbles to disappear.

### Fix: MinΔ spinbox response latency (`analysis/liq_hm_window.py`)

- MinΔ change now redraws immediately from cached ticks instead of re-querying `ticks.db`.
  DB reload only occurs when ticks are not yet cached or the display window shifts.
- MinΔ range lowered to 10–100 000 (was 100), step 10 (was 100).

### New: Absorption bubble hover tooltip (`analysis/liq_hm_window.py`)

- Hovering a bubble shows a `QToolTip` with direction and absorbed Δvol:
  `Buyers absorbed by sellers (bearish) / Δvol: N`
- Uses `pg.ScatterPlotItem(spots=…, hoverable=True)` + `sigHovered`.

### New: Absorption bubble legend entry (`analysis/liq_hm_window.py`)

- Legend bar now shows gold/purple colour swatches with passive-voice labels
  when the `Absorb` checkbox is checked; hidden when unchecked.

### New: Tick collector zombie watchdog (`analysis/scheduler.py`)

- `_tick_db_stale_minutes()`: checks WAL file mtime (O(1)) to detect DB write stalls.
- If the collector process is alive but no DB write for > 30 min, the scheduler
  terminates and restarts it automatically.

### Config

- `config/schedule.json`: added `US.AVGO` to monitored targets.

---

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
