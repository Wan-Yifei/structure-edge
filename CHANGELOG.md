# Changelog

## v0.27.1 — Walls labels follow the Gamma control (2026-10-06)

### Fix: labels ranked on raw size, independent of what rendered (`analysis/liq_hm_window.py`)

v0.27.0 picked walls by size alone, so the labels and the picture could
disagree: a level the current Gamma had faded to black still got a label, and
raising Gamma thinned the bands without thinning the labels.

A candidate now also has to render at `_WALL_MIN_BRIGHT` (0.5, the amber band
on the colour ramp's 0.25/0.5/0.75 scale), computed through the same
`log1p -> percentile -> gamma` pipeline as the image by calling the renderer's
own `_percentile_norm`, over whichever grid the active mode draws -- combined,
or one side. Measured on live book data, SOXL at 1800s:

| Gamma | combined | bid/ask |
|---|---|---|
| 0.5 - 2.0 | 10 labels | 10 |
| 3.0 | 4 | 6 |
| 5.0 | 1 | 6 |
| 8.0 | 1 | 3 |

Turning Gamma up now culls the labels along with the bands, leaving the one
wall that is actually there.

The normalisation reference spans the whole painted grid rather than the zoomed
slice, so panning does not relabel; ranking still uses the visible columns.

## v0.27.0 — label the thick walls with price and size (2026-10-06)

### Feat: a Walls overlay on the heatmap (`analysis/liq_hm_window.py`)

The heatmap shows where size sits but not what price that is -- reading a
bright band off a 100-bin axis is guesswork. A new **Walls** checkbox labels
the thickest resting levels outright: `B 164.00  21,852`, teal for bids, red
for asks, N per side (default 3).

Ranked on the largest size a price bin reached anywhere in the **visible
window**, not in the newest column. The bands that read as walls are the ones
that persisted, and a level that flashed for one snapshot should not outrank a
standing one. Adjacent bins are merged so a wall spanning two bins is labelled
once, at its heavier half.

The price is the bin's centre, since the grid knows nothing finer. That is
sub-penny while the band is narrow, but a day-long replay spreads ~$25 over 100
bins and a bare `164.00` would then mean anywhere in a quarter-dollar -- so
once half a bin exceeds a cent the label says so: `164.00±0.13`.

Verified against live book data: every label's price, side and size match the
grid peak at that bin, and the labelled levels are the actual top-N.

## v0.26.1 — collapse long expiry lists (2026-10-04)

### Fix: five daily expiries overflowed the stats column and filled the title (`strategy/option/gex.py`)

QQQ and SPY both list five expiries inside a week. Spelled out that is 68
characters, which ran out of the quarter-width "Key Strikes" column and printed
over the ITM Call stats next to it, and filled the title bar edge to edge with
no room for a sixth.

`_expiry_label` lists a few and collapses a longer run to its range plus a
count -- `2026-10-05 → 2026-10-09  (5)`. The panel collapses from three, the
title from four, so SOXL's three-expiry chart still spells them out where it
fits. Not wrapped to a second line: the hedging verdict row sits directly
below and would have been the next collision.

## v0.26.0 — GEX chart says whether the walls matter (2026-10-04)

### Feat: dealer hedging intensity against actual volume (`strategy/option/gex.py`)

GEX is an absolute number: it says how much gamma dealers carry, not whether
that is large against how much the stock trades. The same GEX is decisive on a
thin name and invisible on a liquid one, and nothing in the chart distinguished
the two.

A new full-width line converts GEX to **shares dealers must trade per 1% move**
(GEX is gamma x OI x 100 x spot, the delta change for a move of one whole spot,
so a percent of it is the per-1% figure) and divides by 20-day average volume,
with a verdict: negligible < 0.5% < minor < 2% < material < 5% < dominant.

On SOXL today the strongest strike needs 9.4K shares per 1% move, **0.02% of
average volume** -- negligible. All strikes combined reach 0.18%. Against 61M
shares a day the walls cannot pin anything, which is worth knowing before
trading off them. The same measure on a thinner name is where it earns its
place.

Also checked a second way, repricing gamma with Black-Scholes at spot = strike
to capture the rise as price approaches a wall: $165 goes from 15.3K to 16.9K
shares, still 0.03%.

The verdict line spans the full width under the stats panel. A first attempt
put it in a quarter-width column, where it printed over the next column's
stats -- the same class of collision v0.25.1 fixed.

## v0.25.1 — GEX chart labels stop overlapping (2026-10-04)

### Fix: three collisions in the option GEX chart (`strategy/option/gex.py`)

- **Spot against the Call Wall.** Both labels were pinned to the same y, so a
  spot of 163.71 under a 165 call wall printed the price through the wall's
  box. Top-of-chart labels now take the highest row still clear of every label
  already placed, "clear" being more than 7% of the plotted strike range away.
- **"Strike Price" clipped.** The axes bottom sat at 0.30 of figure height with
  the separator at 0.285, leaving the tick labels and the axis title nowhere to
  go. The figure is taller and the axes start higher.
- **The footnote printed over the stats.** It lived inside the stats panel on
  the same band as the "% 20d Avg Vol" row. It now has its own strip on the
  figure below the panel.

## v0.25.0 — the live heatmap fills its whole window on open (2026-09-30)

### Feat: pre-fill the rolling buffer from the database (`analysis/liq_hm_window.py`)

Leaving a replay dropped back to a near-blank panel that took `Max cols`
seconds to rebuild -- four minutes at the defaults -- losing whatever happened
in between. The pre-fill asked for the last **5 snapshots**, under a second of
chart at the feed's ~6/s.

`_WindowPrefillWorker` now fetches one snapshot per Col(s) bucket for the whole
visible window, reusing the `bucketed_snapshots` query the replay is built on.
Returning from any replay range repopulates all 240 columns -- a full four
minutes of chart -- in **0.3-0.4s**, and a cold open fills the same way instead
of starting empty. 240 columns measured 25ms at the database, 1440 columns
138ms, so the cost is in the thread hand-off rather than the query.

The band is now derived once from the newest snapshot rather than re-derived
per column. `_maybe_init_price_range` wipes the grid when it decides to
rebuild, so calling it 240 times mid-fill would have thrown away most of what
had just been painted -- harmless at 5 snapshots, not at 240.

`_BulkSnapshotWorker` and `_query_n_snapshots` are removed; nothing else used
them.

## v0.24.2 — load as many days of ticks as CVD accumulates over (2026-09-29)

### Fix: CVD was flat at zero before the current trading day (`analysis/trade_viewer_qt.py`)

v0.24.0 gave CVD a 2D/3D/1W window but left the tick loader at one day.
`load_local_ticks` spans the prior 20:00 through 23:59, so every bar older than
the current trading day had no bucket, contributed a delta of 0, and the curve
sat flat at zero until 20:00. Reported after switching to 2D/3D.

`load_tick_days` now loads as many days as the Range spans. Two things had to
be right, and the first attempt at each was wrong:

- **Caching.** Past days never change, and re-reading five of them measured
  13.3s against 0.2-2.8s for one -- far too slow on a fetch that runs on a 1s
  timer. Days are cached by (code, tf, date); the newest is always re-read
  since it is still growing, and empty days are not cached at all rather than
  spending a slot of a bounded cache on nothing. Warm, a 5-day load is 0.2s.
- **Counting trading days, not calendar days.** Stepping back three calendar
  days over a weekend reaches Sunday, leaving the oldest session empty. Then
  counting *non-empty* calendar days was still wrong: Sunday's file holds the
  20:00 buckets that open Monday's trading day, so it retired a slot without
  adding a session. The quota is now the number of distinct trading days the
  loaded buckets cover.

Changing Range also refetches rather than just redrawing -- the extra days'
ticks have to be on hand before the curve means anything.

Verified against live data: at 1D/2D/3D/1W every trading day in the block now
carries a real curve, where 3D previously left its oldest session at zero.

## v0.24.1 — the rest of the ways a replay could be wiped (2026-09-28)

### Fix: a symbol switch during a replay stranded the window (`analysis/liq_hm_window.py`)

`set_code` cleared the grid without clearing `_hist_secs`, so the window still
believed it was replaying. Every later `set_live()` then stood down -- the
guard added in v0.24.0 -- and the feed never came back. Worse than the bug that
guard was added for, and only reachable by changing symbol while replaying. It
now leaves replay mode and puts the combo back to Live.

`_on_push` stands down during a replay too. `_stop_push()` should mean nothing
arrives, but a push that slipped through would rebuild the grid against the
replay's much wider band, which is exactly the original failure.

### The check now enumerates the ranges instead of listing them

The v0.24.0 verification covered 30m/2h/6h/1d and silently skipped **All** --
a hand-written list that did not match `HIST_RANGES`. It now derives the cases
from `HIST_RANGES`, so a range added later is covered without anyone
remembering to. All five ranges pass four scenarios: the main viewer's
per-second `set_code`/`set_live`, a symbol switch, returning to Live, and
toolbar changes reloading rather than resetting.

## v0.24.0 — CVD anchors on the Range control (2026-09-28)

### Fix: a replay was wiped a second after loading (`analysis/liq_hm_window.py`)

Reported: the 1D replay flashed up and reverted to live, while 30m was fine.
The main viewer calls `set_live(True)` on the heatmap every refresh cycle --
once a second at the current floor -- which re-armed the feed underneath the
replay. The first live push carries a near-touch band a few dollars wide;
`_maybe_init_price_range` compared it against the replay's band (a day is
~$25), fired the `too_wide` rebuild, and zeroed the grid. 30m survived only
because its band is $1.75, close enough to live not to trip the check.

`set_live` and `_on_bulk_ready` now both return early while a replay is up. The
second one matters for short ranges: a pre-fill worker still in flight would
restart the timer after `_load_history` had stopped it.

### Feat: CVD accumulates over the Range window (`analysis/trade_viewer_qt.py`)

CVD reset at every trading-day boundary. It now resets per *block* of trading
days, where the block size comes from the same Range control that scopes the
session volume profile -- 1D (default, unchanged behaviour), 2D, 3D or 1W.

Blocks are counted back from the newest trading day, so the most recent block
always holds a full N days and its boundary does not move when the lookback
changes. That also puts the reset on the same bar the profile's window starts,
which is the point: a level read off the profile and the CVD under it now
describe the same stretch of tape.

`RANGE_DAYS` is now one map both read, and the trading-day rule the two share
is one function (`_trading_day_of_bucket`) instead of the expression CVD had
repeated.

## v0.23.1 — aggressor bubbles come back in replay (2026-09-28)

### Feat: the replay rebuilds aggressor bubbles (`analysis/liq_hm_window.py`)

v0.23.0 turned every order-flow overlay off while replaying. That was one
blanket rule for two different things. Iceberg, spoof and imbalance read
`_raw_snaps` and look for a level refreshing between consecutive books --
consecutive in a replay are a bucket apart, so the signal genuinely is not
there. Aggressor bubbles are built from ticks, which the bucketing never
touched, and `detect_aggressor_bubbles` assigns each tick to a column by
bisecting `col_ts`, so uneven bucket-wide columns are fine.

The replay now loads ticks for its whole range when Aggressor is checked --
644k ticks for a day, 1.2s -- and draws the bubbles across it: 1,949 of them
over a 6h replay, verified against the scatter item's own point count. With
the checkbox off no ticks are fetched at all, so the cost is opt-in. MinΔ is
per column, so a wide bucket needs a higher threshold; the status line says so.

### Fix: an empty `_raw_snaps` silently dropped the bubbles too

`_redraw_orderflow_markers` bailed out on `not self._raw_snaps` before reaching
the aggressor block at the bottom, so the replay detected bubbles and drew none
of them. The guard now gates only the three book-based detections. Caught by
counting drawn markers rather than detected bubbles -- the detection was right
all along.

## v0.23.0 — heatmap replays history from the database (2026-09-28)

### Feat: a Replay control redraws the heatmap from stored snapshots (`analysis/liq_hm_window.py`, `feeds/order_book_store.py`)

The heatmap kept a rolling 240-column buffer -- four minutes at Col(s)=1 --
and everything older was gone. Measuring first showed the destructive
price-window rebuild was not the cause: over 15.4 minutes of real data it fired
**zero** times. It was simply the buffer rolling.

Panning already worked; there was just nothing to pan to. Rather than hold the
history in RAM (a day at 1s is 138MB of grid plus ~600MB of raw levels), the
new **Replay** control re-draws it from the collector's database, which has
been storing every snapshot all along: 30m / 2h / 6h / 1d / All, about 6MB and
under 1.3s each.

`OrderBookStore.bucketed_snapshots()` returns one snapshot per time bucket --
the newest in each, matching what the live path paints. The bucket is coarsened
so a range lands under 4000 columns, which is also about where columns stop
being a pixel wide. Measured on the full 535k-snapshot table: 0.46s at a 60s
bucket, 0.61s at 5s, but **73s and 192MB at 1s**, so long ranges must coarsen
rather than be capped afterwards. Selecting by a materialised id list collapses
the same way: `rowid IN (?,?,…)` with 11k parameters measured 51s against 42ms
at 3.8k, hence the nested subquery.

Live updates stop while replaying, and order-flow overlays are off: iceberg and
spoof look for a level refreshing between consecutive books, and consecutive
here are a bucket apart, so the signal is not there to find.

### Fix: junk levels were setting the price scale (`feeds/order_book_merge.py`)

The feed emits padding levels at price 0.0, and the stored history holds one at
66568.05. 63 such levels among 3.1M, and they never mattered -- until the replay
took a min()/max() over a day of book and stretched a $12 session into a $158
band, crushing the chart into a sliver. The replay band now comes from the mid
path at the 0.5/99.5 percentiles, and `parse_side` drops non-positive prices so
no more of them are stored.

## v0.22.1 — profile level labels moved to the right edge (2026-09-28)

### Fix: Last / POC / VAH / VAL sat on top of the histogram (`analysis/trade_viewer_qt.py`)

All three profile panels draw horizontal bars growing rightward from x=0, so
the left edge is the one x every bar passes through -- and that is where the
level labels were pinned, over the bars they describe. Reported as the blue
`Last` tag occluding the profile.

They now sit right-aligned on the right edge, which the X range already keeps
clear: it is pinned to 15% past the longest bar. Covers the session profile
(POC trail, VAH/VAL, Last), the single-candle tick VP panel (POC/VAH/VAL) and
the range profile panel (POC/VAH/VAL/Last).

The range panel's Tot/Buy/Sell/Neu readout stays top-left: it is a panel
header rather than a price level, and leaving it there keeps it from colliding
with the four that moved. The main chart's inline range labels were already on
the right and are unchanged.

`tests/analysis/test_profile_label_side.py` drives the real widgets offscreen
and reads the placed items back, since the claim is geometric -- an anchor
change without the matching x coordinate still renders, just wrongly.

## v0.22.0 — pair estimate anchors on the live quote (2026-09-25)

### Feat: anchor the inverse-pair estimate on the live pair, not the previous close (`analysis/trade_viewer_qt.py`)

The error in the estimate is drift accumulated since whatever moment both legs
were last known together, so a fresh anchor is worth a lot. The anchor is now
the live pair from the market snapshot each fetch cycle already requests --
at Refresh=1s that is a one-second-old anchor.

Measured on 1m and 5m bars, regular session, 2026-08-01..09-25, the two
timeframes agreeing throughout:

| anchor | median | p90 |
|---|---|---|
| previous close | $0.030 | $0.138 |
| live, 1m old | **$0.011** | $0.035 |
| live, 15m old | $0.021 | $0.081 |
| live, 60m old | $0.029 | $0.139 |

Against a fresh anchor the error scales with how far the hovered level sits
from the live price, which is what matters when reading a level off the chart:
within 0.1% it is $0.009 median / $0.022 p90, rising to $0.027 / $0.080 at
0.5-1% away. Roughly a 3x improvement on the levels actually being read.

`_pair_refs` drops BOTH legs to previous close when either has no live print,
rather than pairing one leg's live price with the other's close -- two
different moments make the ratio meaningless. That happens pre-market, when
one leg has traded and the other has not.

### Note on validating this against history

moomoo's daily and 5m series are split-adjusted while 1m is raw. Mixing a
daily prev_close into 1m bars manufactures a constant ~$0.58 offset that reads
as model error; an earlier pass of this measurement hit exactly that.

## v0.21.1 — fix: the pair tag took the whole price readout down (2026-09-25)

### Fix: `_pair_implied` read a `self._code` the window does not have (`analysis/trade_viewer_qt.py`)

v0.21.0 shipped with the crosshair raising `AttributeError` on every mouse
move: the viewer keeps its symbol in the code box, not in a `_code` attribute,
so the lookup threw and aborted the handler before it could show anything --
taking the ordinary gold price tag with it. Reported as no price showing at all.

The tests did not catch it because they built a bare object and assigned
`_code` themselves, inventing the very state the window lacks. They now
construct a real `TradeViewerQt` and set the symbol through the code box, and
a new `TestCrosshairLabels` drives the actual mouse-move handler -- including
the case that broke, an ordinary symbol still showing its price. Only
`_pair_anchor` is set directly, since it comes from a live market snapshot.

The anchor also carries its base code now. `_trigger_fetch` returns early on a
live code change, so the previous symbol's anchor outlives the switch until
the next load; matching the stored code against the box keeps it from being
read against the wrong symbol in that window.

## v0.21.0 — inverse-pair price on the crosshair (2026-09-25)

### Feat: hovering a SOXL level also shows where SOXS would be (`analysis/trade_viewer_qt.py`)

The crosshair's price tag now splits in two on SOXL: the gold tag lifts just
above the line and a purple one hangs just below it with the SOXS price
implied by that same level, so a level picked on one leg can be read straight
off as a level on the other. Both tags carry their ticker, which they do not
in single-symbol mode -- with one number there is nothing to disambiguate.

Both funds are +/-3x the same index and reset daily from the previous close,
so from that shared anchor a SOXL return of 3r implies a SOXS return of -3r:

    SOXS = SOXS_prev_close * (2 - SOXL / SOXL_prev_close)

Checked against live quotes (SOXL 151.845 / prev 146.33, SOXS 32.350 / prev
33.63): predicted 32.362 against 32.350 actual, 0.04% off. The leverage
cancels, so a SOXL move of +4% reads as SOXS -4%. Tracking error and fund
drift make this an estimate, not an arbitrage relation.

The pair's prev_close rides along in the market snapshot the viewer already
requests on each load, so the feature costs no extra round trip. Nothing is
drawn when there is no anchor, on symbols outside `_INVERSE_PAIR`, or in
historical mode, where the anchor would be today's close rather than the
displayed date's -- no tag beats a silently wrong one.

`tests/analysis/test_gui_window_smoke.py` now also constructs the main viewer,
so the crosshair items are covered by the suite rather than only by opening it.

## v0.20.2 — overlay markers re-anchor on every column roll (2026-09-23)

### Fix: quiet book left the bubbles pinned while the grid scrolled (`analysis/liq_hm_window.py`)

Reported in the after-hours session: the price path kept advancing but the
aggressor bubbles stayed where they were drawn.

`_on_snap_ready` (the DB-poll path) rolls a column on **every** tick -- when
the book has not changed it forward-fills, so the time axis keeps moving -- but
it only called `_redraw_orderflow_markers()` / `_load_absorb_ticks()` when the
snapshot carried new data. Overlays are positioned by column index, so an
unchanged book still moves every marker's correct position one column left.
After-hours is where this shows: the book can sit still for many ticks at a
stretch, and a stale push feed (>90s) falls back to this same path.

The push path had the identical bug and was fixed earlier by redrawing in
`_on_tick`; the comment there claimed the poll path already did this, which was
only true for the ticks that happened to carry new data.

`tests/analysis/test_liq_hm_marker_refresh.py` drives the real widget offscreen
and pins one redraw per column roll whatever the mix of quiet and active ticks.

## v0.20.1 — order-book retention sized in hours (2026-09-21)

### Fix: retention was still sized for the throttled write rate (`analysis/order_book_collector.py`)

`_RETENTION_KEEP_OVERRIDE = {"US.SOXL": 60_000}` was chosen when the collector
wrote one snapshot per 2s, where it meant 33 hours of history. Removing the
throttle raised the rate ~12x and the same constant quietly became **2.7
hours** -- the heatmap's pre-fill and any order-book backtracking could only
reach that far.

The counts are now derived from an hours target (`_keep_for_hours`) rather
than written down, so a future rate change is a one-line edit with a visible
consequence instead of a silent one. SOXL keeps **24 hours** (535,680
snapshots, ~1.1 GB); other codes keep one regular session, 6.5 hours
(145,080, ~0.3 GB each). Sizing uses the regular-session rate, the densest
one, which makes the hours target a floor across every session.

## v0.20.0 — one row per order-book snapshot (2026-09-21)

### Feat: the collector stores every push instead of one every 2s (`feeds/order_book_store.py`, `analysis/order_book_collector.py`)

The old schema stored one row per price level and the collector rate-limited
writes to one snapshot per code per 2s, because 140 levels x every push was
judged too much to write. A push-rate probe run against the live feed settled
what that was costing: ORDER_BOOK arrives on a fixed ~300ms cadence (p90 =
0.302s in all three samples) with activity-dependent bursting, and **zero
duplicate books** -- 1.0 push/tick overnight, 2.25 at the open. Nothing was
being deduplicated away; the throttle was discarding distinct books.

`ob_snapshots` now holds one row per snapshot, with the two sides as JSON, and
the throttle is gone. **12x more snapshots stored** -- 0.5/s before, 6.2/s
measured after, median inter-snapshot gap 0.291s.

Identity is `rowid`, deliberately not `(code, ts)`. An intermediate version
used that as a primary key with `INSERT OR REPLACE` and lost books: the feed
bursts several distinct books 12-19ms apart while `datetime.now()` on Windows
only advances in ~15.6ms steps, so 30 pushes stored 9. Reads order by
`ts DESC, rowid DESC` and `prune` counts rows, not timestamps.

### Fix: a locked book is not a crossed book (`feeds/order_book_merge.py`)

The cross guard tested `best_bid >= best_ask`, throwing away every snapshot
where the two sides quoted the same price. That is a routine state -- two
venues on the same price before the trade prints -- and it clusters around the
open, which is when the heatmap can least afford a hole. Only `bid > ask` is
physically impossible.

It surfaced now because removing the write throttle let all ~6 pushes/s reach
the merger, filling the console with
`crossed book bid=141.44 >= ask=141.44 (both/neither side fresh), skipping`.
Warnings for genuine crosses are now throttled to one per code per 5s with a
suppressed count on the next one through; throttling the log never affects
whether the snapshot is skipped.

### Fix: definitions lost to an index-based refactor (`analysis/liq_hm_window.py`)

The refactor that routed every read through `OrderBookStore` replaced a block
between two text anchors by index, and the end anchor sat further down the file
than intended. It also took `_TICK_DB_PATH`, `_query_ticks` and four
`QThread`/`QObject` classes -- among them `_PushBridge`, which `__init__`
instantiates. Opening the heatmap raised `NameError` while the whole suite
still passed, because nothing in it ever built the window.

`tests/analysis/test_gui_window_smoke.py` now constructs `LiqHmWindow` and
`DomWindow` offscreen. Confirmed to fail if `_PushBridge` goes missing again.

### Fix: two scripts were reading a frozen table (`scripts/check_db.py`, `scripts/analyze_absorb_price_change.py`)

Both still queried `order_book_snapshots` directly after it stopped being
written, so they reported a 325-snapshot pruning remnant as if it were live
data. Adds `earliest_ts()` and `codes()` to the store, which is what they were
hand-rolling as `MIN(ts)` and `SELECT DISTINCT code`.

### Migration

`order_book_snapshots` is no longer read or written. `OrderBookStore.
drop_legacy_table()` removes it; a `VACUUM` afterwards reclaims the freed
pages (96% of a 308MB file in practice). Neither runs automatically.

## v0.19.0 — incremental live kline fetch, Refresh down to 1s (2026-09-21)

### Feat: live cycles top up a cached frame instead of re-pulling the window (`analysis/trade_viewer_qt.py`)

Every live refresh re-fetched the whole lookback window: four paginated
`request_history_kline` calls (1m looks back 5 days, ~7200 bars at 2000/page).
That dominated the cycle and is why `Refresh(s)` had a 5s floor -- at 1s it
would have been ~240 history calls a minute, well past the API's limits.

A live cycle now hands last cycle's frame back to the fetcher, which tops it
up with a single `get_cur_kline` and merges. **One API call per cycle instead
of four**, and no re-parse of bars that have not changed. **`Refresh(s)`'s
floor drops from 5s to 1s**, validated live at 1s.

Correctness rests on two things:

- `_merge_klines` **overwrites** on an equal `time_key` rather than appending
  only unseen ones. The newest bar is still forming, so the same key returns
  with a different close/high/low/volume; keeping the first copy would freeze
  the live bar. The old `get_cur_kline` supplement had exactly that blind spot
  -- it filtered to unseen keys -- which went unnoticed only because the full
  re-pull replaced the frame every cycle anyway.
- Gaps are **detected, not predicted**. `get_cur_kline` returns the most
  recent `_CUR_KLINE_BARS` (200) bars, so it can only bridge a shorter gap.
  Rather than reason about elapsed time, the fetcher compares its oldest
  returned bar against the cache's newest and falls back to a full fetch when
  bars are missing between them -- correct however long the app was asleep,
  paused or disconnected.

The cache is `_klines` itself, keyed by `(code, tf)`, so a symbol or timeframe
switch cannot top up a frame with the wrong bars. Historical mode never tops
up; its window is pinned to a date rather than rolling. The merged frame is
trimmed to the same rolling start the full fetch would have used, so it does
not grow all session. A full fetch still happens on connect, on a code or
timeframe switch, and on any detected gap.

### Known: the remaining cost at 1s

The API cost is solved, but the per-cycle local work is not: `load_local_ticks`
re-reads the whole day's ticks (~181ms measured) and `_render()` repaints
1500 bars with all overlays (~205ms). At Refresh=1s that is roughly 20% of
wall-clock spent redrawing. Usable -- confirmed at 1s -- but the next
bottleneck if it needs to go lower is the tick reload, which could be made
incremental the same way.

## v0.18.1 — the forming candle's tick profile follows the chart (2026-09-21)

### Fix: top-right tick profile went stale on the newest candle (`analysis/trade_viewer_qt.py`)

The tick profile panel is hover-driven: it repaints when the cursor moves to a
different candle, when an S/M/L/Net/VP/Imb toggle changes, or on a range
selection. `_render()` never touched it.

That is correct for a closed candle -- its ticks cannot change -- but the
newest candle is still forming. Resting the cursor on it in Live mode showed
the book as it was at the moment the cursor arrived, and it stayed there while
the candle beside it kept growing; the only way to refresh was to move the
mouse away and back.

`_render()` now repaints the panel when the hovered candle is the last one, so
it advances on the same cycle as the bar it describes (Refresh(s), default
15s). Older indices are left alone -- they are finished, and repainting them
would be pure waste. Range mode is unaffected, since `_show_tick_profile`
already returns immediately there.

Verified offscreen: redrawn when hovering the last candle, not when hovering
an older one, not when nothing is hovered; when a new candle arrives the
previous last stops refreshing and the new one starts; range mode leaves the
panel untouched; and end to end, a tick added to the forming bar shows up
without any mouse movement.

## v0.18.0 — push-driven Liquidity Heatmap (2026-09-20)

### Feat: the heatmap consumes the ORDER_BOOK push feed directly (`analysis/liq_hm_window.py`)

The heatmap used to read the collector's database on a 1s timer, so its
latency was the collector's 2s write throttle plus that poll: up to ~3s from
a book change to a pixel. It now subscribes to ORDER_BOOK itself and repaints
on the push.

Measured during the 2026-09-20 overnight session with the new probe: moomoo
pushes on a **~300ms cadence** (2,001 pushes in 600s; p10 0.294 / median 0.300
/ p90 0.306 -- regular enough that it is clearly a fixed server-side interval,
not market activity), and the collector's throttle was discarding **85.7%** of
them. End-to-end latency goes from ~1.5-3s to ~300ms, which is the feed's own
floor.

- `_PushBridge` carries snapshots from the SDK's socket thread to the GUI
  thread via a queued signal; the handler only merges and emits.
- Merging is `feeds.order_book_merge.OrderBookMerger`, shared with the
  collector, so partial pushes, side staleness and crossed books are handled
  by the one implementation that has been through production.
- The rightmost column is repainted in place while the Col(s) timer still
  rolls a new one on schedule, so the x axis stays a uniform time grid with a
  live edge. Appending per push would make it non-uniform and unreadable.
- Every push now lands in `_raw_snaps`, where before only the throttled writes
  did -- iceberg and spoof detection look for volume refreshing at a price, so
  they see every refresh the feed reports instead of one sample per 2s.
- A watchdog re-subscribes after `_PUSH_STALE_SECS` (90s) of silence from a
  feed that had been delivering. An OpenD subscription can stop delivering
  with no error surfacing, and a frozen heatmap looks exactly like a quiet
  book. Silence *before* the first push does not count, so opening the window
  hours before the session starts does not churn.
- The DB is still read for the cold-start pre-fill, and the old polling path
  survives as a fallback when the subscription fails or has not delivered yet.

### Refactor: shared ORDER_BOOK merge module (`feeds/order_book_merge.py`)

`order_book_collector.py`'s partial-push handling, side-staleness window and
crossed-book attribution move into `OrderBookMerger`, shared by the collector
and the heatmap. Pure by design: a push goes in, a complete snapshot or None
comes out; throttling and persistence stay with the collector. Verified
behaviour-identical to the inline original over 7 hand-built cases and 40
randomised push sequences, comparing the full write series.

### Fix: overlay markers froze on the push path (`analysis/liq_hm_window.py`)

Found during live validation. The push path did not call
`_redraw_orderflow_markers()` / `_load_absorb_ticks()`, so aggressor bubbles
stayed on the column they were first drawn on while the grid scrolled out from
under them. They now refresh on the column roll -- a push moves no column, and
redrawing at 3.3/s would be wasted work. `Reset` repaints the overlays too; it
only rescaled the axes before, which is why it appeared to do nothing.

### Tooling: ORDER_BOOK push-rate probe (`analysis/ob_push_probe.py`)

Times every push and reports the gap distribution plus what the collector's
throttle would discard. Writes nothing, so it runs alongside the collector.
`order_book.db` cannot answer this on its own -- every gap measured there is
floored by the throttle, not by the feed.

### Test: throttle coverage the collector never had (`tests/analysis/test_order_book_collector.py`)

`test_handler_accumulates_session_count` had been failing for as long as the
throttle existed: it called `on_recv_rsp` twice back to back and asserted both
writes landed. It now isolates accumulation from the throttle, and two new
tests cover the throttle itself (a burst inside the window produces one write;
the throttle keys on code). Confirmed by mutation.

### Known, not addressed

`order_book_snapshots` stores one row per price level, so a snapshot is ~120
rows, and `prune(keep=1000)` is row-based -- each code retains only 8-9
snapshots (~18s of history). That is why the pre-fill is so short. Removing
the collector's throttle is not viable until this changes: at the measured
3.3 pushes/s the current schema would write 400 rows/s (~0.94 GB per regular
session per code), which is what the throttle was added to prevent. One row
per snapshot would make it 3.3 rows/s.

## v0.17.1 — fix blank tick panel after using VP mode (2026-09-13)

### Fix: turning `VP` off left the tick profile panel blank (`analysis/trade_viewer_qt.py`)

Once VP mode had been used, switching back to the order-flow view showed an
empty panel with a volume-scale X axis (reported with a screenshot: axis
reading 5K-20K while nothing was drawn).

- Cause: VP mode pins its X range with `setXRange`, and pyqtgraph's `setRange`
  disables auto-range for that axis as a side effect (`disableAutoRange`
  defaults to `True`). Nothing turned it back on, so the order-flow view kept
  VP's range -- tens of thousands wide -- while its own log-scale bars span
  roughly +/-10, collapsing them into an invisible sliver at the left edge.
- Fix: `_draw_tick_profile` re-enables X auto-range on every draw, and the
  order-flow branch calls `updateAutoRange()` once its bars are in place.
  Re-enabling alone is not enough -- that only sets a flag, and the range is
  not recomputed until something forces it (verified directly against
  pyqtgraph: `enableAutoRange` left the stale range untouched,
  `updateAutoRange` restored it). Recomputing is preferred over pinning an
  explicit range so the view keeps the padding behaviour it always had.
- Verified on real ticks.db data (SOXL 2026-09-10): the order-flow X range
  comes back to within 0.01 of its never-visited-VP value, holds across eight
  alternating toggles, works when VP is the first mode drawn, and behaves in
  both hover and range-accumulation modes.

## v0.17.0 — volume-profile mode for the tick panel (2026-09-12)

### Feat: `VP` mode on the tick profile panel (`analysis/trade_viewer_qt.py`)

The top-right panel had one view of a candle's ticks: the order-flow split
(diverging buy/sell bars, log-scaled, imbalance highlighting, delta title),
which answers "who was aggressing at each price". The new `VP` checkbox
switches it to the market-profile question instead -- "where did the volume
actually trade" -- for the same hovered candle or selected range.

- One bar per price level for total volume (buy + sell + neutral), with
  POC / VAH / VAL drawn and labelled. Levels inside the value area take the
  saturated colour; the tails fade back.
- Bars are linear, not log-scaled like the order-flow view: a profile is read
  by comparing bar lengths, which log scaling destroys.
- Makes the two right-hand panels read the same way at two zoom levels -- the
  session profile below already answers this question for a whole session.
- Tick-only. A candle with no tick coverage never reaches the renderer, so
  unlike the session panel there is no OHLCV fallback: a level shown here is a
  price that genuinely traded, not one a bar's range merely spanned.
- The S/M/L order-size filters still apply. The Imb highlight does not -- it
  compares the two sides this mode merges. `VP` takes precedence over `Net`.
- Works in both single-candle hover and range-accumulation modes.

## v0.16.0 — session-profile POC hysteresis + POC trail (2026-09-10)

### Feat: POC switch hysteresis and a POC1/POC2 trail in the session profile (`analysis/trade_viewer_qt.py`)

The session profile's POC is a from-scratch `argmax` on every rebuild with no
memory of the last one, so on a two-cluster profile it teleports between the
peaks as their totals cross. Observed on SOXL 2026-09-10: four flips across
$5 (123.8 <-> 118.7) inside twelve minutes, and 91 POC changes over the day.

- **Hysteresis** (`Hold:` spinbox, default 5%, 0 disables): a rival cluster
  must hold that much more volume than the one the POC currently sits in
  before the POC moves. On the day above this takes 91 switches down to 4,
  and removes the 12-minute flip-flop entirely, leaving one decisive move.
- The comparison is over a **cluster**, not a bin. A single bin's volume is
  far too noisy to gate on: the profile is re-binned from scratch over a
  lo/hi that grows with the session, so a peak bin routinely hands a big
  slice of its volume to a neighbour. Measured rival/held single-bin ratios
  of 1.6-2.3 inside what was visually one stable cluster, which a percentage
  margin cannot separate from a real move. `_cluster_mass()` sums a fixed
  price window (`_POC_CLUSTER_PCT`, 0.5%) around each candidate instead.
- A held POC is reported at its own stored price rather than re-snapped to
  the nearest bin centre each rebuild, which would make it jitter a few cents
  on its own as the bin boundaries shift.
- **POC trail** (`POCs:` spinbox, default 2, up to 6): POC1 is the current
  level, POC2 the one it moved off, and so on, newest first — the pair reads
  as a direction of travel. Older entries fade and are dotted; levels closer
  than half a bin count as the same level. The trail resets when the profile
  is redefined (symbol / date / mode / range / bins / session filter), since
  a POC over a different set of bars is not a previous position of this one.
- `_compute_poc_vah_val()` takes an optional `poc_idx` so the value area
  expands from the held bin, keeping POC and VAH/VAL agreeing on the peak.
  VAH/VAL are widened to contain the held POC when the area collapses onto
  its own bin — a sub-bin correction that restores VAL <= POC <= VAH.

## v0.15.0 — indicator-alert rule disable + session value-area proximity alerts (2026-09-10)

### Feat: per-rule enable/disable for indicator alerts (`analysis/indicator_alert_watcher.py`, `analysis/signal_scanner.py`)

- Rules in `config/scanner/indicator_alert_params.json` take an optional
  `"enabled": false` to be parked without deleting them. A missing key means
  enabled, so every config written before this flag keeps working untouched.
- Distinct from the existing Mute button, which stays as it was: mute is a
  temporary silence the scanner auto-lifts once the reading goes negative
  again, and a muted rule keeps being evaluated so its row stays live in the
  table. A disabled rule is skipped by the scan entirely — no kline fetch, no
  DB row, no notification — until it is switched back on.
- Disabling deletes the rule's `indicator_alert_state` row and clears its
  notify-throttle timestamp, so it leaves no frozen last reading behind in the
  Live tab and re-enabling alerts immediately instead of waiting out a repeat
  window measured from before it was switched off.
- `_rule_id()` (now public as `rule_id()`) deliberately excludes `enabled` from
  a rule's identity, so toggling off and back on lands on the same row rather
  than orphaning the old one.
- The rules dialog gains a leading "On" checkbox column.

### Feat: `session_va` indicator — price proximity to the current session's POC / VAH / VAL (`analysis/indicator_alert_watcher.py`)

- New entry in `INDICATOR_REGISTRY`, so it reuses the whole existing alert
  path unchanged: no DB migration, no scanner dispatch change, and the generic
  rules dialog already edits it.
- Builds the volume profile of the session occurrence the latest bar belongs
  to and reports where price sits relative to it. Params: `session` (only run
  while that session is active; omit to follow whichever one is), `warmup_minutes`
  (freeze the profile over the session's first N minutes, default 30 — set 0
  for a developing whole-session profile), `n_bins` (default 50), `va_pct`
  (default 0.70).
- Fields per level (`poc` / `vah` / `val`): `dist_*_pct` and `dist_*_abs` are
  the unsigned gap, for proximity rules (`condition: below`); `off_*_pct` and
  `off_*_abs` are signed (+ above / − below), for break rules against a
  threshold of 0. Distance is therefore configurable as either a percentage or
  an absolute price. `poc`/`vah`/`val`/`price` are exposed as fields too.
- Frozen (`warmup_minutes > 0`) is the default because a developing VAH tracks
  price as price makes new highs, so a "near VAH" rule on a rolling profile
  self-triggers on the breakout it is meant to anticipate. The rolling mode is
  kept for watching a level that is still forming.
- OHLCV-based (`strategy.smc.fvg.compute_volume_profile` +
  `strategy.session_vp.profile.compute_value_area`), no `ticks.db` dependency —
  the same choice `strategy/session_vp` makes, since the scanner runs on
  symbols and dates that may have no tick coverage.
- Returns full-length arrays with only the last bar filled: `scan_indicator_alert()`
  reads `series[-1]`, and a true per-bar series would rebuild the profile once
  per bar for a value nothing reads.
- Tests: `tests/analysis/test_indicator_alert_watcher.py` (18 cases).

### Fix: overlapping Call/Put Wall labels rendered as an illegible smudge (`analysis/trade_viewer_qt.py`)

- Option wall labels were all right-anchored at the last bar, so walls at the
  same or adjacent strikes (a call and a put wall on one strike, or ranks #2/#3
  one strike apart) drew their text on top of each other.
- New `_pin_option_wall_labels()` lays them out in columns: labels stay glued
  to their own price (moving one vertically would misattribute it to the wrong
  level) and a label too close to the one above it shifts one column leftward.
  Recomputed on `sigRangeChanged`, since the pixel gap between two prices
  depends on the current Y zoom.
- Label text is no longer alpha-faded by wall rank — only the dashed line is.
  Same rule the AVWAP sigma-band labels already follow: a faint line is visual
  hierarchy, faint text is just unreadable. Labels also gained the `_BG_TIP`
  background fill used by the POC/VAH/VAL labels, went 7pt → 8pt, and sit at
  `zValue=55` above the candles.

## v0.14.1 — live tick handler fixes (2026-07-06)

### Fix: Live mode Δ Delta / CVD / Tick Profile / heatmap never updated from live ticks (`analysis/trade_viewer_qt.py`)

- `_TickHandler.on_recv_rsp` (inside `_start_live`) read `row["direction"]` from
  the live TICKER push DataFrame, but the moomoo SDK names that column
  `ticker_direction`. This raised `KeyError` on every single tick since the
  handler was first added (2026-05-26) — not a regression from the
  moomoo-api 10.5.6508 → 10.7.6708 bump, the old wheel already used
  `ticker_direction` too. The exception was silently swallowed by the SDK's
  callback executor (logged as a warning only), so `_live_ticks` stayed
  permanently empty and the bug went unnoticed: `ticks.db`, kept fresh by
  the separate `analysis/tick_collector.py` collector process, backfilled
  the gap within a refresh cycle in normal use. Only the newest 1-3
  still-forming bars — not yet covered by that backfill — visibly showed no
  delta/heatmap.
- Fixed: read `row["ticker_direction"]` instead.

### Fix: Live Δ Delta briefly showed absurdly inflated (M/B-scale) values after the above fix (`analysis/trade_viewer_qt.py`)

- Fixing the field-name bug above exposed a second, more serious bug: two
  places (`DataFetcher.run()` and `_render()`) merged the in-process live
  tick accumulator (`_live_ticks`) by ADDING it on top of the freshly
  reloaded `ticks.db` data for any bucket present in both — double-counting
  the same real trades, since ticks.db is independently kept live-updated by
  the external collector process on the same feed.
- Worse, `_render()` writes its merged result back into `self._ticks`, and
  is also invoked from `_on_indicator_toggle` (any checkbox toggle) using
  whatever `self._ticks` currently holds — so every re-render (not just
  every fetch cycle) added another full copy of the ever-growing
  `_live_ticks` on top, compounding without bound over a long-running
  session.
- Fixed: both merge sites now treat `ticks.db` as authoritative once a
  bucket appears there — `_live_ticks` only fills buckets `ticks.db` doesn't
  have yet, never adds to one already present.
- Also: `_on_tf_changed` now clears `_live_ticks` (previously cleared only
  on code switch). Bucket keys are aligned to `candle_mins`, so switching
  timeframe changes the alignment and stale entries from the old alignment
  could otherwise collide with a new timeframe's bucket key.

## v0.14.0 — CVD/A-D subplots, Range Profile direction stats, pin-to-top (2026-06-29)

### Feat: Cumulative Volume Delta (CVD) subplot (`analysis/trade_viewer_qt.py`)

- New "CVD" indicator toggle adds a subplot below the candle chart, X-linked
  to the main chart like the existing KD/MAVOL subplots.
- Per-bar delta = buy_vol - sell_vol from tick-tagged trades (same bucket
  source as the Δ Delta overlay), cumulatively summed from the start of the
  loaded klines (no per-session reset). Bars with no tick coverage contribute
  0 (line holds flat rather than gapping).
- Caveat: moomoo's `ticker_direction` tags a large share of same-price prints
  NEUTRAL (~68.5% of volume observed on SOXL), which the delta excludes — CVD
  here only reflects the tagged subset of volume, not the full tape.

### Feat: Accumulation/Distribution Line subplot (`analysis/trade_viewer_qt.py`)

- New "A/D" indicator toggle adds a second subplot using `ta.volume.AccDistIndexIndicator`
  (close position within each bar's high-low range, weighted by volume).
- Pure OHLCV indicator — no tick data required, so it has no NEUTRAL-volume
  blind spot, but it's a price-action proxy for direction rather than actual
  tagged trade sides. Intended to be read alongside CVD: agreement between
  the two (real tagged flow vs. price-action proxy) is a stronger signal
  than either alone.

### Fix: CVD / A-D subplots rendered as a flat line (`analysis/trade_viewer_qt.py`)

- Both are unbounded running cumulative sums (no per-session reset), so
  PyQtGraph's default bounding-rect Y autorange fit the *entire* loaded
  history. The default 150-bar view is a small slice of that history, so its
  real variation was dwarfed by the full-history range and rendered as a
  near-flat line — same class of bug the Volume subplot was already fixed
  for (see `_reset_view` docstring).
- Fixed: `_reset_view` now also derives CVD/A-D Y range from the visible
  bars only, mirroring the existing Volume subplot fix.

### Feat: Buy/Sell/Neutral/Medium volume stats in Range Profile (`analysis/trade_viewer_qt.py`)

- `_compute_profile_bins` now also returns a `stats` dict (`total`, `buy`,
  `sell`, `neutral`, `medium`) alongside the existing price-bin histogram.
- Range Profile panel header shows `Tot / Buy / Sell / Neu / Mid*` for the
  selected range, with a hover tooltip explaining the relationships.
- `total` = `buy + sell + neutral` from tick-covered bars only (a closed,
  self-checkable sum); `medium` (`Mid*`) is a buy_m+sell_m size breakdown
  *within* buy+sell, not a separate additive bucket.

### Feat: Pin-to-top button on main viewer (`analysis/trade_viewer_qt.py`)

- Added a "📌 Pin" toggle button to the main toolbar (same
  `WindowStaysOnTopHint` pattern already used by the Liquidity Heatmap window).

### Fix: LiqHm window could crash the whole process on close (`analysis/liq_hm_window.py`)

- `_reset_grid()` (stale-worker discard on code switch) and `set_live()`
  (prefill restart) dropped the last Python reference to a still-running
  `QThread`, which destroys the C++ object while the OS thread is alive —
  Qt aborts the process with "QThread: Destroyed while thread is still
  running". `closeEvent` previously masked this with `terminate()`, which
  carries its own risk of corrupting the GIL if the kill lands while the
  thread holds it (observed as the main window going unresponsive after
  repeatedly closing the heatmap window).
- Fixed: new `_retire_worker()` disconnects the result signal and, if the
  worker is still running, holds a module-level reference until its own
  `finished` signal confirms the thread actually exited — no `terminate()`,
  no dropped live references.

### Chore: moomoo-api 10.5.6508 → 10.7.6708 (`pyproject.toml`, `uv.lock`)

- Picks up the `order_book_type` param on `get_order_book` (NORMAL/ODD board)
  and `SubType.ORDER_BOOK_ODD`; verified backward compatible with all
  existing `ORDER_BOOK` push call sites (no code changes needed).

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
