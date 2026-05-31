# SMC Strategy — Logic & Parameter Reference

## Overview

Multi-timeframe Smart Money Concepts (SMC) strategy. A higher timeframe (HTF)
establishes trend direction and marks Fair Value Gap (FVG) zones; a lower
timeframe (LTF) triggers the entry when price retraces into a zone at sufficient
depth.

---

## Pipeline (one LTF bar at a time)

### Step 1 — HTF structure update

Triggered only when a new HTF bar closes (O(log n) lookup via binary search).
Recomputes the rolling HTF window `[htf_pos - htf_window_bars, htf_pos]`.

Within that window:
- `find_swings` — alternating swing highs/lows with `swing_lookback` bars on each side.
- `detect_bos_choch` — Break-of-Structure and Change-of-Character signals.
- `detect_fvg` — three-candle gaps (FVGs) above `fvg_min_width_pct`.
- `compute_volume_profile` — volume distribution bins (used by the LVN filter).

### Step 2 — Trend determination

Each method in `htf_trend_methods` produces `"bull"` / `"bear"` / `None`.
**All methods must agree** on the same non-None direction; otherwise `trend = None`
and the bar is skipped.

#### Method: `bos_choch`
Uses `determine_trend(bos_signals, bos_count)`.

A **CHoCH immediately confirms** the trend direction (`consecutive = 1`).
Same-direction BOS signals that follow increment the counter further.
The `bos_count` parameter controls the minimum required count (CHoCH alone
satisfies `bos_count = 1`; CHoCH + 1 BOS satisfies `bos_count = 2`).

**Veto rule**: if any BOS in the *opposite* direction appears after the last CHoCH
in the rolling window, `determine_trend` returns `None` regardless of the counter.
This prevents entries when price has already reclaimed the structural level that
the CHoCH broke through.

#### Method: `kd`
Uses `kd_trend()` — a KD channel indicator (fast/slow EMA channel midline spread).
Two operating modes (selected by `kd_smooth`):

- **Adaptive** (`kd_smooth > 0`, default): segments the spread by zero-crossings
  of the smoothed width, with conditional lag compensation and minimum segment
  length enforcement. The current segment's `|avg_width| / avg_ATR` is compared
  to `kd_atr_threshold`.
- **Legacy** (`kd_smooth = 0`): averages width over the last `kd_window` bars and
  classifies the window as flat if either `kd_flat_threshold` or `kd_atr_threshold` is triggered.

### Step 3 — FVG touch detection

For a bull trend the wick price is the bar **low**; for bear it is the bar **high**.
An FVG is "touched" when the wick enters the zone on the correct side:
- Bull: `wick <= fvg.top`
- Bear: `wick >= fvg.bottom`

Active FVGs must satisfy:
- `not filled` — the gap has not been fully closed by a prior bar.
- `direction == trend` — bull FVG in a bull trend, bear FVG in a bear trend.
- Age ≤ `fvg_max_age_bars` HTF bars since the FVG formed.
- Not already used today (each zone resets at the start of each trading day).

If multiple FVGs are touched simultaneously, the one whose midpoint is nearest to
the wick price is selected.

### Step 4 — Entry depth check

`depth = (fvg.top - wick) / (fvg.top - fvg.bottom)` for bull (0 = zone edge, 1 = far side).

Entry proceeds only when `depth >= fvg_entry_depth_pct`.

### Step 4a — Over-refilling guard

If the bar **close** has punched through the far side of the FVG, the gap has been
over-filled (a reversal signal). The bar is rejected:
- Bull: `close < fvg.bottom` → skip.
- Bear: `close > fvg.top` → skip.

### Step 5a — LVN overlap filter *(optional)*

When `require_lvn_overlap = True`, the FVG zone must overlap a Low Volume Node
(LVN) in the HTF volume profile. A zone qualifies as LVN when its average volume
density is below `lvn_threshold × max_bin_volume`.

### Step 5b — Displacement candle filter *(optional)*

When `displacement_required = True`, the FVG-creating candle (middle of the
three-bar gap) must be a "displacement" (strong momentum) candle:
- Range > `displacement_atr_mult × mean_range` over the prior `displacement_lookback` bars.
- Body / Range ratio ≥ `displacement_body_ratio`.

### Step 5c — Gap-fill filter *(optional)*

When `gap_fill_lookback > 0`, the engine scans the `gap_fill_lookback` LTF bars
ending at (and including) the first FVG touch bar.  If any bar opens with a gap
in the fill direction, the touch is rejected:
- Bear FVG (fill direction = up): `open > prev_close × (1 + gap_fill_min_pct)`
- Bull FVG (fill direction = down): `open < prev_close × (1 − gap_fill_min_pct)`

The intent is to catch sessions where the market opened aggressively toward the
FVG zone (e.g. a large overnight gap-up before a bear FVG touch), indicating that
momentum may have shifted and the FVG is more likely to be blown through than to
act as a reversal point.

### Step 6 — LTF structure confirmation *(optional)*

When `require_ltf_confirmation = True`, a CHoCH + BOS sequence in the trend
direction must appear on the LTF **within the current HTF candle** and **after
the wick first entered the FVG zone**.

### Step 6b — LTF trend-bar confirmation *(optional)*

When `require_ltf_trend_bar = True`, the entry bar itself must close in the
trend direction: `close > open` for a bull entry, `close < open` for a bear
entry.  This is a looser, bar-level momentum check that does not require a
structural signal.  Steps 6 and 6b are independent and can be combined.

### Step 7 — SL / TP levels

Primary anchor (swing-based):
- **Bull SL**: highest swing low below the entry close, minus `sl_buffer_pct`.
- **Bull TP**: nearest swing high above the entry close.
- **Bear SL**: lowest swing high above the entry close, plus `sl_buffer_pct`.
- **Bear TP**: nearest swing low below the entry close.

KD slow-channel fallback (when `kd_sl_fallback = True`):
- Activated when no swing candidate exists for SL/TP, or when the swing-based SL exceeds `max_sl_pct`.
- Uses the KD slow channel boundary: `lo2` (lower band) as bull SL, `up2` (upper band) as bear SL.
- Accepted only if the fallback SL, after `sl_buffer_pct`, satisfies both `max_sl_pct` and `min_rr`.

Trade is skipped when:
- No valid SL/TP can be found from either source.
- `SL distance / entry_price > max_sl_pct`.
- `(TP - entry) / (entry - SL) < min_rr`.

### Step 8 — Trade management

Entry at close of the triggering LTF bar. Exit by vectorised scan (no bar loop):
first bar where `low ≤ SL` (loss) or `high ≥ TP` (win). After `max_bars_in_trade`
LTF bars the trade times out at the close.

When `intraday_only = True`, exit is capped to end-of-day; position is
force-closed at the day's last bar close if still open.

---

## Parameter Reference

### Timeframes

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trend_tf` | `"60m"` | HTF: used for trend, swing points, and FVG detection. |
| `entry_tf` | `"15m"` | LTF: used for entry timing and (optionally) confirmation structure. |

### Structure detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swing_lookback` | `2` | Bars on each side required to confirm a swing high/low. Higher = fewer, larger swings. |
| `bos_count` | `1` | Minimum `(CHoCH + same-direction BOS)` count to confirm the trend. `1` = CHoCH alone is enough; `2` = CHoCH + at least one confirming BOS required. A reverse BOS after the CHoCH always vetoes the trend regardless of this value. |
| `htf_window_bars` | `20` | Rolling HTF window size. Controls how much history is used for swings, BOS, and FVGs. ~5 h at 15 m. |

### FVG filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fvg_min_width_pct` | `0.002` | Minimum FVG size as a fraction of price (0.002 = 0.2%). Filters noise gaps. |
| `fvg_entry_depth_pct` | `0.10` | How deep into the FVG the wick must reach before entry. 0 = zone edge, 1 = far side. |
| `fvg_max_age_bars` | `50` | FVGs older than this many HTF bars are ignored. Prevents stale zones from triggering entries. |

### Displacement filter

| Parameter | Default | Description |
|-----------|---------|-------------|
| `displacement_required` | `False` | Enable displacement candle requirement. |
| `displacement_atr_mult` | `1.5` | The FVG-creating candle's range must exceed `mult × baseline_mean_range`. |
| `displacement_body_ratio` | `0.5` | Body / total range of the FVG-creating candle. 0 = doji, 1 = marubozu. |
| `displacement_lookback` | `5` | Bars used to compute the baseline mean range. |

### Entry confirmation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `require_ltf_confirmation` | `False` | Require LTF CHoCH + BOS in trend direction after zone touch. More selective, fewer trades. |
| `require_ltf_trend_bar`    | `False` | Require the entry bar to close in the trend direction (`close > open` for bull; `close < open` for bear). Looser than `require_ltf_confirmation`; can be used alone or combined with it. |

### Gap-fill filter

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gap_fill_lookback` | `0` | Bars in the window ending at the first FVG touch to scan for a fill-direction gap. `0` = filter disabled. Typical values: 3–5. |
| `gap_fill_min_pct`  | `0.001` | Minimum gap size to trigger the filter (fraction of prev close, e.g. `0.001` = 0.1%). Prevents noise from micro-gaps. |

### LVN filter

| Parameter | Default | Description |
|-----------|---------|-------------|
| `require_lvn_overlap` | `False` | FVG zone must sit in a Low Volume Node of the HTF volume profile. |
| `lvn_threshold` | `0.30` | LVN criterion: zone volume density < `threshold × max bin volume`. |

### Risk management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sl_buffer_pct` | `0.001` | Extra margin added beyond the swing level when placing SL (0.001 = 0.1%). |
| `max_sl_pct` | `0.005` | Skip trade if `SL distance / entry_price` exceeds this. Guards against over-wide stops. |
| `min_rr` | `2.0` | Minimum reward-to-risk ratio `(TP − entry) / (entry − SL)`. |
| `kd_sl_fallback` | `False` | When `True`, use the KD slow-channel boundary (`lo2` for bull SL, `up2` for bear SL) as a fallback anchor when: (a) no swing candidate exists for SL/TP, or (b) the swing-based SL exceeds `max_sl_pct`. The fallback SL is accepted only if, after applying `sl_buffer_pct`, it still satisfies both `max_sl_pct` and `min_rr`. |

### Trade behaviour

| Parameter | Default | Description |
|-----------|---------|-------------|
| `allow_short` | `True` | When `False`, only bull (long) setups are taken. |
| `intraday_only` | `False` | When `True`, force-close any open position at end of the trading day. |

### Trend methods

| Parameter | Default | Description |
|-----------|---------|-------------|
| `htf_trend_methods` | `("bos_choch",)` | Ordered list of trend methods. All must agree. Options: `"bos_choch"`, `"kd"`. |
| `htf_trend_params` | `{}` | Per-method config dict. Keys prefixed by method name (see below). |

#### KD trend method params (via `htf_trend_params`)

| Key | Default | Description |
|-----|---------|-------------|
| `kd_fast` | `25` | EMA span for the fast channel (higher/lower midline). |
| `kd_slow` | `90` | EMA span for the slow channel. |
| `kd_smooth` | `3` | Pre-smoothing window for zero-crossing detection. `> 0` = adaptive segment mode; `0` = legacy fixed-window mode. |
| `kd_min_bars` | `3` | Adaptive mode only. Segments shorter than this are merged into the preceding segment. |
| `kd_atr_threshold` | `0.036` | Segments where `\|avg_width\| / avg_ATR < threshold` are classified as flat. Scale-invariant. Default 0.036 = p25 of empirical distribution on SNDK 1545-bar HTF history. |
| `kd_window` | `10` | Legacy mode only. Number of bars to average width over. |
| `kd_flat_threshold` | `0.0` | Legacy mode only. Price-unit flat filter (superseded by `kd_atr_threshold`; keep at 0). |
| `kd_atr_period` | `14` | ATR rolling period used for normalisation. |

---

## Consensus rule

When `htf_trend_methods = ("bos_choch", "kd")`:
- Both methods are evaluated independently on each HTF bar.
- `trend` is set to `"bull"` or `"bear"` only when both return the same non-None value.
- If either returns `None` or they disagree, `trend = None` and all LTF bars in that HTF period are skipped.

---

## Versioning

The engine reads the most recent `smc_v*` git tag (`ALGO_VERSION`) and includes it
in each trade's ID hash to prevent cross-version primary-key collisions in the
trade database.
