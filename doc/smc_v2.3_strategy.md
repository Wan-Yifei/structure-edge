# SMC Strategy — What's New in `smc_v2.3`

**Git tag:** `smc_v2.3`  
**Date:** 2026-05-29  
**Prior tag:** `smc_v2.2`  
**Full parameter reference:** [`strategy/smc/STRATEGY.md`](../strategy/smc/STRATEGY.md)  
**Full version history:** [`doc/smc_v2_strategy.md`](smc_v2_strategy.md)

---

## 1. `determine_trend` — veto rule (behaviour change)

**File:** `strategy/smc/market_structure.py`

### Problem (`smc_v2.2` behaviour)

In `smc_v2.2` the rule was: after a CHoCH the counter was reset to 0, and at
least one same-direction BOS had to follow before the trend was confirmed.  This
was too strict — in practice the rolling HTF window often contains only `[CHoCH
bear]` without a subsequent `[BOS bear]`, causing valid setups to be skipped.

### New rule

A **CHoCH alone immediately confirms** the trend direction (equivalent to
`consecutive = 1`).  Same-direction BOS signals that appear after it increment
the counter further (relevant only when `bos_count > 1`).

**Veto**: if any BOS in the *opposite* direction appears after the last CHoCH in
the window, `determine_trend` returns `None`.  A reverse BOS proves that price
reclaimed the structural level that the CHoCH broke through — the setup is
structurally uncertain and no trade is taken.

### Signal list examples

| Signals in rolling window | `smc_v2.2` result | `smc_v2.3` result |
|---|---|---|
| `[CHoCH bear]` | `None` (no BOS) | `"bear"` ✓ |
| `[CHoCH bear, BOS bear]` | `"bear"` | `"bear"` ✓ |
| `[CHoCH bear, BOS bull]` | `"bear"` (bull BOS ignored) | `None` (veto) ✓ |
| `[CHoCH bear, BOS bear, BOS bull]` | `"bear"` | `None` (veto) |
| `[BOS bear, BOS bear]` (no CHoCH) | `"bear"` | `"bear"` (unchanged) |

The third row is the critical fix: `eda9d355` (CSCO) showed a `[CHoCH bear, BOS
bull]` window where the reverse BOS proved the prior bearish structure was
invalidated, yet v2.2 entered short anyway.

### `bos_count` parameter semantics

| `bos_count` | Minimum required |
|---|---|
| `1` (default) | CHoCH alone is sufficient |
| `2` | CHoCH + at least one same-direction BOS |
| `N` | CHoCH counts as 1; need N−1 additional same-direction BOS signals |

The reverse-BOS veto always applies regardless of `bos_count`.

---

## 2. BOS scan boundary fix (`detect_bos_choch`)

**File:** `strategy/smc/market_structure.py`

### Problem

When the engine processed swing `i` (a new high), it referenced the previous
swing high as the structural level and scanned forward for a close above its wick.
The scan started at `sw["idx"] + 1`, skipping the swing bar itself.

If the current swing's own close already exceeded the prior swing's wick (i.e.,
the break happened *at* the swing), the scan would miss this and continue
scanning beyond the swing into later lower bars — producing a BOS line that
visually crossed over the obvious structural high.

### Fix

The scan now starts at `sw["idx"]` (inclusive) so the swing bar itself is the
first candidate for the break bar.

Additionally, the scan stops before the *next* swing of the same kind
(`scan_end_h = next_high["idx"]`).  Without this cap, the scan could extend into
a period where a different higher swing had already formed, producing a BOS whose
reference level is stale with respect to current structure.

```
Before:  for j in range(sw["idx"] + 1, n):
After:   for j in range(sw["idx"], next_same_swing["idx"]):
```

Both the bull (high-break) and bear (low-break) paths were fixed symmetrically.

---

## 3. Infrastructure changes (no algo impact)

### Per-stock output subdirectories

Each stock's results now land in `<run_dir>/<CODE_slug>/`:

```
backtest/results/20260529_…_grid/
    US_NVDA/
        run_US_NVDA.log
        results_US_NVDA.csv
        viz_US_NVDA.png
        report_US_NVDA.html
    US_AMD/
        …
```

### Self-contained HTML reports

`report.py` embeds the Plotly JS bundle inline (`~3 MB`) instead of loading from
CDN.  Reports open correctly without an internet connection.

### UTF-8 config loading on Windows

`_load_json_config` in `run.py` opens config JSON files with `encoding="utf-8"`,
fixing a `UnicodeDecodeError` on Windows systems whose default locale is GBK/CP936.

### Infinite `profit_factor` guard

`_cap_inf_pf()` replaces `float("inf")` in the `profit_factor` column with the
largest finite value before stats are computed, preventing `RuntimeWarning` from
NumPy arithmetic on infinite values.

---

## Upgrade notes

- No parameter schema changes — existing `BacktestParams` dicts and config JSONs
  are forward-compatible.
- The `bos_count` default remains `1`; existing configs will now allow CHoCH-only
  confirmations (previously required a BOS).  This may increase trade frequency
  slightly for windows where only CHoCH was present.
- Trades whose ID hash was computed under `smc_v2.2` will not collide with v2.3
  trades because `ALGO_VERSION` is included in the hash.
- Re-run any previous backtest results directories under the new tag to get
  comparable versioned trade IDs.
