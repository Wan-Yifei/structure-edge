# SMC FVG Backtest System — Design Guide

> This document reflects the actual codebase. Do not assume a `src/` layout —
> existing modules live at the repo root (`strategy/`, `backtest/`, `feeds/`, `analysis/`).
> All new work extends existing code; nothing is rewritten from scratch.

---

## 1. Goals

| Goal | Status |
|---|---|
| Grid search over SMC parameters | Done (`backtest/run.py`) |
| Checkpoint / resume | Done (`backtest/run.py`) |
| Parallel execution | Done (ProcessPoolExecutor) |
| **Engine speed — 1m entry TF viable** | **Planned (Track A)** |
| **Structured logging across workers** | **Planned (Track B0)** |
| **Trade auditability (DB + trade_id)** | **Planned (Track B)** |
| **Interactive trade reviewer** | **Planned (Track B)** |

---

## 2. Existing Structure (Do Not Reorganise)

```
backtest/
├── engine.py          ← single-combo backtest loop (BacktestParams → BacktestResult)
├── run.py             ← grid orchestrator, ProcessPoolExecutor, checkpoint
├── viz.py             ← matplotlib equity / param heatmap charts
└── results/
    └── checkpoints/   ← <hash>.pkl resume files

strategy/
└── smc/
    ├── fvg.py         ← detect_fvg, fvg_entry_depth, compute_volume_profile
    ├── market_structure.py  ← find_swings, detect_bos_choch, determine_trend
    ├── confirmation.py      ← check_ltf_confirmation
    └── order_blocks.py

feeds/
└── fetcher.py         ← Moomoo API → DuckDB cache (db/backtest_klines.duckdb)

db/
└── backtest_klines.duckdb   ← market data (OHLCV, read-only during backtest)

config/
└── backtest.json      ← codes, start/end, workers, tf_pairs, tf_pairs_fast
```

---

## 3. New Modules (Files to Add)

```
backtest/
├── logger.py          ← NEW (B0): multiprocessing-safe logging
├── bench.py           ← NEW (A1): benchmark harness
├── db.py              ← NEW (B):  backtest.duckdb schema + writer
├── stats.py           ← NEW (B):  Sharpe, heatmap, time-distribution
└── trade_reviewer.py  ← NEW (B):  Plotly interactive trade chart

strategy/
└── base.py            ← NEW (B):  BaseStrategy ABC

db/
└── backtest.duckdb    ← NEW (B):  runs / trades / run_stats (write target)
```

### Dependency Order

```
logger.py              (no internal deps)
    ↓
bench.py               (engine, feeds)
engine.py  [modified]  (logger)
run.py     [modified]  (logger, db)
    ↓
db.py                  (logger)
stats.py               (db)
trade_reviewer.py      (db, feeds)
    ↓
base.py                (no internal deps — strategy interface only)
```

---

## 4. Track A — Performance (Implement First)

### A1 · Benchmark (`backtest/bench.py`)

Measure baseline before any optimisation. Run 5 fixed combos and report:
- Total elapsed per combo
- Number of trades generated
- Average `bars_held` per trade
- Estimated time for a full 1m grid run

```python
# backtest/bench.py
def run_benchmark(
    tf_pairs: list[tuple[str, str]],
    n_combos: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a DataFrame: tf_pair | combo | elapsed_s | n_trades | avg_bars_held"""
```

Run:
```
uv run python -m backtest.bench
```

### A2 · Vectorised Trade Exit (`backtest/engine.py`)

**Problem**: while a trade is open the engine walks bar-by-bar in Python
(up to `max_bars_in_trade` = 200 iterations per trade).  
For 1m data with ~86 k bars this dominates runtime.

**Fix**: replace the `for i` loop with `while i`, and on entry call `_find_exit()`
which uses `np.argmax` to jump directly to the exit bar.

```python
def _find_exit(
    lows: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    from_bar: int,
    sl: float,
    tp: float,
    direction: str,       # "bull" | "bear"
    max_bars: int,
) -> tuple[int, float, str]:
    """Return (exit_bar_abs, exit_price, result).  Pure numpy, no Python loop."""
    end = min(from_bar + max_bars, len(lows))
    lo  = lows[from_bar:end]
    hi  = highs[from_bar:end]

    sl_mask = (lo <= sl) if direction == "bull" else (hi >= sl)
    tp_mask = (hi >= tp) if direction == "bull" else (lo <= tp)

    first_sl = int(np.argmax(sl_mask)) if sl_mask.any() else max_bars
    first_tp = int(np.argmax(tp_mask)) if tp_mask.any() else max_bars

    if first_sl == max_bars and first_tp == max_bars:
        j = end - 1
        return j, float(closes[j]), "timeout"
    if first_sl <= first_tp:
        return from_bar + first_sl, sl, "loss"
    return from_bar + first_tp, tp, "win"
```

Main loop change (schematic):

```python
# Before
for i in range(_WARMUP, n_ltf):
    if active_trade is not None:
        # ... check SL/TP bar by bar ...
        continue

# After
i = _WARMUP
while i < n_ltf:
    if active_trade is not None:
        exit_bar, exit_price, result = _find_exit(
            ltf_lows, ltf_highs, ltf_cls,
            from_bar=i, sl=active_trade.sl, tp=active_trade.tp,
            direction=active_trade.direction, max_bars=max_bars_in_trade,
        )
        # record trade ...
        i = exit_bar + 1
        active_trade = None
        continue
    # ... signal scanning unchanged ...
    i += 1
```

Expected speedup: 5–20× for 1m entry TF (trade duration ~100–300 bars).

**Correctness check**: run `--fast` before and after; results must be identical for the same params.

### A3 · Random Search (`backtest/run.py`)

Replace exhaustive Cartesian product with random sampling when `--random N` is passed.

```python
def build_param_list_random(
    pairs: list[tuple[str, str]],
    grid: dict,
    n_samples: int = 300,
    seed: int = 42,
) -> list[BacktestParams]:
    rng = random.Random(seed)
    result = []
    for trend_tf, entry_tf in pairs:
        for _ in range(n_samples):
            combo = {k: rng.choice(v) for k, v in grid.items()}
            result.append(BacktestParams(trend_tf=trend_tf, entry_tf=entry_tf, **combo))
    return result
```

CLI:
```
uv run backtest/run.py --random 300    # random search, 300 samples per TF pair
uv run backtest/run.py                 # exhaustive grid (unchanged behaviour)
```

Checkpoint key must include `n_samples` and `seed` to avoid collisions with exhaustive runs.

**Why 300**: if the top 1 % of combos are "good" (≈ 39 out of 3 888), the probability
of missing all of them in 300 random draws is < 5 %. For top 10 %, coverage is near 100 %.

**Recommended pruned grid** (use with `--random` to reduce wasted draws):

```python
PARAM_GRID_PRUNED = {
    "swing_lookback":           [2, 3],
    "bos_count":                [1, 2],
    "fvg_min_width_pct":        [0.002, 0.004],
    "fvg_entry_depth_pct":      [0.10, 0.30, 0.50],
    "require_ltf_confirmation": [True, False],
    "displacement_required":    [False],
    "sl_buffer_pct":            [0.001, 0.003],
    "max_sl_pct":               [0.005, 0.010],
    "min_rr":                   [1.5, 2.0],
}
# 2×2×2×3×2×1×2×2×2 = 384 combos per pair (was 3 888)
```

### A4 · Escalation Path (if A2+A3 insufficient)

| Step | Technique | Effort | Expected gain |
|---|---|---|---|
| A2 | numpy vectorised exit | 1 day | 5–20× |
| A3 | random search 300 | 1 day | 10× fewer combos |
| A5 | Numba `@njit` on inner loop | 2–3 days | additional 5–10× |
| A6 | Rust + PyO3 (engine rewrite) | weeks | 50–100× — only after strategy stabilises |

Do **not** use Rust until A2–A5 are profiled and the strategy logic is frozen.

---

## 5. Track B0 — Logging (`backtest/logger.py`)

**Must be implemented before B** — all new modules depend on it.

### Why a queue is required

`ProcessPoolExecutor` on Windows uses `spawn` mode: each worker is an independent
process with its own file handles. Multiple workers writing to the same log file
causes interleaved / truncated output. The fix: workers enqueue `LogRecord` objects;
a single listener in the main process performs all I/O.

```
Worker-0 ──→ ┐
Worker-1 ──→ ├──→ multiprocessing.Queue ──→ QueueListener ──→ run.log
Worker-N ──→ ┘                                             ──→ stderr
```

### API

```python
# backtest/logger.py

LOG_FORMAT = "%(asctime)s [%(tag)-26s] %(levelname)-7s %(message)s"

def make_listener(log_path: str | Path) -> tuple[Queue, QueueListener]:
    """Call in main process before spawning workers. Call listener.start() on result."""

def worker_init(log_queue: Queue, level: int = logging.INFO) -> None:
    """Pass as ProcessPoolExecutor(initializer=worker_init, initargs=(q, level))."""

def get_logger(tag: str) -> LoggerAdapter:
    """Returns a logger pre-tagged with [tag]. Use like a standard logger."""
```

### Tag format

| Context | Tag example |
|---|---|
| Worker (backtest) | `W03 4h/15m lb2 bos1` |
| Main process | `main` |
| Trade reviewer | `reviewer a1b2c3d4` |

### Log levels

| Level | Content | Default |
|---|---|---|
| `INFO` | trade open/close, every 500-combo checkpoint | yes |
| `DEBUG` | per-bar FVG / SL / TP state | `--verbose` only |
| `WARNING` | data gaps, invalid param combos | yes |
| `ERROR` | single-combo failure (does not abort the run) | yes |

### Output paths

- Backtest run: `backtest/results/<timestamp>/run.log`
- Trade reviewer: `backtest/results/<timestamp>/review.log`

---

## 6. Track B — Auditability

### B1 · Database Schema (`backtest/db.py`)

New file: `db/backtest.duckdb` (alongside existing `db/backtest_klines.duckdb`).

```sql
CREATE TABLE IF NOT EXISTS runs (
    run_id       VARCHAR PRIMARY KEY,
    config_hash  VARCHAR NOT NULL,
    config_json  JSON    NOT NULL,
    symbol       VARCHAR NOT NULL,
    trend_tf     VARCHAR NOT NULL,
    entry_tf     VARCHAR NOT NULL,
    start_date   DATE    NOT NULL,
    end_date     DATE    NOT NULL,
    status       VARCHAR NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
    created_at   TIMESTAMP DEFAULT now(),
    finished_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id     VARCHAR PRIMARY KEY,
    run_id       VARCHAR NOT NULL REFERENCES runs(run_id),
    symbol       VARCHAR NOT NULL,
    direction    VARCHAR NOT NULL,     -- 'bull' | 'bear'
    entry_time   TIMESTAMP NOT NULL,
    entry_price  DOUBLE  NOT NULL,
    sl_price     DOUBLE  NOT NULL,
    tp_price     DOUBLE  NOT NULL,
    exit_time    TIMESTAMP,
    exit_price   DOUBLE,
    result       VARCHAR,              -- 'win' | 'loss' | 'timeout'
    r_multiple   DOUBLE,
    bars_held    INTEGER
);

CREATE TABLE IF NOT EXISTS run_stats (
    run_id          VARCHAR PRIMARY KEY REFERENCES runs(run_id),
    n_trades        INTEGER,
    win_rate        DOUBLE,
    profit_factor   DOUBLE,
    sharpe_ratio    DOUBLE,
    max_drawdown_r  DOUBLE,
    avg_r           DOUBLE,
    total_r         DOUBLE,
    computed_at     TIMESTAMP DEFAULT now()
);
```

```python
# backtest/db.py
class BacktestDB:
    def __init__(self, db_path: Path): ...

    def get_or_create_run(self, config_hash: str, config_json: dict,
                          symbol: str, trend_tf: str, entry_tf: str,
                          start: str, end: str) -> tuple[str, bool]:
        """Return (run_id, is_new). Reuses existing run if config_hash matches."""

    def write_trades(self, run_id: str, trades: list[Trade]) -> None:
        """Batch insert in a single transaction."""

    def write_stats(self, run_id: str, stats: dict) -> None: ...

    def mark_done(self, run_id: str) -> None: ...
```

**Concurrency**: workers return `BacktestResult` objects via `Future`; the main process
calls `db.write_trades()` serially inside `as_completed()`. No concurrent writers.
Existing `PARAM_GRID` + CSV output is preserved — DB is an additional layer.

### B2 · Statistics (`backtest/stats.py`)

Extends the existing `BacktestResult.summary_dict()`.

```python
def compute_stats(trades: list[Trade]) -> dict:
    """Sharpe, profit factor, max drawdown, time-distribution pivot."""

def time_heatmap(trades: list[Trade]) -> pd.DataFrame:
    """pivot_table: index=hour-of-day, columns=day-of-week, values=avg pnl_pct."""
```

Sharpe formula: `mean(daily_pnl) / std(daily_pnl) × √252`

### B3 · Trade Reviewer (`backtest/trade_reviewer.py`)

Interactive Plotly chart for a single trade, identified by `trade_id`.

```python
def review_trade(
    trade_id: str,
    db_path: Path | None = None,
    lookback: int = 50,
    lookahead: int = 20,
    overlays: list = [],     # reserved for future order-flow layers
) -> go.Figure:
    """Fetch trade from DB, build Plotly candlestick figure. Returns Figure."""
```

Overlays rendered:
- Candlestick chart (`go.Candlestick`, LTF bars in window)
- FVG zones — semi-transparent rectangles (bull=green, bear=red)
- Volume profile — horizontal bar chart on right axis (`go.Bar(orientation='h')`)
- Entry marker — triangle annotation on entry bar
- SL line — red dashed horizontal
- TP line — green dashed horizontal
- Exit marker — coloured by result (green=win, red=loss, grey=timeout)

**Performance**:
- WebGL via Plotly — no tkinter, no SVG per-element lag
- Window capped at `lookback + lookahead ≈ 70` candles
- FVG rectangles filtered to window only (typically < 10)

**Usage**:
```python
# Jupyter — renders inline
from backtest.trade_reviewer import review_trade
review_trade("a1b2c3d4-...")

# Save to HTML for offline review
review_trade("a1b2c3d4-...").write_html("reviews/a1b2c3d4.html")
```

### B4 · Strategy Interface (`strategy/base.py`)

Abstract base so future strategies (order-flow, OB-only, etc.) are drop-in replacements.

```python
# strategy/base.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def detect_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add zone columns to df. Pure function — no DB access, no side effects."""

    @abstractmethod
    def generate_signals(self, htf: pd.DataFrame, ltf: pd.DataFrame,
                         config: dict) -> list[dict]:
        """Return list of signal dicts: entry_time, entry_price, sl, tp, direction."""
```

Existing `strategy/smc/` functions remain unchanged; a thin `SMCStrategy(BaseStrategy)`
wrapper can be added later without touching the engine.

---

## 7. Implementation Order

```
A1  bench.py               → establish baseline numbers
A2  engine.py              → vectorised exit; verify identical results on --fast
A3  run.py                 → --random flag; checkpoint key update
B0  logger.py              → queue-based logging; wire into run.py + engine
B1  db.py                  → schema + BacktestDB; write trades from run.py
B2  stats.py               → compute_stats; replace summary_dict where appropriate
B3  trade_reviewer.py      → Plotly viewer
B4  strategy/base.py       → ABC; no behaviour change
```

Do not start B1 before B0 — `BacktestDB` methods need a logger.

---

## 8. Verification

```bash
# A1 — baseline benchmark
uv run python -m backtest.bench

# A2 — correctness (results must match pre-optimisation on same params)
uv run backtest/run.py --fast --no-viz
# diff the CSV against a saved reference

# A2 — speed benchmark (compare before/after)
uv run python -m backtest.bench   # run again after A2

# A3 — random search smoke test
uv run backtest/run.py --random 10 --fast --no-viz

# B0 — logging sanity
uv run backtest/run.py --fast --no-viz
# → backtest/results/<ts>/run.log should contain worker tags

# B1+B2 — DB write check
uv run backtest/run.py --fast --no-viz
uv run python -c "
import duckdb
con = duckdb.connect('db/backtest.duckdb')
print(con.execute('SELECT * FROM runs').df())
print(con.execute('SELECT count(*) FROM trades').fetchone())
"

# B3 — trade reviewer
uv run python -c "
from backtest.trade_reviewer import review_trade
review_trade('<any trade_id from DB>').write_html('review.html')
"
# open review.html in browser
```

---

## 9. Out of Scope (Intentional)

| Item | Reason |
|---|---|
| `src/` directory restructure | Existing layout works; reorganisation adds churn with no functional gain |
| Rewrite `fvg.py` as vectorised | HTF has ~200 bars — Python loop is microseconds; not a bottleneck |
| GPU / cuDF | Requires Linux or WSL2 + CUDA; current data volumes do not warrant it |
| Rust + PyO3 for engine | Only after strategy logic is frozen and Numba is insufficient |
| Replace CSV output with DB-only | CSV kept as-is; DB is additive |

---

## 10. Implementation Status (as of 2026-05-21)

All planned items completed. Key deviations from original spec:

| Item | Original plan | Actual |
|------|--------------|--------|
| `backtest/trade_reviewer.py` (Plotly) | Standalone module | Integrated into `analysis/trade_viewer.py` as Trade Review mode (matplotlib); user requested consolidation |
| `analysis/orderflow.py` | Unchanged | Renamed to `analysis/trade_viewer.py` |
| `run_stats` schema | `sharpe_ratio` column | `sharpe` + `sortino` (two separate columns) |
| A1 benchmark | Measure before optimising | Skipped — A2 implemented directly |

Unit test coverage: **163 tests, 0 failures** (2026-05-21).

---

## References

**Random Search (A3)**

Bergstra, J. & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization.*
Journal of Machine Learning Research, 13, 281–305.
https://jmlr.org/papers/v13/bergstra12a.html

Key result: random search finds equally good or better configurations than grid search
in the same compute budget, because most hyperparameter dimensions are low-importance
and the effective search space is lower-dimensional than it appears. With N=300 random
draws per TF pair, the probability of missing all top-1% configurations (≈ 4 combos
out of 384) is (0.99)^300 ≈ 5%.
