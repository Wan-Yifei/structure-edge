# File Index

Quick reference for locating important documentation and configuration files.
Use this when modifying strategy logic to ensure all downstream files are updated.

---

## Strategy & Algorithm

| File | Scope |
|------|-------|
| `strategy/smc/STRATEGY.md` | **Current** strategy pipeline (all steps), full parameter reference. **Update first** when adding/changing any algo parameter or filter step. |
| `strategy/smc/market_structure.py` | BOS/CHoCH detection, `find_swings`, `determine_trend` |
| `strategy/smc/fvg.py` | FVG detection (`detect_fvg`, `fvg_entry_depth`) |
| `strategy/smc/kd_trend.py` | KD channel trend detector (`compute_kd`, `kd_trend`) |
| `strategy/smc/order_blocks.py` | Order block detection |
| `strategy/smc/confirmation.py` | LTF CHoCH+BOS confirmation (`check_ltf_confirmation`) |

---

## Backtest Engine

| File | Scope |
|------|-------|
| `backtest/engine.py` | Core engine loop, `BacktestParams` dataclass, `ALGO_VERSION`, `run_backtest`. **Primary file for algo changes.** |
| `backtest/run.py` | Multi-process grid/random runner, checkpoint/resume, DuckDB persistence |
| `backtest/fvg_inspect.py` | FVG rejection log inspector — `_OUTCOME_META`, CLI args, `_build_params_from_args`. Update when adding new rejection reasons or `BacktestParams` fields. |
| `backtest/audit.py` | Per-trade audit HTML generator |
| `backtest/report.py` | Grid-results HTML report |
| `backtest/export_csv_from_db.py` | Reconstruct per-stock CSVs from DuckDB (use when HPC CSVs are lost) |
| `backtest/merge_db.py` | Merge a secondary DuckDB into the master DB |
| `backtest/post_backtest.sh` | Chains report → audit → fvg_inspect after every run |

---

## Version History & Docs

| File | Scope |
|------|-------|
| `CHANGELOG.md` | System version changelog (`vX.Y.Z` releases). Update on every version bump. |
| `VERSION` | Single-line system version (`vX.Y.Z`). Kept in sync with `CHANGELOG.md`. |
| `doc/smc_v2_strategy.md` | Algo version timeline (`smc_v2` … current), pipeline diffs, quick-reference table. Update when adding a new `smc_v*` tag. |
| `doc/smc_v2.3_strategy.md` | Change note for smc_v2.3 |
| `doc/smc_v2.4_strategy.md` | Change note for smc_v2.4 |
| `doc/smc_v2.5_strategy.md` | Change note for smc_v2.5 |
| `doc/BACKTEST_SYSTEM_GUIDE.md` | End-to-end backtest workflow guide |
| `doc/ORDER_FLOW_GUIDE.md` | Order flow collector & heatmap guide |

---

## Configuration

| File | Scope |
|------|-------|
| `config/chart.json` | Chart viewer defaults (session filter, BOS span, TF-specific gap settings) |
| `config/backtest/cross_stock_grid_v3.json` | Current cross-stock grid config (CSCO/AMD/NVDA/QCOM, 3 TF pairs) |
| `config/backtest/soxl_grid_v2.json` | SOXL-focused grid config (long-only, 30m/3m) |
| `config/backtest/default_smc_v2.json` | Reference param set for single-run usage |
| `config/backtest/focused_smc_v2.json` | Focused grid derived from analysis |

> **When adding a new `BacktestParams` field:** update `config/backtest/cross_stock_grid_v3.json`
> and `config/backtest/soxl_grid_v2.json` with the new param in `param_grid`.

---

## Infrastructure

| File | Scope |
|------|-------|
| `docker/entrypoint.sh` | HPC container entrypoint: S3 download → backtest → post-processing → S3 upload |
| `docker/backtest.env.example` | Example env vars for Docker runs (including `ALGO_VERSION`) |
| `feeds/fetcher.py` | Kline fetcher with DuckDB cache |
| `db/backtest.duckdb` | Master backtest database (runs, run_stats, trades) |

---

## Checklist: Adding a new `BacktestParams` field

1. `backtest/engine.py` — add field to `BacktestParams`, use in engine loop, update `label()`, update `to_dict` / `from_dict` if needed
2. `backtest/fvg_inspect.py` — add CLI arg, update `_build_params_from_args`, add entry to `_OUTCOME_META` if new rejection reason
3. `strategy/smc/STRATEGY.md` — document in pipeline steps and parameter reference table
4. `doc/smc_v2_strategy.md` — add row to version timeline table and quick-reference table
5. `doc/smc_vX.Y_strategy.md` — create new change note file
6. `config/backtest/cross_stock_grid_v3.json` — add to `param_grid`
7. `config/backtest/soxl_grid_v2.json` — add to `param_grid`
8. `tests/backtest/test_engine.py` — add unit tests
9. `VERSION` + `CHANGELOG.md` — bump system version
10. `docker/backtest.env.example` — update `ALGO_VERSION` to new tag
11. Create git tag `smc_vX.Y`
