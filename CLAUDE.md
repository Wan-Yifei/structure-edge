# CLAUDE.md — Project conventions for Claude Code

## Environment

- **Python runtime**: always use `uv run <script>` — never activate the venv manually
- **OS**: Windows 10, PowerShell; use PowerShell syntax in shell commands
- **Git user**: Wan-Yifei — never append `Co-Authored-By: Claude …` to commits

## Code style

- All code **comments and docstrings must be in English** (no Chinese in source files)
- Commit messages may be in English or Chinese

---

## Versioning — two independent schemes

### 1. System version `vX.Y.Z` — application / tooling

Stored in `VERSION` (root) and git tag `vX.Y.Z`.  Covers the whole codebase:
viewer, scheduler, backtest tools, infrastructure, scripts.

Follows standard semver semantics, with `X=0` reserved for internal / dev builds:

| Change type | Bump | Examples |
|-------------|------|---------|
| Formal release to `main` / major milestone | **major** `X.Y.Z → (X+1).0.0` | First production release |
| New feature or new subsystem (backward-compatible) | **minor** `X.Y.Z → X.(Y+1).0` | New viewer, new overlay, new DB schema |
| Bug fix — no new behaviour | **patch** `X.Y.Z → X.Y.(Z+1)` | FVG overlay fix, crash fix |

`X=0` means internal / pre-release; `X≥1` is reserved for the first merge to `main`
that is considered production-ready.

Update `VERSION` file and tag HEAD with `vX.Y.Z`.  Document in `CHANGELOG.md`.

### 2. Algo version `smc_vX.Y.Z` — backtest algorithm

Derived automatically from the most recent `smc_v*` git tag via
`git describe --tags --match smc_v*`.  Embedded in every trade ID in the DB.

| Change type | Bump | Rule |
|-------------|------|------|
| New algorithm / entry logic / structural strategy change | **minor** `smc_vX.Y` | Backtest results change |
| Bug fix that changes backtest results (engine / strategy) | **patch** `smc_vX.Y.Z` | Same params → different trades |
| Pure tooling / viewer fix (no effect on backtest output) | — | No algo tag; only bump system version |

### Rules (both schemes)

1. **Tags are immutable once pushed** and DB records exist for them.
   Exception: if a tag never produced valid records (e.g. a bug caused
   0 trades throughout), moving it once is acceptable — document in CHANGELOG.

2. **`smc_v` minor bump checklist** — update all of:
   - `CHANGELOG.md`, `strategy/smc/STRATEGY.md`
   - `doc/smc_v2_strategy.md` (version timeline table)
   - Create `doc/smc_vX.Y_strategy.md` (dedicated change note)

### Example sequence

```
v0.3.0        — system: PyQtGraph viewer release (new feature → minor)
smc_v2.3      — algo:   determine_trend veto + BOS scan fix
v0.3.1        — system: viewer overlay fixes + compare_versions tool (bug fix → patch)
smc_v2.3.1    — algo:   detect_fvg default regression fix
v0.4.0        — system: order flow overlays — liquidity HM, iceberg, spoof (new feature → minor)
smc_v2.4      — algo:   next algorithmic change
v1.0.0        — system: first formal release merged to main
```

---

## Backtest workflow

- Grid / random search configs live in `config/backtest/`
- Results land in `backtest/results/<timestamp>_<algo>_<run_name>/`
- Compare two algo versions: `uv run backtest/compare_versions.py --csv … --config …`
- Inspect rejection reasons: `uv run backtest/fvg_inspect.py --from-csv … --inspect-start … --inspect-end …`

## Key files

| File | Purpose |
|------|---------|
| `backtest/engine.py` | Core backtest engine + `BacktestParams` + `ALGO_VERSION` |
| `strategy/smc/market_structure.py` | BOS/CHoCH detection, `determine_trend` |
| `strategy/smc/fvg.py` | FVG detection (`require_displacement` defaults `False`) |
| `strategy/smc/order_blocks.py` | Order block detection |
| `feeds/fetcher.py` | Kline fetcher with DuckDB cache |
| `analysis/trade_viewer_qt.py` | PyQtGraph chart viewer (current) |
| `analysis/trade_viewer.py` | Matplotlib viewer (legacy) |
| `analysis/scheduler.py` | Target manager / scheduler UI |
