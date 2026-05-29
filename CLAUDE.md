# CLAUDE.md — Project conventions for Claude Code

## Environment

- **Python runtime**: always use `uv run <script>` — never activate the venv manually
- **OS**: Windows 10, PowerShell; use PowerShell syntax in shell commands
- **Git user**: Wan-Yifei — never append `Co-Authored-By: Claude …` to commits

## Code style

- All code **comments and docstrings must be in English** (no Chinese in source files)
- Commit messages may be in English or Chinese

---

## Versioning — `smc_v*` git tags

`ALGO_VERSION` in `backtest/engine.py` is derived from the most recent
`smc_v*` git tag via `git describe --tags --match smc_v*`.  It is embedded in
every trade ID stored in the database, so tag discipline matters.

### Tag types

| Change type | Tag | Rule |
|-------------|-----|------|
| New algorithm / new entry logic / structural strategy change | `smc_vX.Y` | Minor bump |
| Bug fix that changes backtest results (engine / strategy code) | `smc_vX.Y.Z` | Patch bump |
| Pure tooling / viewer / script fix (no effect on backtest output) | — | No tag; commit only |

### Rules

1. **Tags are immutable once pushed** — never move a tag that has produced DB records.
   Exception: if the tag was placed before any valid backtest records existed (e.g.
   a bug caused 0 trades throughout the tag's lifetime), moving it is acceptable as
   a one-time cleanup; document it in CHANGELOG.

2. **Patch tags (`X.Y.Z`) are for engine/strategy bug fixes** — use them when the
   fix would change trade outcomes (entry bars, SL/TP, filters) for the same params
   and data.  Purely additive infrastructure fixes (new CLI flag, new output column)
   that don't alter existing results do not need a patch tag.

3. **Minor tags (`X.Y`) mark intentional algorithmic changes** — any time the
   strategy logic, signal detection, or entry rules change in a way that makes new
   results incompatible with old results, bump the minor version and update:
   - `CHANGELOG.md` (what changed, why)
   - `strategy/smc/STRATEGY.md` (parameter table, method description)
   - `doc/smc_v2_strategy.md` (version timeline table)
   - Create `doc/smc_vX.Y_strategy.md` (dedicated change note)

### Example sequence

```
smc_v2.3      — algorithmic release (determine_trend veto + BOS scan fix)
smc_v2.3.1    — engine bug fix (detect_fvg default regression)
smc_v2.4      — next algorithmic change
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
