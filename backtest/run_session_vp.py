"""Session Value-Area reversal strategy backtest runner (long-only, SOXL).

Leaner counterpart to backtest/run.py: single timeframe (no trend/entry TF
pair — session_vp_engine.py computes both the profile and the reversal
signal off the same klines), single stock code by convention (SOXL can't
be shorted, so this strategy has no reason to sweep multiple symbols), and
no checkpoint/coverage-gap-reuse machinery (v1 scope -- see the approved
plan's "Explicit non-goals").

Usage:
    uv run backtest/run_session_vp.py --config config/backtest/session_vp_smoke.json
    uv run backtest/run_session_vp.py --config <cfg>.json --random 300 --seed 42
    uv run backtest/run_session_vp.py --config <cfg>.json --fast   # tiny smoke test

Two-phase parameter search (matches this project's established workflow):
  1. Random search over a wide range:
       uv run backtest/run_session_vp.py --config session_vp_wide.json --random 300
  2. Distill into a narrowed grid, then re-run in grid mode:
       uv run backtest/aggregate_random_session_vp.py --run-dir <phase-1 dir> \\
           --top-n 30 --out-config session_vp_grid_v1.json
       uv run backtest/run_session_vp.py --config config/backtest/session_vp_grid_v1.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

from tqdm import tqdm

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from feeds.fetcher import fetch_klines
from backtest.session_vp_engine import (
    ALGO_VERSION, SessionVPParams, precompute_session_context, run_backtest_session_vp,
)
from backtest.report import generate_report
from backtest.db import BacktestDB


def _params_hash(params: SessionVPParams) -> str:
    """Hash strategy params for DB run identification -- see backtest/run.py's
    _params_hash() for the identical rationale (distinguishes combos so
    get_or_create_run's lookup doesn't collide two different param sets)."""
    import hashlib
    d = params.to_dict()
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

_RESULTS_DIR    = pathlib.Path(__file__).parent / "results"
_SCHEDULE_PATH  = pathlib.Path(__file__).parent.parent / "config" / "schedule.json"


def _git_commit_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


_COMMIT_HASH = _git_commit_hash()


def _load_schedule_sessions() -> dict:
    with open(_SCHEDULE_PATH, encoding="utf-8") as f:
        return json.load(f)["sessions"]


@dataclass
class RunConfig:
    code:      str       = "US.SOXL"
    entry_tf:  str       = "1m"
    start:     str       = "2025-02-13"
    end:       str       = "2025-12-31"
    workers:   int       = 0    # <=0 = auto (cpu_count - 2, leaving 2 cores free)
    top_n:     int       = 20
    param_grid: dict     = field(default_factory=dict)


def _resolve_workers(w: int) -> int:
    import os
    if w <= 0:
        return max(1, (os.cpu_count() or 4) - 2)  # leave 2 cores free for the system
    return w


def _load_config(path: pathlib.Path) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return RunConfig(
        code       = raw.get("code", "US.SOXL"),
        entry_tf   = raw.get("entry_tf", "1m"),
        start      = raw.get("start", "2025-02-13"),
        end        = raw.get("end", "2025-12-31"),
        workers    = _resolve_workers(raw.get("workers", -1)),
        top_n      = raw.get("top_n", 20),
        param_grid = raw.get("param_grid", {}),
    )


_DEFAULT_GRID: dict = {
    "warmup_minutes":       [30, 45, 60],
    "va_pct":               [0.70],
    "n_bins":               [60],
    "rsi_period":           [6],
    "rsi_threshold":        [20, 25, 30],
    "tradeable_sessions":   [["premarket"], ["regular"], ["afterhours"], ["overnight"]],
    "max_bars":             [120],
    "min_val_poc_dist_pct": [0.001],
}

_FAST_GRID: dict = {
    "warmup_minutes":       [45],
    "va_pct":               [0.70],
    "n_bins":               [60],
    "rsi_period":           [6],
    "rsi_threshold":        [30],
    "tradeable_sessions":   [["regular"]],
    "max_bars":             [120],
    "min_val_poc_dist_pct": [0.001],
}


def build_param_list(grid: dict, entry_tf: str) -> list[SessionVPParams]:
    """Expand a parameter grid into a flat list of SessionVPParams via cartesian product."""
    keys   = list(grid.keys())
    values = list(grid.values())
    return [
        SessionVPParams(entry_tf=entry_tf, **dict(zip(keys, combo)))
        for combo in itertools.product(*values)
    ]


def build_param_list_random(
    grid: dict, entry_tf: str, n_samples: int = 300, seed: int = 42,
) -> list[SessionVPParams]:
    """Random search: sample n_samples combinations (with replacement)."""
    import random as _random
    rng = _random.Random(seed)
    return [
        SessionVPParams(entry_tf=entry_tf, **{k: rng.choice(v) for k, v in grid.items()})
        for _ in range(n_samples)
    ]


_worker_klines: pd.DataFrame | None = None
_worker_schedule: dict | None = None
_worker_context: dict | None = None


def _pool_init(klines: pd.DataFrame, schedule_sessions: dict, session_context: dict) -> None:
    """ProcessPoolExecutor initializer: stash the large, combo-independent
    data (klines, schedule, precomputed session context) as worker-process
    globals ONCE per worker, instead of re-pickling and re-sending them as
    part of every one of thousands of per-combo task arguments."""
    global _worker_klines, _worker_schedule, _worker_context
    _worker_klines = klines
    _worker_schedule = schedule_sessions
    _worker_context = session_context


def _worker(args: tuple) -> tuple[int, "object"]:
    """Execute a single backtest combo inside a worker process.

    Module-level so ProcessPoolExecutor can pickle it on Windows (spawn mode
    does not support closures/lambdas) -- same constraint as backtest/run.py.
    """
    idx, params = args
    result = run_backtest_session_vp(_worker_klines, params, _worker_schedule, _worker_context)
    return idx, result


def _fmt_row(d: dict) -> str:
    return (
        f"  trades={d['n_trades']:3d}  wr={d['win_rate']:.1%}  "
        f"R={d['total_r']:+.1f}  avgR={d['avg_r']:+.3f}  "
        f"PF={d['profit_factor']:.2f}  DD={d['max_drawdown_r']:.2f}  "
        f"maxL={d['max_loss_r']:.2f}"
    )


def run_grid(
    klines: pd.DataFrame,
    params_list: list[SessionVPParams],
    schedule_sessions: dict,
    workers: int,
) -> list:
    """Run all parameter combinations in parallel worker processes."""
    # Computed once here (not inside run_backtest_session_vp per combo) --
    # session_info/occurrence_id/groups depend only on klines+schedule, not
    # on any SessionVPParams field. See precompute_session_context's
    # docstring: skipping this turned a ~45min run into single-digit minutes.
    session_context = precompute_session_context(klines, schedule_sessions)
    tasks = [(idx, p) for idx, p in enumerate(params_list, 1)]
    bt_results: dict[int, object] = {}

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_pool_init,
        initargs=(klines, schedule_sessions, session_context),
    ) as ex:
        futures = {ex.submit(_worker, t): t[0] for t in tasks}
        bar = tqdm(
            total=len(tasks), ncols=90,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for fut in as_completed(futures):
            idx, bt = fut.result()
            bt_results[idx] = bt
            d = bt.summary_dict()
            bar.set_postfix_str(
                f"T={d['n_trades']} WR={d['win_rate']:.0%} "
                f"R={d['total_r']:+.1f} PF={d['profit_factor']:.2f}",
                refresh=False,
            )
            bar.update(1)
        bar.close()

    return [bt_results[i] for i in sorted(bt_results)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Session Value-Area reversal strategy backtest runner.")
    ap.add_argument("--config", type=pathlib.Path, default=None,
                     help="Config JSON (code/entry_tf/start/end/workers/top_n/param_grid)")
    ap.add_argument("--fast", action="store_true", help="Smoke test with a tiny fixed grid")
    ap.add_argument("--force", action="store_true", help="Re-fetch klines even if cached")
    ap.add_argument("--random", type=int, default=0, help="Random search: sample N combos instead of full grid")
    ap.add_argument("--seed", type=int, default=42, help="Random search seed (default 42)")
    ap.add_argument("--min-trades", type=int, default=5, help="Exclude combos with fewer trades from the printed top-N")
    ap.add_argument("--no-report", action="store_true", help="Skip the HTML report")
    ap.add_argument("--no-db", action="store_true", help="Skip writing to backtest.duckdb")
    ap.add_argument("--run-name", type=str, default=None, help="Results subdirectory name (default: timestamp)")
    args = ap.parse_args()

    cfg = _load_config(args.config) if args.config else RunConfig()
    grid = _FAST_GRID if args.fast else (cfg.param_grid or _DEFAULT_GRID)
    schedule_sessions = _load_schedule_sessions()

    if args.random > 0:
        params_list = build_param_list_random(grid, cfg.entry_tf, n_samples=args.random, seed=args.seed)
    else:
        params_list = build_param_list(grid, cfg.entry_tf)

    print(f"\n-- Fetching klines: {cfg.code} {cfg.entry_tf} ------------------------------\n")
    klines = fetch_klines(code=cfg.code, ktype=cfg.entry_tf, start=cfg.start, end=cfg.end, force_refresh=args.force)
    bar_range = f"{klines['time_key'].iloc[0]} ... {klines['time_key'].iloc[-1]}" if len(klines) else "-"
    print(f"  {cfg.entry_tf}: {len(klines)} bars  ({bar_range})")
    if klines.empty:
        print("ERROR: no klines returned for the requested range.")
        sys.exit(1)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M")
    algo_tag = "random" if args.random > 0 else "grid"
    results_dir = _RESULTS_DIR / f"{run_name}_svp_{algo_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n-- Running {len(params_list)} combos ({cfg.workers} workers) -----------------\n")
    print(f"  Output dir: {results_dir}")

    bt_results = run_grid(klines, params_list, schedule_sessions, workers=cfg.workers)
    if not bt_results:
        print("No results.")
        return

    code_slug = cfg.code.replace(".", "_")
    df = pd.DataFrame([r.summary_dict() for r in bt_results])
    df.insert(0, "code", cfg.code)
    csv_path = results_dir / f"results_{code_slug}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} results -> {csv_path}")

    if not args.no_db:
        db = BacktestDB()
        try:
            for p, bt in zip(params_list, bt_results):
                run_id, _ = db.get_or_create_run(
                    config_hash=_params_hash(p),
                    config_json=p.to_dict(),
                    symbol=cfg.code,
                    trend_tf=p.entry_tf,   # single-TF strategy: trend_tf/entry_tf both = entry_tf,
                    entry_tf=p.entry_tf,   # kept equal only to satisfy the shared NOT NULL schema
                    start_date=cfg.start,
                    end_date=cfg.end,
                    algo_version=ALGO_VERSION,
                    commit_hash=_COMMIT_HASH,
                )
                db.mark_running(run_id)
                db.write_trades(run_id, cfg.code, bt.trades)
                db.write_stats(run_id, bt)
                db.mark_done(run_id)
        finally:
            db.close()

    df_ranked = (
        df[df["n_trades"] >= args.min_trades]
        .sort_values(["profit_factor", "total_r"], ascending=[False, False])
        .head(cfg.top_n)
    )
    n_excl = len(df) - len(df[df["n_trades"] >= args.min_trades])
    print(f"\n-- Top {cfg.top_n}  (min_trades>={args.min_trades}, {n_excl} excluded) --\n")
    for _, row in df_ranked.iterrows():
        p = SessionVPParams.from_dict(row.to_dict())
        print(f"  {p.label()}")
        print(_fmt_row(row.to_dict()))
        print()

    if not args.no_report:
        generate_report(
            csv_path, output_path=results_dir / f"report_{code_slug}.html",
            top_n=cfg.top_n,
        )


if __name__ == "__main__":
    main()
