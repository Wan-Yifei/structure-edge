"""Extended statistics for backtest results.

Supplements BacktestResult.summary_dict() with:
  - Sharpe / Sortino ratio (R-series)
  - Parameter importance (variance of group-mean metric per param)
  - Time-of-day and day-of-week breakdown DataFrames
  - Pivot heatmap helpers for report / notebook use
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtest.engine import BacktestResult


# ── Trade-level stats ─────────────────────────────────────────────────────────

def sharpe_ratio(r_series: list[float] | np.ndarray, rf: float = 0.0) -> float:
    """Annualised Sharpe on a per-trade R series (no time-weighting).

    Uses N as the denominator (population std) to avoid division issues on
    small samples. Returns 0.0 if std == 0 or fewer than 2 trades.
    """
    arr = np.asarray(r_series, dtype=float)
    if len(arr) < 2:
        return 0.0
    excess = arr - rf
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(len(arr)))


def sortino_ratio(r_series: list[float] | np.ndarray, rf: float = 0.0) -> float:
    """Sortino ratio using downside deviation (losses only).

    Returns 0.0 if no losing trades or fewer than 2 trades.
    """
    arr = np.asarray(r_series, dtype=float)
    if len(arr) < 2:
        return 0.0
    excess = arr - rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(len(arr)))


def compute_extended_stats(bt: "BacktestResult") -> dict:
    """Return a dict of extended metrics for one BacktestResult.

    Merges into summary_dict() output — call both and combine if needed.
    """
    r_vals = [t.r_multiple for t in bt.trades]
    return {
        "sharpe":  round(sharpe_ratio(r_vals),  3),
        "sortino": round(sortino_ratio(r_vals), 3),
    }


# ── Parameter importance ──────────────────────────────────────────────────────

def parameter_importance(
    df: pd.DataFrame,
    params: list[str],
    metric: str = "profit_factor",
) -> pd.Series:
    """Rank parameters by variance of group-mean metric.

    Higher variance → changing this parameter has a bigger impact on outcomes.

    Args:
        df:      DataFrame with one row per combo, containing param cols + metric col.
        params:  List of parameter column names to analyse.
        metric:  Target metric column.

    Returns:
        pd.Series indexed by param name, sorted descending by importance score.
    """
    scores: dict[str, float] = {}
    for col in params:
        if col not in df.columns:
            continue
        group_means = df.groupby(col)[metric].mean()
        scores[col] = float(group_means.var())
    return pd.Series(scores).sort_values(ascending=False)


# ── Time-of-day / day-of-week breakdown ──────────────────────────────────────

def time_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate R by entry hour and day-of-week.

    Args:
        trades_df: DataFrame with at least 'entry_time' (str/datetime) and
                   'r_multiple' columns (from BacktestDB.get_trades()).

    Returns:
        DataFrame with columns: hour, day_of_week, n_trades, avg_r, win_rate.
    """
    df = trades_df.copy()
    df["entry_dt"] = pd.to_datetime(df["entry_time"])
    df["hour"]     = df["entry_dt"].dt.hour
    df["dow"]      = df["entry_dt"].dt.dayofweek  # 0=Mon … 6=Sun

    grouped = df.groupby(["hour", "dow"]).agg(
        n_trades =("r_multiple", "count"),
        avg_r    =("r_multiple", "mean"),
        win_rate =("r_multiple", lambda x: (x > 0).mean()),
    ).reset_index()
    grouped["avg_r"]    = grouped["avg_r"].round(3)
    grouped["win_rate"] = grouped["win_rate"].round(3)
    return grouped


def time_heatmap_pivot(
    trades_df: pd.DataFrame,
    metric: str = "avg_r",
) -> pd.DataFrame:
    """Return a pivot table (rows=hour, cols=day_of_week) for heatmap plotting.

    Day-of-week columns are labelled Mon…Fri (Sat/Sun included if data exists).
    """
    _DOW = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    breakdown = time_breakdown(trades_df)
    pivot = breakdown.pivot(index="hour", columns="dow", values=metric)
    pivot = pivot.rename(columns=_DOW)
    pivot.index.name   = "Hour"
    pivot.columns.name = "Day"
    return pivot


# ── Grid-level heatmap helpers ────────────────────────────────────────────────

def param_heatmap_pivot(
    df: pd.DataFrame,
    row_param: str,
    col_param: str,
    metric: str = "profit_factor",
    agg: str = "mean",
) -> pd.DataFrame:
    """Pivot mean(metric) for two parameters — ready for px.imshow() or sns.heatmap().

    Example:
        pivot = param_heatmap_pivot(df, 'fvg_entry_depth_pct', 'min_rr')
        px.imshow(pivot, title='Depth × RR — mean profit_factor')
    """
    return df.pivot_table(
        index=row_param,
        columns=col_param,
        values=metric,
        aggfunc=agg,
    )


def top_param_pairs(
    df: pd.DataFrame,
    params: list[str],
    metric: str = "profit_factor",
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Return the top-k (param_a, param_b) pairs ranked by interaction effect.

    Interaction effect = std of cell means in the 2D pivot minus the sum of
    individual marginal variances (rough measure of whether the pair has
    synergistic explanatory power beyond the individual params).

    Returns list of (param_a, param_b, score) sorted by score descending.
    """
    import itertools
    results = []
    for a, b in itertools.combinations(params, 2):
        if a not in df.columns or b not in df.columns:
            continue
        try:
            pivot = param_heatmap_pivot(df, a, b, metric)
            score = float(pivot.stack().var())
            results.append((a, b, score))
        except Exception:
            continue
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]
