"""Backtest grid results report generator.

Generates a self-contained HTML report from a backtest results CSV.
Charts are rendered via Plotly (interactive, WebGL-accelerated).

Usage:
    from backtest.report import generate_report
    generate_report("backtest/results/20260521_1200/results_US_SNDK.csv")

    # or from CLI:
    uv run python -m backtest.report backtest/results/20260521_1200/results_US_SNDK.csv
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colour palette (matches viz.py dark theme) ────────────────────────────────
_BG      = "#0b1120"
_BG2     = "#131f30"
_FG      = "#cdd6f4"
_GRID    = "#1e2d42"
_GREEN   = "#26a69a"
_RED     = "#ef5350"
_GOLD    = "#f9a825"
_BLUE    = "#5c9cf5"
_PURPLE  = "#ce93d8"
_CYAN    = "#80cbc4"
_ORANGE  = "#ff8a65"
_ACCENT  = [_GREEN, _BLUE, _GOLD, _PURPLE, _CYAN, _ORANGE, "#a5d6a7", "#ef9a9a"]

_PARAM_COLS = [
    "swing_lookback", "bos_count", "fvg_min_width_pct",
    "fvg_entry_depth_pct", "require_ltf_confirmation",
    "displacement_required", "sl_buffer_pct", "max_sl_pct", "min_rr",
]
_METRIC_COLS = ["n_trades", "win_rate", "total_r", "avg_r",
                "profit_factor", "max_drawdown_r", "max_loss_r", "sharpe", "sortino"]

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG2,
    font=dict(color=_FG, size=11),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=_GRID, linecolor=_GRID, zerolinecolor=_GRID),
    yaxis=dict(gridcolor=_GRID, linecolor=_GRID, zerolinecolor=_GRID),
)


def _apply_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    layout = dict(_PLOTLY_LAYOUT)
    layout.update(kwargs)
    if title:
        layout["title"] = dict(text=title, font=dict(size=13, color=_FG))
    fig.update_layout(**layout)
    return fig


def _to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


# ── 1. KPI cards ──────────────────────────────────────────────────────────────

_MIN_TRADES_REPORT = 10   # minimum trades for a combo to appear in ranking sections

def _kpi_section(df: pd.DataFrame) -> str:
    active = df[df["n_trades"] >= _MIN_TRADES_REPORT]
    if active.empty:
        active = df[df["n_trades"] > 0]
    # Cap inf PF so idxmax works correctly
    pf = active["profit_factor"].replace(float("inf"), -1)   # exclude inf from "best"
    best = active.loc[pf.idxmax()] if not active.empty else None

    def card(label: str, value: str, colour: str = _FG) -> str:
        return (
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{colour}">{value}</div>'
            f'</div>'
        )

    n_pos = (df["total_r"] > 0).sum()
    pct_pos = n_pos / len(df) * 100 if len(df) else 0

    cards = [
        card("Total combos", f"{len(df):,}"),
        card("With trades", f"{len(active):,}"),
        card("Profitable", f"{n_pos:,} ({pct_pos:.0f}%)",
             _GREEN if pct_pos > 25 else _RED),
    ]
    if best is not None:
        cards += [
            card("Best total R", f"{best['total_r']:.2f}R", _GREEN),
            card("Best PF", f"{best['profit_factor']:.2f}", _GREEN),
            card("Best win rate", f"{best['win_rate']:.1%}",
                 _GREEN if best["win_rate"] > 0.5 else _GOLD),
            card("Trades (best)", f"{int(best['n_trades'])}"),
        ]
        if "sharpe" in best.index:
            sh = best["sharpe"]
            so = best["sortino"]
            cards += [
                card("Best Sharpe", f"{sh:.2f}",
                     _GREEN if sh > 1.0 else (_GOLD if sh > 0.5 else _FG)),
                card("Best Sortino", f"{so:.2f}" if so != float("inf") else "∞",
                     _GREEN if so > 1.5 else (_GOLD if so > 0.75 else _FG)),
            ]

    return (
        '<div class="kpi-row">'
        + "".join(cards)
        + "</div>"
    )


# ── 2. Top N table ────────────────────────────────────────────────────────────

_INIT_CAPITAL = 10_000.0

def _top_table(df: pd.DataFrame, top_n: int = 20, min_trades: int = 10) -> str:
    active = df[df["n_trades"] >= min_trades].copy()
    if active.empty:
        active = df[df["n_trades"] > 0].copy()   # fallback if nothing passes threshold
    # Replace inf PF with a finite cap so nlargest is stable
    active = active.copy()
    active["profit_factor"] = active["profit_factor"].replace(
        [float("inf")], active["profit_factor"][active["profit_factor"] != float("inf")].max() * 1.01
        if (active["profit_factor"] != float("inf")).any() else 999.0
    )
    top = active.nlargest(top_n, "profit_factor")
    cols = ["trend_tf", "entry_tf", "n_trades", "win_rate", "total_r", "final_value",
            "avg_r", "profit_factor", "max_drawdown_r", "sharpe", "sortino",
            "swing_lookback", "bos_count", "fvg_min_width_pct",
            "fvg_entry_depth_pct", "require_ltf_confirmation",
            "sl_buffer_pct", "max_sl_pct", "min_rr"]
    cols = [c for c in cols if c in top.columns]
    top = top[cols].reset_index(drop=True)

    def fmt(val, col: str) -> str:
        if col == "win_rate":
            colour = _GREEN if val > 0.5 else (_GOLD if val > 0.35 else _RED)
            return f'<span style="color:{colour}">{val:.1%}</span>'
        if col == "profit_factor":
            colour = _GREEN if val >= 1.5 else (_GOLD if val >= 1.0 else _RED)
            return f'<span style="color:{colour}">{val:.2f}</span>'
        if col == "total_r":
            colour = _GREEN if val > 0 else _RED
            return f'<span style="color:{colour}">{val:.2f}</span>'
        if col == "final_value":
            colour = _GREEN if val > _INIT_CAPITAL else _RED
            gain_pct = (val / _INIT_CAPITAL - 1) * 100
            sign = "+" if gain_pct >= 0 else ""
            return (f'<span style="color:{colour}">'
                    f'${val:,.0f} ({sign}{gain_pct:.1f}%)</span>')
        if col == "sharpe":
            colour = _GREEN if val > 1.0 else (_GOLD if val > 0.5 else (_FG if val >= 0 else _RED))
            return f'<span style="color:{colour}">{val:.2f}</span>'
        if col == "sortino":
            if val == float("inf") or val > 999:
                return f'<span style="color:{_GREEN}">∞</span>'
            colour = _GREEN if val > 1.5 else (_GOLD if val > 0.75 else (_FG if val >= 0 else _RED))
            return f'<span style="color:{colour}">{val:.2f}</span>'
        if isinstance(val, float):
            return f"{val:.3g}"
        if isinstance(val, bool):
            return "✓" if val else "✗"
        return str(val)

    header = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for _, row in top.iterrows():
        cells = "".join(f"<td>{fmt(row[c], c)}</td>" for c in cols)
        rows += f"<tr>{cells}</tr>"

    note = (f'<p style="color:{_FG};opacity:0.55;font-size:11px;margin:4px 0 0 4px">'
            f'final_value: $10,000 initial capital, 1% compounding risk per trade.</p>')
    return (
        '<div class="table-wrap"><table class="result-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table>{note}</div>"
    )


# ── 3. Equity curves (top N by profit factor) ─────────────────────────────────

def _equity_fig(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Cumulative R bar chart (no trade-level data available from CSV)."""
    active = df[df["n_trades"] >= _MIN_TRADES_REPORT].copy()
    if active.empty:
        active = df[df["n_trades"] > 0].copy()
    active["profit_factor"] = active["profit_factor"].replace(float("inf"), -1)
    top = active.nlargest(top_n, "profit_factor").reset_index(drop=True)

    fig = go.Figure()
    for i, row in top.iterrows():
        label = (
            f"{row['trend_tf']}/{row['entry_tf']} "
            f"PF={row['profit_factor']:.2f} "
            f"WR={row['win_rate']:.0%} "
            f"T={int(row['n_trades'])}"
        )
        colour = _ACCENT[i % len(_ACCENT)]
        fig.add_trace(go.Bar(
            name=label,
            x=[label],
            y=[row["total_r"]],
            marker_color=colour,
            text=f"{row['total_r']:.2f}R",
            textposition="outside",
        ))

    fig.add_hline(y=0, line_color=_GRID, line_width=1)
    _apply_layout(fig, f"Top {top_n} Combos — Total R",
                  showlegend=False, barmode="group",
                  yaxis_title="Total R", height=380)
    return fig


# ── 4. R-multiple distribution ────────────────────────────────────────────────

def _r_dist_fig(df: pd.DataFrame) -> go.Figure:
    """Distribution of avg_r across all combos."""
    active = df[df["n_trades"] > 0]
    if active.empty:
        return go.Figure()

    fig = go.Figure()
    wins  = active[active["avg_r"] >= 0]["avg_r"]
    loses = active[active["avg_r"] <  0]["avg_r"]

    bins = dict(start=active["avg_r"].min() - 0.05,
                end=active["avg_r"].max() + 0.05, size=0.1)

    if len(wins):
        fig.add_trace(go.Histogram(
            x=wins, xbins=bins, name="avg_r ≥ 0",
            marker_color=_GREEN, opacity=0.75,
        ))
    if len(loses):
        fig.add_trace(go.Histogram(
            x=loses, xbins=bins, name="avg_r < 0",
            marker_color=_RED, opacity=0.75,
        ))

    fig.add_vline(x=0, line_color=_FG, line_width=1, line_dash="dash")
    fig.add_vline(x=active["avg_r"].mean(), line_color=_GOLD,
                  line_width=1.5, line_dash="dot",
                  annotation_text=f"mean {active['avg_r'].mean():.2f}R",
                  annotation_font_color=_GOLD)

    _apply_layout(fig, "Avg R Distribution (all combos)",
                  barmode="overlay", xaxis_title="Avg R per combo",
                  yaxis_title="Count", height=340)
    return fig


# ── 5. Win rate × Profit factor scatter ───────────────────────────────────────

def _scatter_fig(df: pd.DataFrame) -> go.Figure:
    active = df[df["n_trades"] > 0].copy()
    if active.empty:
        return go.Figure()

    active["tf_pair"] = active["trend_tf"] + "/" + active["entry_tf"]
    active["size"] = np.clip(active["n_trades"] * 5, 8, 80)

    fig = go.Figure()
    for i, (pair, grp) in enumerate(active.groupby("tf_pair")):
        fig.add_trace(go.Scatter(
            x=grp["win_rate"],
            y=grp["profit_factor"],
            mode="markers",
            name=str(pair),
            marker=dict(
                size=grp["size"],
                color=_ACCENT[i % len(_ACCENT)],
                opacity=0.65,
                line=dict(width=0.3, color=_BG),
            ),
            text=[
                f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} T={int(r['n_trades'])}"
                for _, r in grp.iterrows()
            ],
            hoverinfo="text+name",
        ))

    fig.add_hline(y=1.0, line_color=_GRID, line_width=1, line_dash="dash")
    _apply_layout(fig, "Win Rate × Profit Factor (all combos)",
                  xaxis_title="Win Rate", yaxis_title="Profit Factor",
                  xaxis_tickformat=".0%", height=400)
    return fig


# ── 6. 2D Sensitivity heatmap ─────────────────────────────────────────────────

def _pick_axes(df: pd.DataFrame) -> tuple[str, str]:
    """Pick the two param columns with highest group-mean variance."""
    candidates = [c for c in _PARAM_COLS if c in df.columns and df[c].nunique() >= 2]
    variances = {}
    for col in candidates:
        gm = df.groupby(col)["profit_factor"].mean()
        variances[col] = float(gm.var())
    top2 = sorted(variances, key=variances.get, reverse=True)[:2]  # type: ignore[arg-type]
    return (top2[0], top2[1]) if len(top2) >= 2 else (candidates[0], candidates[1])


def _heatmap_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    row_col, col_col = _pick_axes(df)
    pivot = df.pivot_table(
        values="profit_factor",
        index=row_col, columns=col_col,
        aggfunc="mean",
    )

    z = pivot.values
    z_text = [[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(v) for v in pivot.columns],
        y=[str(v) for v in pivot.index],
        text=z_text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale="RdYlGn",
        zmin=max(0, float(np.nanmin(z))),
        zmax=float(np.nanmax(z)),
        colorbar=dict(title="Mean PF", tickfont=dict(color=_FG)),
    ))

    _apply_layout(
        fig,
        f"Sensitivity: {row_col} × {col_col}  (mean profit factor)",
        xaxis_title=col_col, yaxis_title=row_col, height=380,
    )
    return fig


# ── 7. 3D surface — two params vs metric ──────────────────────────────────────

def _surface_fig(df: pd.DataFrame, metric: str = "total_r") -> go.Figure:
    """3D scatter/surface of best two params vs chosen metric."""
    if df.empty:
        return go.Figure()

    x_col, y_col = _pick_axes(df)
    active = df[df["n_trades"] > 0].copy()
    if active.empty:
        return go.Figure()

    grp = active.groupby([x_col, y_col])[metric].mean().reset_index()

    fig = go.Figure(go.Scatter3d(
        x=grp[x_col].astype(str),
        y=grp[y_col].astype(str),
        z=grp[metric],
        mode="markers",
        marker=dict(
            size=6,
            color=grp[metric],
            colorscale="RdYlGn",
            colorbar=dict(title=metric, thickness=12,
                          tickfont=dict(color=_FG)),
            opacity=0.85,
        ),
        text=[
            f"{x_col}={r[x_col]}<br>{y_col}={r[y_col]}<br>{metric}={r[metric]:.2f}"
            for _, r in grp.iterrows()
        ],
        hoverinfo="text",
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=f"3D: {x_col} × {y_col} → {metric}", font=dict(size=13, color=_FG)),
        scene=dict(
            xaxis=dict(title=x_col, backgroundcolor=_BG2, gridcolor=_GRID,
                       tickfont=dict(color=_FG)),
            yaxis=dict(title=y_col, backgroundcolor=_BG2, gridcolor=_GRID,
                       tickfont=dict(color=_FG)),
            zaxis=dict(title=metric, backgroundcolor=_BG2, gridcolor=_GRID,
                       tickfont=dict(color=_FG)),
            bgcolor=_BG2,
        ),
        height=480,
    )
    return fig


# ── 8. Parameter importance ───────────────────────────────────────────────────

def _importance_fig(df: pd.DataFrame) -> go.Figure:
    """Variance of group-mean profit_factor for each parameter (higher = more impact)."""
    candidates = [c for c in _PARAM_COLS if c in df.columns and df[c].nunique() >= 2]
    records = []
    for col in candidates:
        gm = df.groupby(col)["profit_factor"].mean()
        records.append({"param": col, "variance": float(gm.var()),
                        "range": float(gm.max() - gm.min())})

    imp = pd.DataFrame(records).sort_values("range", ascending=True)
    colours = [_GREEN if v > imp["range"].median() else _BLUE for v in imp["range"]]

    fig = go.Figure(go.Bar(
        x=imp["range"],
        y=imp["param"],
        orientation="h",
        marker_color=colours,
        text=[f"{v:.3f}" for v in imp["range"]],
        textposition="outside",
    ))

    _apply_layout(fig, "Parameter Impact (PF range across param values)",
                  xaxis_title="Mean PF range (max − min)", height=340)
    return fig


# ── 9. Long / short breakdown ─────────────────────────────────────────────────

def _direction_fig(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Bull vs bear win-rate and total-R for the top N combos by profit factor."""
    need = {"bull_trades", "bear_trades", "bull_win_rate", "bear_win_rate",
            "bull_total_r", "bear_total_r"}
    if not need.issubset(df.columns):
        return go.Figure()

    active = df[df["n_trades"] >= _MIN_TRADES_REPORT].copy()
    if active.empty:
        active = df[df["n_trades"] > 0].copy()
    if active.empty:
        return go.Figure()

    active["profit_factor"] = active["profit_factor"].replace(float("inf"), -1)
    top = active.nlargest(top_n, "profit_factor").reset_index(drop=True)
    labels = [
        f"#{i+1} {r['trend_tf']}/{r['entry_tf']} PF={r['profit_factor']:.2f} T={int(r['n_trades'])}"
        for i, (_, r) in enumerate(top.iterrows())
    ]

    fig = go.Figure()

    # Win-rate bars
    fig.add_trace(go.Bar(
        name="Bull win rate", x=labels, y=top["bull_win_rate"],
        marker_color=_GREEN, opacity=0.85,
        yaxis="y", offsetgroup="bull",
        text=[f"{v:.0%}" for v in top["bull_win_rate"]], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Bear win rate", x=labels, y=top["bear_win_rate"],
        marker_color=_RED, opacity=0.85,
        yaxis="y", offsetgroup="bear",
        text=[f"{v:.0%}" for v in top["bear_win_rate"]], textposition="outside",
    ))

    # Total-R dots on secondary axis
    fig.add_trace(go.Scatter(
        name="Bull total R", x=labels, y=top["bull_total_r"],
        mode="markers", marker=dict(symbol="circle", size=9, color=_GREEN,
                                    line=dict(width=1, color=_BG)),
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        name="Bear total R", x=labels, y=top["bear_total_r"],
        mode="markers", marker=dict(symbol="diamond", size=9, color=_RED,
                                    line=dict(width=1, color=_BG)),
        yaxis="y2",
    ))

    layout = dict(_PLOTLY_LAYOUT)
    layout.update(dict(
        title=dict(text=f"Long vs Short — Top {top_n} by PF",
                   font=dict(size=13, color=_FG)),
        barmode="group",
        height=420,
        yaxis=dict(title="Win Rate", tickformat=".0%",
                   gridcolor=_GRID, linecolor=_GRID, range=[0, 0.9]),
        yaxis2=dict(title="Total R", overlaying="y", side="right",
                    gridcolor=_GRID, zeroline=True, zerolinecolor=_GRID),
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
    ))
    fig.update_layout(**layout)
    return fig


# ── 10. Parallel coordinates ──────────────────────────────────────────────────

def _parcoords_fig(df: pd.DataFrame, metric: str = "total_r") -> go.Figure:
    """Multi-factor view: all numeric params × metric, coloured by metric."""
    active = df[df["n_trades"] > 0].copy()
    if active.empty:
        return go.Figure()

    num_params = [
        c for c in _PARAM_COLS
        if c in active.columns and pd.api.types.is_numeric_dtype(active[c])
        and active[c].nunique() >= 2
    ]
    cols = num_params + [metric]

    # encode booleans as 0/1 for parcoords
    plot_df = active[cols].copy()
    for c in plot_df.columns:
        if plot_df[c].dtype == bool:
            plot_df[c] = plot_df[c].astype(int)

    dimensions = [
        dict(label=c, values=plot_df[c])
        for c in cols
    ]

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=plot_df[metric],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title=metric, tickfont=dict(color=_FG)),
        ),
        dimensions=dimensions,
        labelangle=-30,
        labelside="bottom",
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=f"Parallel Coordinates — coloured by {metric}",
                   font=dict(size=13, color=_FG)),
        height=460,
    )
    return fig


# ── HTML template ─────────────────────────────────────────────────────────────

_CSS = """
<style>
  body { background: #0b1120; color: #cdd6f4; font-family: 'Segoe UI', sans-serif;
         margin: 0; padding: 20px 30px; }
  h1   { color: #5c9cf5; font-size: 1.4rem; margin-bottom: 4px; }
  h2   { color: #cdd6f4; font-size: 1.05rem; margin: 28px 0 10px;
         border-bottom: 1px solid #1e2d42; padding-bottom: 4px; }
  .meta { color: #6e7a9b; font-size: 0.85rem; margin-bottom: 20px; }
  .kpi-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 14px 0 24px; }
  .kpi-card { background: #131f30; border: 1px solid #1e2d42; border-radius: 8px;
              padding: 12px 18px; min-width: 120px; }
  .kpi-label { font-size: 0.75rem; color: #6e7a9b; text-transform: uppercase;
               letter-spacing: 0.04em; }
  .kpi-value { font-size: 1.35rem; font-weight: 600; margin-top: 4px; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .chart-full { margin-bottom: 16px; }
  .table-wrap { overflow-x: auto; margin-bottom: 20px; }
  .result-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  .result-table th { background: #131f30; color: #6e7a9b; padding: 7px 10px;
                     text-align: left; font-weight: 500; white-space: nowrap;
                     border-bottom: 1px solid #1e2d42; }
  .result-table td { padding: 5px 10px; border-bottom: 1px solid #131f30;
                     white-space: nowrap; }
  .result-table tr:hover td { background: #131f30; }
  .plotly-graph-div { border-radius: 8px; overflow: hidden; }
</style>
"""


def _html_section(title: str, content: str) -> str:
    return f"<h2>{title}</h2>\n{content}\n"


def _chart_grid(*html_parts: str) -> str:
    cells = "".join(f'<div class="chart-cell">{p}</div>' for p in html_parts)
    return f'<div class="chart-grid">{cells}</div>'


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(
    csv_path: str | pathlib.Path,
    output_path: Optional[str | pathlib.Path] = None,
    top_n: int = 20,
    metric: str = "total_r",
    open_browser: bool = False,
) -> pathlib.Path:
    """Generate a self-contained HTML audit report from a backtest results CSV.

    Args:
        csv_path:     Path to results CSV produced by run.py.
        output_path:  Where to save the HTML (default: same dir as CSV).
        top_n:        Number of combos shown in top-N table and equity chart.
        metric:       Primary metric for 3D surface and parallel coords.
        open_browser: If True, open the report in the default browser.

    Returns:
        Path to the generated HTML file.
    """
    csv_path = pathlib.Path(csv_path)
    if output_path is None:
        output_path = csv_path.parent / (csv_path.stem + "_report.html")
    output_path = pathlib.Path(output_path)

    df = pd.read_csv(csv_path)

    # ── Build charts ──────────────────────────────────────────────────────────
    kpi_html    = _kpi_section(df)
    table_html  = _top_table(df, top_n)
    equity_html = _to_html(_equity_fig(df, top_n))
    rdist_html  = _to_html(_r_dist_fig(df))
    scatter_html = _to_html(_scatter_fig(df))
    heatmap_html = _to_html(_heatmap_fig(df))
    surface_html = _to_html(_surface_fig(df, metric))
    import_html    = _to_html(_importance_fig(df))
    direction_html = _to_html(_direction_fig(df, top_n))
    parcoords_html = _to_html(_parcoords_fig(df, metric))

    # ── Assemble ──────────────────────────────────────────────────────────────
    title = f"SMC Backtest Report — {csv_path.parent.name} / {csv_path.stem}"
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta  = (f"<div class='meta'>Generated {ts} &nbsp;|&nbsp; "
             f"{len(df):,} combos &nbsp;|&nbsp; source: {csv_path.name}</div>")

    body = (
        meta
        + kpi_html
        + _html_section(f"Top {top_n} Combinations", table_html)
        + _html_section("Performance Overview", _chart_grid(equity_html, rdist_html))
        + _html_section("Long vs Short Breakdown",
                        f'<div class="chart-full">{direction_html}</div>')
        + _html_section("All Combos: Win Rate × Profit Factor",
                        f'<div class="chart-full">{scatter_html}</div>')
        + _html_section("Parameter Sensitivity",
                        _chart_grid(heatmap_html, import_html))
        + _html_section(f"3D: Parameter Interaction → {metric}",
                        f'<div class="chart-full">{surface_html}</div>')
        + _html_section("Parallel Coordinates — Multi-factor View",
                        f'<div class="chart-full">{parcoords_html}</div>')
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{_CSS}
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved → {output_path}")

    if open_browser:
        import webbrowser
        webbrowser.open(output_path.resolve().as_uri())

    return output_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m backtest.report <results.csv> [--open]")
        sys.exit(1)

    csv = sys.argv[1]
    open_b = "--open" in sys.argv
    generate_report(csv, open_browser=open_b)
