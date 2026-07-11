#!/usr/bin/env python3
"""Near-expiry option GEX (Gamma Exposure) analyzer for market-maker activity assessment.

Usage:
    uv run strategy/option/gex.py --code US.SOXL --dte 7
    uv run strategy/option/gex.py --code US.NVDA --dte 14 --out output/

    --out <dir>   save to <dir>/SOXL_20260705_gex_dte7.png  (auto filename)
    --out <file>  save to exactly that path (must end in .png or .svg)

GEX interpretation:
    Net GEX > 0  ->  Call-dominated: dealers net long gamma  ->  stabilizing (sell rips, buy dips)
    Net GEX < 0  ->  Put-dominated:  dealers net short gamma ->  destabilizing (amplifies moves)

ITM (In The Money):
    Call ITM: strike < spot  (deep ITM delta ~1, gamma low; ATM gamma is highest)
    Put  ITM: strike > spot  (deep ITM delta ~-1, gamma low)
"""

import argparse
import os
import sys
from datetime import date, timedelta

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from moomoo import AuType, KLType, OpenQuoteContext, RET_OK  # noqa: F401 (AuType/KLType used as string constants via their class attributes)

SHARES_PER_CONTRACT = 100  # US standard
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11111


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Near-expiry option GEX analyzer")
    p.add_argument("--code", required=True, help="Underlying, e.g. US.SOXL")
    p.add_argument("--dte", type=int, default=7, help="Max days-to-expiry (default 7)")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--out", default=None,
        help="Save chart: directory → auto filename (TICKER_DATE_gex_dteN.png); "
             "file path with extension → save exactly there; omit → show window",
    )
    return p.parse_args()


# ── Data fetching ──────────────────────────────────────────────────────────────

def _get_spot_and_avg_vol(ctx: OpenQuoteContext, code: str, days: int = 20) -> tuple[float, float]:
    """Return (spot_price, avg_daily_volume) from recent daily klines."""
    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days * 2)).isoformat()  # extra buffer for non-trading days
    ret, data, _ = ctx.request_history_kline(
        code, start=start_date, end=end_date,
        ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=days * 2,
    )
    if ret != RET_OK:
        raise RuntimeError(f"Failed to fetch klines for {code}: {data}")
    data = data.tail(days)
    spot = float(data["close"].iloc[-1])
    avg_vol = float(data["volume"].mean())
    return spot, avg_vol


def _get_near_expiry_dates(ctx: OpenQuoteContext, code: str, dte_limit: int) -> list[str]:
    """Return list of expiry date strings (YYYY-MM-DD) within dte_limit calendar days."""
    ret, data = ctx.get_option_expiration_date(code)
    if ret != RET_OK:
        raise RuntimeError(f"Failed to fetch expiry dates for {code}: {data}")
    results = []
    for _, row in data.iterrows():
        dist = row.get("option_expiry_date_distance", None)
        strike_time = str(row["strike_time"])[:10]
        if dist is not None:
            try:
                if 0 <= int(dist) <= dte_limit:
                    results.append(strike_time)
                continue
            except (ValueError, TypeError):
                pass
        # Fallback: parse date directly
        try:
            exp = date.fromisoformat(strike_time)
            if date.today() <= exp <= date.today() + timedelta(days=dte_limit):
                results.append(strike_time)
        except ValueError:
            pass
    return sorted(set(results))


def _fetch_option_data(ctx: OpenQuoteContext, code: str, expiries: list[str]) -> pd.DataFrame:
    """Fetch option chain and greeks for the given expiry dates via market snapshot."""
    chain_frames = []
    for expiry in expiries:
        ret, df = ctx.get_option_chain(code, start=expiry, end=expiry)
        if ret != RET_OK:
            print(f"  Warning: chain fetch failed for {expiry}: {df}", file=sys.stderr, flush=True)
            continue
        chain_frames.append(df[["code"]].copy())

    if not chain_frames:
        raise RuntimeError("No option chain data returned for any expiry.")

    all_codes = pd.concat(chain_frames, ignore_index=True)["code"].tolist()

    snap_frames = []
    batch = 200
    for i in range(0, len(all_codes), batch):
        ret, snap = ctx.get_market_snapshot(all_codes[i : i + batch])
        if ret != RET_OK:
            print(f"  Warning: snapshot batch {i} failed: {snap}", file=sys.stderr, flush=True)
            continue
        snap_frames.append(snap)

    if not snap_frames:
        raise RuntimeError("No snapshot data returned.")

    snap = pd.concat(snap_frames, ignore_index=True)

    needed = [
        "code", "option_type", "option_strike_price",
        "option_open_interest", "option_delta", "option_gamma",
        "option_implied_volatility", "strike_time",
    ]
    available = [c for c in needed if c in snap.columns]
    return snap[available].copy()


# ── GEX computation ────────────────────────────────────────────────────────────

def _compute(df: pd.DataFrame, spot: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-row GEX, ITM flag; return enriched df and per-strike aggregate.

    by_strike columns: option_strike_price, call_gex, put_gex, net_gex, cum_gex
    """
    df = df.copy()
    for col in ("option_gamma", "option_delta", "option_open_interest", "option_strike_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Dealers assumed net short options: call GEX positive, put GEX negative
    sign = df["option_type"].map({"CALL": 1.0, "PUT": -1.0}).fillna(0.0)
    df["gex"] = sign * df["option_gamma"] * df["option_open_interest"] * SHARES_PER_CONTRACT * spot
    df["equiv_shares"] = df["option_open_interest"] * SHARES_PER_CONTRACT

    df["itm"] = (
        ((df["option_type"] == "CALL") & (df["option_strike_price"] < spot)) |
        ((df["option_type"] == "PUT")  & (df["option_strike_price"] > spot))
    )

    call_agg = (
        df[df["option_type"] == "CALL"]
        .groupby("option_strike_price", as_index=False)
        .agg(call_gex=("gex", "sum"))
    )
    put_agg = (
        df[df["option_type"] == "PUT"]
        .groupby("option_strike_price", as_index=False)
        .agg(put_gex=("gex", "sum"))
    )
    by_strike = (
        call_agg.merge(put_agg, on="option_strike_price", how="outer")
        .fillna(0.0)
        .sort_values("option_strike_price")
        .reset_index(drop=True)
    )
    by_strike["net_gex"] = by_strike["call_gex"] + by_strike["put_gex"]
    by_strike["cum_gex"] = by_strike["net_gex"].cumsum()

    return df, by_strike


def _zero_gamma(by_strike: pd.DataFrame) -> float | None:
    """Interpolate the strike where cumulative GEX crosses zero (gamma flip point)."""
    strikes = by_strike["option_strike_price"].values
    cum     = by_strike["cum_gex"].values
    for i in range(len(strikes) - 1):
        y1, y2 = float(cum[i]), float(cum[i + 1])
        if y1 * y2 <= 0 and y1 != y2:
            x1, x2 = float(strikes[i]), float(strikes[i + 1])
            return x1 - y1 * (x2 - x1) / (y2 - y1)
    return None


def _build_stats(df: pd.DataFrame, by_strike: pd.DataFrame,
                 spot: float, avg_vol: float, expiries: list[str]) -> dict:
    itm_calls = df[(df["option_type"] == "CALL") & df["itm"]]
    itm_puts  = df[(df["option_type"] == "PUT")  & df["itm"]]

    call_oi  = int(itm_calls["option_open_interest"].sum())
    put_oi   = int(itm_puts["option_open_interest"].sum())
    call_sh  = call_oi * SHARES_PER_CONTRACT
    put_sh   = put_oi  * SHARES_PER_CONTRACT
    total_sh = call_sh + put_sh
    net_gex  = float(df["gex"].sum())

    # Call Wall: strike with highest call GEX
    cw_idx      = by_strike["call_gex"].idxmax()
    call_wall   = float(by_strike.loc[cw_idx, "option_strike_price"])
    call_wall_v = float(by_strike.loc[cw_idx, "call_gex"])

    # Put Wall: strike with most negative put GEX
    pw_idx     = by_strike["put_gex"].idxmin()
    put_wall   = float(by_strike.loc[pw_idx, "option_strike_price"])
    put_wall_v = float(by_strike.loc[pw_idx, "put_gex"])

    def _pct(n: float) -> float:
        return n / avg_vol * 100 if avg_vol else float("nan")

    return {
        "spot": spot,
        "avg_vol": avg_vol,
        "expiries": expiries,
        "as_of": date.today().isoformat(),
        "call_oi": call_oi,
        "put_oi": put_oi,
        "call_sh": call_sh,
        "put_sh": put_sh,
        "total_sh": total_sh,
        "call_pct": _pct(call_sh),
        "put_pct": _pct(put_sh),
        "total_pct": _pct(total_sh),
        "net_gex": net_gex,
        "stable": net_gex >= 0,
        "zero_gamma": _zero_gamma(by_strike),
        "call_wall": call_wall,
        "call_wall_v": call_wall_v,
        "put_wall": put_wall,
        "put_wall_v": put_wall_v,
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt(n: float) -> str:
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n)}"


# ── Chart ──────────────────────────────────────────────────────────────────────

def _plot(code: str, by_strike: pd.DataFrame, stats: dict, out_path: str | None) -> None:
    SURF       = "#1a1a19"
    PAGE       = "#0d0d0d"
    CALL_COL   = "#e34948"   # red  — calls (moomoo convention)
    PUT_COL    = "#1baf7a"   # green — puts
    CUM_COL    = "#3987e5"   # blue  — cumulative GEX curve
    SPOT_COL   = "#eda100"   # yellow dashed — current price
    ZERO_G_COL = "#ff8c00"   # orange solid  — gamma flip strike
    TEXT_PRI   = "#ffffff"
    TEXT_SEC   = "#c3c2b7"
    TEXT_MUTED = "#898781"
    GRID       = "#2c2c2a"

    plt.rcParams.update({
        "figure.facecolor": SURF,
        "axes.facecolor":   SURF,
        "text.color":       TEXT_PRI,
        "axes.labelcolor":  TEXT_SEC,
        "xtick.color":      TEXT_MUTED,
        "ytick.color":      TEXT_MUTED,
        "axes.edgecolor":   GRID,
        "grid.color":       GRID,
        "grid.linewidth":   0.5,
        "font.family":      "sans-serif",
        "font.size":        9,
    })

    fig = plt.figure(figsize=(14, 8), facecolor=SURF)
    ax  = fig.add_axes([0.07, 0.30, 0.90, 0.61], facecolor=SURF)
    ax_r = ax.twinx()   # secondary y-axis for cumulative GEX curve

    strikes   = by_strike["option_strike_price"].values
    call_gex  = by_strike["call_gex"].values
    put_gex   = by_strike["put_gex"].values
    cum_gex   = by_strike["cum_gex"].values

    n     = len(strikes)
    bar_w = (float(strikes[-1]) - float(strikes[0])) / max(n, 1) * 0.72 if n > 1 else 1.0

    # Separate Call (red, positive) and Put (green, negative) bars
    ax.bar(strikes, call_gex, width=bar_w, color=CALL_COL, edgecolor=SURF, linewidth=0.4, zorder=3, label="Call GEX (dealers long gamma)")
    ax.bar(strikes, put_gex,  width=bar_w, color=PUT_COL,  edgecolor=SURF, linewidth=0.4, zorder=3, label="Put GEX  (dealers short gamma)")

    # Zero baseline
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.8, zorder=2)

    # Cumulative GEX curve on right axis
    ax_r.plot(strikes, cum_gex, color=CUM_COL, linewidth=1.8, zorder=5, label="Cumulative GEX")
    ax_r.axhline(0, color=CUM_COL, linewidth=0.4, linestyle=":", alpha=0.4, zorder=2)
    ax_r.set_ylabel("Cumulative GEX (shares)", color=CUM_COL, fontsize=8)
    ax_r.tick_params(axis="y", colors=CUM_COL, labelsize=7)
    ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt(v)))
    for sp in ax_r.spines.values():
        sp.set_visible(False)

    # ── Key strike lines ───────────────────────────────────────────────────────

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    span = ymax - ymin or 1.0

    # Spot price — yellow dashed
    ax.axvline(stats["spot"], color=SPOT_COL, linewidth=1.4, linestyle="--", zorder=6, alpha=0.9)
    ax.text(stats["spot"] + bar_w * 0.3, ymax - span * 0.02,
            f"  ${stats['spot']:.2f}", color=SPOT_COL, fontsize=8, va="top", ha="left", zorder=7)

    # Zero Gamma (Gamma Flip) — orange solid
    zg = stats.get("zero_gamma")
    if zg is not None:
        ax.axvline(zg, color=ZERO_G_COL, linewidth=1.6, zorder=6)
        ax.text(zg + bar_w * 0.3, ymax - span * 0.12,
                f"Zero Gamma\n${zg:.2f}", color=ZERO_G_COL, fontsize=7.5,
                va="top", ha="left", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=ZERO_G_COL, alpha=0.85, lw=0.8))

    # Call Wall — highest call GEX strike
    cw = stats["call_wall"]
    ax.axvline(cw, color=CALL_COL, linewidth=0.8, linestyle=":", zorder=5, alpha=0.7)
    ax.text(cw, ymax - span * 0.02,
            f"Call Wall\n${cw:.0f}", color=CALL_COL, fontsize=7.5,
            va="top", ha="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=CALL_COL, alpha=0.85, lw=0.8))

    # Put Wall — most negative put GEX strike
    pw = stats["put_wall"]
    ax.axvline(pw, color=PUT_COL, linewidth=0.8, linestyle=":", zorder=5, alpha=0.7)
    ax.text(pw, ymin + span * 0.02,
            f"Put Wall\n${pw:.0f}", color=PUT_COL, fontsize=7.5,
            va="bottom", ha="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=PUT_COL, alpha=0.85, lw=0.8))

    # ── Axis styling ───────────────────────────────────────────────────────────
    ax.set_xlim(xmin, xmax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt(v)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}"))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", zorder=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xlabel("Strike Price", fontsize=9)
    ax.set_ylabel("Call / Put GEX (shares)", fontsize=9)

    # ── Title ──────────────────────────────────────────────────────────────────
    ticker  = code.split(".")[-1]
    exp_str = "  ·  ".join(stats["expiries"]) or "N/A"
    dir_str = "▲ Stabilizing" if stats["stable"] else "▼ Destabilizing"
    dir_col = CUM_COL if stats["stable"] else CALL_COL
    ax.set_title(
        f"{ticker}  —  Option GEX  |  Data: {stats['as_of']}  |  Spot: ${stats['spot']:.2f}"
        f"  |  Expiries: {exp_str}",
        color=TEXT_PRI, fontsize=10.5, fontweight="bold", pad=10,
    )

    # ── Legend ─────────────────────────────────────────────────────────────────
    import matplotlib.lines as mlines
    legend_handles = [
        mpatches.Patch(color=CALL_COL,   label="Call GEX  (dealers long gamma)"),
        mpatches.Patch(color=PUT_COL,    label="Put GEX   (dealers short gamma)"),
        mlines.Line2D([],[], color=CUM_COL,    linewidth=1.8, label="Cumulative GEX"),
        mlines.Line2D([],[], color=SPOT_COL,   linewidth=1.4, linestyle="--", label="Spot price"),
        mlines.Line2D([],[], color=ZERO_G_COL, linewidth=1.6, label="Zero Gamma (flip point)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7.5,
              facecolor="#252521", edgecolor=GRID, labelcolor=TEXT_SEC,
              framealpha=0.92, borderpad=0.7)

    # ── Separator ──────────────────────────────────────────────────────────────
    fig.add_artist(plt.Line2D([0.07, 0.98], [0.285, 0.285],
                               transform=fig.transFigure, color=GRID, linewidth=0.8))

    # ── Stats panel ────────────────────────────────────────────────────────────
    ax2 = fig.add_axes([0.07, 0.02, 0.91, 0.24], facecolor=PAGE)
    ax2.axis("off")

    def _txt(x, y, s, **kw):
        ax2.text(x, y, s, transform=ax2.transAxes, va="top", **kw)

    COL = [0.01, 0.26, 0.52, 0.76]
    HEAD_Y, V1_Y, V2_Y, V3_Y = 0.90, 0.62, 0.37, 0.12

    # Col 0: overview
    zg_str = f"${zg:.2f}" if zg is not None else "none in range"
    _txt(COL[0], HEAD_Y, "GEX Overview",  color=TEXT_SEC, fontsize=9, fontweight="bold")
    _txt(COL[0], V1_Y,   f"Data: {stats['as_of']}  |  Spot: ${stats['spot']:.2f}", color=TEXT_SEC, fontsize=9)
    _txt(COL[0], V2_Y,   f"Zero Gamma: {zg_str}   20d Avg Vol: {_fmt(stats['avg_vol'])} sh",
         color=TEXT_SEC, fontsize=9)
    _txt(COL[0], V3_Y,   f"Net GEX: {_fmt(stats['net_gex'])} sh  {dir_str}",
         color=dir_col, fontsize=9, fontweight="bold")

    # Col 1: Walls
    _txt(COL[1], HEAD_Y, "Key Strikes",   color=TEXT_SEC, fontsize=9, fontweight="bold")
    _txt(COL[1], V1_Y,   f"Call Wall:  ${stats['call_wall']:.0f}  ({_fmt(stats['call_wall_v'])} sh)",
         color=CALL_COL, fontsize=9)
    _txt(COL[1], V2_Y,   f"Put Wall:   ${stats['put_wall']:.0f}  ({_fmt(stats['put_wall_v'])} sh)",
         color=PUT_COL,  fontsize=9)
    _txt(COL[1], V3_Y,   f"Expiries: {', '.join(stats['expiries']) or 'N/A'}",
         color=TEXT_MUTED, fontsize=8)

    # Col 2: ITM Calls
    _txt(COL[2], HEAD_Y, "ITM Call Options", color=CALL_COL, fontsize=9, fontweight="bold")
    _txt(COL[2], V1_Y,   f"OI: {stats['call_oi']:,} contracts",       color=TEXT_SEC, fontsize=9)
    _txt(COL[2], V2_Y,   f"Equiv Shares: {_fmt(stats['call_sh'])}",   color=TEXT_SEC, fontsize=9)
    _txt(COL[2], V3_Y,   f"% 20d Avg Vol: {stats['call_pct']:.1f}%",  color=TEXT_SEC, fontsize=9)

    # Col 3: ITM Puts
    _txt(COL[3], HEAD_Y, "ITM Put Options",  color=PUT_COL,  fontsize=9, fontweight="bold")
    _txt(COL[3], V1_Y,   f"OI: {stats['put_oi']:,} contracts",       color=TEXT_SEC, fontsize=9)
    _txt(COL[3], V2_Y,   f"Equiv Shares: {_fmt(stats['put_sh'])}",   color=TEXT_SEC, fontsize=9)
    _txt(COL[3], V3_Y,   f"% 20d Avg Vol: {stats['put_pct']:.1f}%",  color=TEXT_SEC, fontsize=9)

    ax2.text(0.99, 0.02,
             "Dealers assumed net short  ·  GEX = γ × OI × 100 × spot  ·  Zero Gamma = cumulative GEX flip point",
             color=TEXT_MUTED, fontsize=7, transform=ax2.transAxes,
             va="bottom", ha="right", style="italic")

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURF)
        print(f"Chart saved: {out_path}", flush=True)
    else:
        plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    ctx = OpenQuoteContext(host=args.host, port=args.port)
    try:
        print(f"Fetching data for {args.code}  (DTE ≤ {args.dte}) ...", flush=True)

        spot, avg_vol = _get_spot_and_avg_vol(ctx, args.code)
        print(f"  Spot: ${spot:.2f}   20d avg vol: {avg_vol:,.0f} sh", flush=True)

        expiries = _get_near_expiry_dates(ctx, args.code, args.dte)
        if not expiries:
            print(f"  No expiries found within {args.dte} DTE. Try --dte 14 or 30.", flush=True)
            return
        print(f"  Expiries: {expiries}", flush=True)

        df = _fetch_option_data(ctx, args.code, expiries)
        print(f"  Options loaded: {len(df)}", flush=True)

        df, by_strike = _compute(df, spot)
        stats = _build_stats(df, by_strike, spot, avg_vol, expiries)

        zg_str = f"${stats['zero_gamma']:.2f}" if stats["zero_gamma"] else "none in range"
        print(f"\n  Net GEX    : {_fmt(stats['net_gex'])} sh  "
              f"{'▲ Stabilizing' if stats['stable'] else '▼ Destabilizing'}", flush=True)
        print(f"  Zero Gamma : {zg_str}  (gamma flip strike)", flush=True)
        print(f"  Call Wall  : ${stats['call_wall']:.0f}  GEX {_fmt(stats['call_wall_v'])} sh", flush=True)
        print(f"  Put Wall   : ${stats['put_wall']:.0f}  GEX {_fmt(stats['put_wall_v'])} sh", flush=True)
        print(f"  ITM Call OI: {stats['call_oi']:,} contracts = {_fmt(stats['call_sh'])} sh"
              f"  ({stats['call_pct']:.1f}% of avg vol)", flush=True)
        print(f"  ITM Put  OI: {stats['put_oi']:,} contracts = {_fmt(stats['put_sh'])} sh"
              f"  ({stats['put_pct']:.1f}% of avg vol)", flush=True)

        out_path = _resolve_out(args.out, args.code, args.dte)
        _plot(args.code, by_strike, stats, out_path)
    finally:
        ctx.close()


def _resolve_out(out_arg: str | None, code: str, dte: int) -> str | None:
    """Return resolved save path, or None to show window instead."""
    if out_arg is None:
        return None
    ticker = code.split(".")[-1]
    today  = date.today().strftime("%Y%m%d")
    auto_name = f"{ticker}_{today}_gex_dte{dte}.png"
    # Directory path (exists or ends with separator)
    if os.path.isdir(out_arg) or out_arg.endswith(("/", "\\")):
        os.makedirs(out_arg, exist_ok=True)
        return os.path.join(out_arg, auto_name)
    # Explicit file path with extension
    if os.path.splitext(out_arg)[1]:
        os.makedirs(os.path.dirname(out_arg) or ".", exist_ok=True)
        return out_arg
    # No extension: treat as directory
    os.makedirs(out_arg, exist_ok=True)
    return os.path.join(out_arg, auto_name)


if __name__ == "__main__":
    main()
