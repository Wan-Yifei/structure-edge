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
import pathlib
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

# Run as a script, so the repo root is not on sys.path -- Python adds the
# SCRIPT's directory, not the working directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from core.market_calendar import (                             # noqa: E402
    CLOSE_FULL_ET, CLOSE_HALF_ET, market_close_et as _calendar_close,
    thanksgiving as _thanksgiving)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from moomoo import AuType, KLType, OpenQuoteContext, RET_OK  # noqa: F401 (AuType/KLType used as string constants via their class attributes)

SHARES_PER_CONTRACT = 100  # US standard

# Time to expiry is measured against the exchange's clock, not the machine's:
# US equity options stop trading 16:00 New York time on the expiry date.
_ET = ZoneInfo("America/New_York")
_CLOSE_FULL_ET = CLOSE_FULL_ET
_CLOSE_HALF_ET = CLOSE_HALF_ET
_SECONDS_PER_YEAR = 365 * 24 * 3600
_MIN_T_YEARS = 3600 / _SECONDS_PER_YEAR      # one hour; see _t_years


def _market_close_et(d: date) -> time:
    """13:00 on a half session, 16:00 otherwise.

    Delegates to core.market_calendar, which is the one place the rules
    live. This function kept its own copy of the three recurring 1pm
    closes, and the duplicate was how the collectors ended up with a
    calendar that knew Black Friday closes early but not that Thanksgiving
    does not open at all.

    Worth 0.22 on Zero Gamma each time a half day is the near expiry (the
    line moves about 7.5 cents per hour of time-to-expiry): assuming 16:00
    for 2026-11-27 put us 0.17 below moomoo's reading, and 13:00 closes
    that to 0.05.

    Still answers for a date that is no session at all, returning the
    full-day close, because an expiry never lands on one. A caller asking
    whether to RUN wants core.market_calendar.is_trading_day -- that is the
    question this signature cannot express.
    """
    return _calendar_close(d)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11111
RISK_FREE_RATE = 0.045  # approx. short-term T-bill; gamma is only weakly sensitive to r
ZERO_GAMMA_GRID_POINTS = 300


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


def get_near_expiry_dates(ctx: OpenQuoteContext, code: str, dte_limit: int) -> list[str]:
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


def coverage_is_full(cov: dict | None) -> bool:
    """Did every requested expiry and every snapshot batch come back?"""
    if not cov:
        return False
    return (len(cov["expiries_ok"]) == len(cov["expiries_requested"])
            and cov["batches_ok"] == cov["batches_total"])


def coverage_note(cov: dict | None) -> str:
    """'2/4 expiries' etc., or '' when nothing is missing.

    Short enough to append to a chart title, a status bar or a stdout line
    without wrapping -- every surface that shows a GEX magnitude has to carry
    it, because the magnitudes are what a partial fetch gets wrong.
    """
    # None means "not reported", which only a synthetic caller produces --
    # fetch_option_data always returns one. Saying nothing is right here:
    # a warning on every hand-built stats dict would train the eye to skip
    # the banner, which is the one thing it must not do.
    if cov is None or coverage_is_full(cov):
        return ""
    bits = []
    ok, want = len(cov["expiries_ok"]), len(cov["expiries_requested"])
    if ok != want:
        bits.append(f"{ok}/{want} expiries")
    if cov["batches_ok"] != cov["batches_total"]:
        bits.append(f"{cov['batches_ok']}/{cov['batches_total']} batches")
    return "PARTIAL: " + ", ".join(bits)


def fetch_option_data(ctx: OpenQuoteContext, code: str,
                      expiries: list[str]) -> tuple[pd.DataFrame, dict]:
    """Fetch option chain and greeks, with a report of what was missed.

    Returns (frame, coverage). Both per-expiry and per-batch failures used to
    print one line to stderr and carry on; in the viewer stderr goes nowhere,
    so a chain that lost half its expiries produced a GEX number 44% smaller
    with the status bar still reading "updated". The walls barely move when
    that happens -- measured: 660/600 before and after -- so nothing on
    screen betrayed it. Hence coverage travels with the data instead of being
    left for the caller to notice.

    expiries_ok is the list, not a count, because build_gex_stats() labels the
    chart with it: fed the REQUESTED list it names expiries whose contracts
    are not in the sum.
    """
    coverage = {"expiries_requested": list(expiries), "expiries_ok": [],
                "batches_ok": 0, "batches_total": 0}
    chain_frames = []
    for expiry in expiries:
        ret, df = ctx.get_option_chain(code, start=expiry, end=expiry)
        if ret != RET_OK:
            print(f"  Warning: chain fetch failed for {expiry}: {df}", file=sys.stderr, flush=True)
            continue
        chain_frames.append(df[["code"]].copy())
        coverage["expiries_ok"].append(expiry)

    if not chain_frames:
        raise RuntimeError("No option chain data returned for any expiry.")

    all_codes = pd.concat(chain_frames, ignore_index=True)["code"].tolist()

    snap_frames = []
    batch = 200
    for i in range(0, len(all_codes), batch):
        coverage["batches_total"] += 1
        ret, snap = ctx.get_market_snapshot(all_codes[i : i + batch])
        if ret != RET_OK:
            print(f"  Warning: snapshot batch {i} failed: {snap}", file=sys.stderr, flush=True)
            continue
        snap_frames.append(snap)
        coverage["batches_ok"] += 1

    if not snap_frames:
        raise RuntimeError("No snapshot data returned.")

    snap = pd.concat(snap_frames, ignore_index=True)

    needed = [
        "code", "option_type", "option_strike_price",
        "option_open_interest", "option_delta", "option_gamma",
        "option_implied_volatility", "strike_time", "volume",
        # Carried so the chart can date itself by the DATA rather than by
        # this machine's clock. Per-contract: a quiet strike's stamp can be
        # hours behind a busy one, so only the newest is quoted.
        "update_time",
    ]
    available = [c for c in needed if c in snap.columns]
    return snap[available].copy(), coverage


# ── GEX computation ────────────────────────────────────────────────────────────

def compute_gex(df: pd.DataFrame, spot: float) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _bs_gamma(S: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray, r: float) -> np.ndarray:
    """Black-Scholes gamma, vectorized over broadcastable arrays."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-0.5 * d1 ** 2) / (np.sqrt(2 * np.pi) * S * sigma * np.sqrt(T))


def zero_gamma(df: pd.DataFrame) -> float | None:
    """Find the hypothetical spot price where total dealer GEX flips sign.

    This mirrors how brokers (e.g. moomoo's own Gamma Exposure chart) define
    "Zero Gamma": reprice every option's gamma via Black-Scholes at a grid of
    hypothetical spot prices (gamma itself changes as spot moves -- it peaks
    ATM), sum signed dealer GEX at each hypothetical spot, and interpolate
    where that curve crosses zero. This is NOT the same as a cumulative sum
    of today's-spot GEX ordered by strike (by_strike["cum_gex"], used for the
    blue cumulative-by-strike line in the chart) -- that simpler quantity
    answers a different question and can land far from the true flip point.
    """
    rows = df[
        (df["option_implied_volatility"] > 0) & (df["option_strike_price"] > 0)
    ].copy()
    if rows.empty:
        return None

    now = datetime.now(_ET)

    def _t_years(strike_time) -> float:
        """Time to expiry in years, from the actual clock.

        This used to be `max((exp - date.today()).days, 0.5) / 365` -- whole
        LOCAL CALENDAR DAYS. Measured consequence: dZG/dT is about -$1.80 per
        day on a real SOXL chain, so the Zero Gamma line sat frozen all day
        and jumped $1.80 at local midnight, with the market shut. It also
        over-counted: against moomoo's own Gamma chart for the same chain and
        the same expiry, ours read 40.80 against their 41.32, and solving for
        T put the difference at 4.705 days vs our 5 -- 7.1 hours of
        over-counting, which accounts for the whole $0.52 with nothing left
        over.

        US equity options stop trading at the close on the expiry date, in
        New York time rather than whatever the machine is set to, and that
        close is 13:00 on a half session -- see _market_close_et, which two
        of the next nine expiries need.
        """
        try:
            exp = date.fromisoformat(str(strike_time)[:10])
        except ValueError:
            return float("nan")
        close = datetime.combine(exp, _market_close_et(exp), tzinfo=_ET)
        years = (close - now).total_seconds() / _SECONDS_PER_YEAR
        # An ALREADY EXPIRED contract has negative time and gets no number:
        # the server does serve expired chains (distance -2 seen live), and
        # clamping one would answer with a confident, plausible-looking
        # level computed from nothing. dropna() below drops it instead.
        if years < 0:
            return float("nan")
        # Inside the last hour the BS assumptions have stopped describing
        # anything -- open interest is being exercised or closed out, not
        # hedged -- and 1/sqrt(T) runs away. Floor there. The floor barely
        # moves zero_gamma (0.20 across 15 min to 4 h, measured, because the
        # root is a ratio) but it does hold down the MAGNITUDES, which run
        # about 13x normal in that last hour.
        return max(years, _MIN_T_YEARS)

    rows["t_years"] = rows["strike_time"].map(_t_years)
    rows = rows.dropna(subset=["t_years"])
    if rows.empty:
        return None

    K     = rows["option_strike_price"].to_numpy()
    T     = rows["t_years"].to_numpy()
    sigma = rows["option_implied_volatility"].to_numpy() / 100.0  # moomoo reports IV in percent
    oi    = rows["option_open_interest"].to_numpy()
    sign  = rows["option_type"].map({"CALL": 1.0, "PUT": -1.0}).fillna(0.0).to_numpy()

    grid = np.linspace(float(K.min()), float(K.max()), ZERO_GAMMA_GRID_POINTS)
    S = grid[:, None]  # (grid_points, 1) broadcasts against (1, n_options)

    gamma = _bs_gamma(S, K[None, :], T[None, :], sigma[None, :], RISK_FREE_RATE)
    total = (sign[None, :] * gamma * oi[None, :] * SHARES_PER_CONTRACT * S).sum(axis=1)

    for i in range(len(grid) - 1):
        y1, y2 = float(total[i]), float(total[i + 1])
        if y1 * y2 <= 0 and y1 != y2:
            x1, x2 = float(grid[i]), float(grid[i + 1])
            return x1 - y1 * (x2 - x1) / (y2 - y1)
    return None


# Hedging demand as a share of average daily volume, and what that is worth.
# GEX on its own is an absolute number: it says how much gamma dealers carry,
# not whether that matters against how much the stock actually trades. The same
# GEX is decisive on a thin name and invisible on a liquid one. Measured on
# SOXL, the strongest strike needs ~0.03% of a day's volume per 1% move, which
# is why its "walls" do not pin anything -- 61M shares a day swamp them.
_HEDGE_BANDS = [
    (0.5,   "negligible"),   # lost in normal flow
    (2.0,   "minor"),        # visible only on quiet days
    (5.0,   "material"),     # shows up in the tape
    (1e9,   "dominant"),     # hedging drives the price
]


def _expiry_label(expiries: list[str], max_listed: int = 3) -> str:
    """Spell out a few expiries, collapse a long run to count plus range.

    Daily-expiry names reach five within a week -- QQQ and SPY both do -- and
    the full list is 68 characters, which overflows a quarter-width stats
    column onto the next one and fills the title bar edge to edge.
    """
    if not expiries:
        return "N/A"
    if len(expiries) <= max_listed:
        return "  ·  ".join(expiries)
    # No word for the count: both call sites already print "Expiries:".
    return f"{expiries[0]} → {expiries[-1]}  ({len(expiries)})"


def _hedge_label(pct_adv: float) -> str:
    if pct_adv != pct_adv:          # NaN -- no volume reference available
        return "n/a"
    for limit, name in _HEDGE_BANDS:
        if pct_adv < limit:
            return name
    return "dominant"


def _data_as_of(df: pd.DataFrame) -> str:
    """When the newest quote in this chain was stamped.

    Falls back to today's date only when the column is absent, and says so,
    because the whole point is that a reader can tell the two apart: on a
    Sunday the two differ by two days, and nothing on the chart used to.
    """
    if "update_time" not in df.columns:
        return f"{date.today().isoformat()} (clock)"
    stamps = pd.to_datetime(df["update_time"], errors="coerce").dropna()
    if stamps.empty:
        return f"{date.today().isoformat()} (clock)"
    return stamps.max().strftime("%Y-%m-%d %H:%M")


def _live_side(by_strike: pd.DataFrame, col: str, want_max: bool) -> pd.DataFrame:
    """The strikes that actually carry GEX on one side.

    A strike whose GEX is exactly zero is not a weak wall -- it is the
    absence of one. Nothing downstream can draw "we looked here and found
    nothing" in a way that differs from the strikes that are never drawn at
    all: the chart fades weaker ranks, so a zero strike arrives looking
    exactly like a genuine wall that happens to rank low. Dropping it is the
    only honest option, so it is dropped here rather than ranked last.
    """
    return by_strike[by_strike[col] > 0] if want_max else by_strike[by_strike[col] < 0]


def _wall_at(by_strike: pd.DataFrame, col: str,
             want_max: bool) -> tuple[float | None, float | None]:
    """(strike, value) of the strongest wall on one side, or (None, None)."""
    live = _live_side(by_strike, col, want_max)
    if live.empty:
        return None, None
    idx = live[col].idxmax() if want_max else live[col].idxmin()
    return (float(live.loc[idx, "option_strike_price"]), float(live.loc[idx, col]))


def build_gex_stats(df: pd.DataFrame, by_strike: pd.DataFrame,
                 spot: float, avg_vol: float, expiries: list[str],
                 coverage: dict | None = None) -> dict:
    itm_calls = df[(df["option_type"] == "CALL") & df["itm"]]
    itm_puts  = df[(df["option_type"] == "PUT")  & df["itm"]]

    call_oi  = int(itm_calls["option_open_interest"].sum())
    put_oi   = int(itm_puts["option_open_interest"].sum())
    call_sh  = call_oi * SHARES_PER_CONTRACT
    put_sh   = put_oi  * SHARES_PER_CONTRACT
    total_sh = call_sh + put_sh
    net_gex  = float(df["gex"].sum())

    # Call Wall: strongest call GEX. Put Wall: most negative put GEX.
    # Both may be None -- see _wall_at() for why a zero strike is dropped
    # rather than ranked last.
    call_wall, call_wall_v = _wall_at(by_strike, "call_gex", want_max=True)
    put_wall,  put_wall_v  = _wall_at(by_strike, "put_gex",  want_max=False)

    def _pct(n: float) -> float:
        return n / avg_vol * 100 if avg_vol else float("nan")

    # Shares dealers must trade per 1% move in spot. "gex" is gamma x OI x 100
    # x spot, which is the delta change for a move of one whole spot, so a
    # percent of it is the delta change for a 1% move. Absolute value: a short
    # call and a short put both oblige a trade, the sign only says which way.
    hedge_per_strike = (
        df.assign(h=df["gex"].abs() * 0.01)
        .groupby("option_strike_price")["h"].sum()
    )
    hedge_total = float(hedge_per_strike.sum())
    if len(hedge_per_strike):
        hedge_peak_strike = float(hedge_per_strike.idxmax())
        hedge_peak = float(hedge_per_strike.max())
    else:
        hedge_peak_strike, hedge_peak = float("nan"), 0.0

    return {
        "spot": spot,
        "avg_vol": avg_vol,
        "expiries": expiries,
        "coverage": coverage,
        # The data's own stamp, not date.today(). Generated on a Sunday, the
        # old field titled a chart of Friday's greeks "Data: 2026-11-22" --
        # and a saved PNG carries that lie for as long as it exists.
        "as_of": _data_as_of(df),
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
        "zero_gamma": zero_gamma(df),
        "call_wall": call_wall,
        "call_wall_v": call_wall_v,
        "put_wall": put_wall,
        "put_wall_v": put_wall_v,
        "hedge_total": hedge_total,
        "hedge_peak": hedge_peak,
        "hedge_peak_strike": hedge_peak_strike,
        "hedge_peak_pct": _pct(hedge_peak),
        "hedge_total_pct": _pct(hedge_total),
        "hedge_label": _hedge_label(_pct(hedge_peak)),
    }


def top_gex_walls(by_strike: pd.DataFrame, n: int,
                  net: bool = False) -> tuple[list[dict], list[dict]]:
    """Return the top *n* Call Wall / Put Wall candidates, strongest first.

    build_gex_stats()'s call_wall/put_wall are just this with n=1 (idxmax/
    idxmin) -- this generalizes to "top N" for callers that want to show
    more than the single strongest strike per side (e.g. a multi-wall
    chart overlay). Each entry is {"strike": float, "gex": float}.

    With net=True both sides rank on net_gex instead, giving the strongest
    POSITIVE net and the most NEGATIVE net strike -- where dealer hedging
    actually bites, after calls and puts at the same strike cancel.

    Returns AT MOST n per side, not exactly n: zero-GEX strikes are excluded
    (see _live_side), so a side with fewer live strikes than asked yields a
    shorter list. That shortfall is itself information -- padding it with
    zeros produced lines indistinguishable from real walls.
    """
    # net=True ranks both sides on net_gex (= call_gex + put_gex), so the
    # pair becomes "strongest positive net" and "most negative net" rather
    # than "most call gamma" and "most put gamma". The two can land on
    # completely different strikes: a strike can hold the day's largest
    # call gamma and almost none of the net, because puts written at the
    # same strike offset it, and the dealer only hedges what is left.
    pos_col = "net_gex" if net else "call_gex"
    neg_col = "net_gex" if net else "put_gex"
    calls = _live_side(by_strike, pos_col, True).nlargest(n, pos_col)
    puts  = _live_side(by_strike, neg_col, False).nsmallest(n, neg_col)
    call_walls = [
        {"strike": float(r["option_strike_price"]), "gex": float(r[pos_col])}
        for _, r in calls.iterrows()
    ]
    put_walls = [
        {"strike": float(r["option_strike_price"]), "gex": float(r[neg_col])}
        for _, r in puts.iterrows()
    ]
    return call_walls, put_walls


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _per_pct(gex):
    """GEX restated as shares to hedge per 1% move in spot.

    The stored `gex` column is gamma x OI x 100 x spot -- the delta change for
    a move of one WHOLE spot, i.e. 100%. Printing that raw next to a
    "% of 20d Avg Vol" column invited a reading 100x too large, so every
    number a reader sees goes through here and the 0.01 lives in exactly one
    place. The stored columns stay in the 100% unit; build_gex_stats() scales
    its own hedge figures independently and must not be routed through this.
    """
    return gex * 0.01


def _wall_str(stats: dict, side: str) -> str:
    """'$660  (3.0K sh/1%)', or 'none' when that side carries no GEX."""
    strike, value = stats[f"{side}_wall"], stats[f"{side}_wall_v"]
    if strike is None:
        return "none"
    return f"${strike:.0f}  ({_fmt(_per_pct(value))} sh/1%)"


def _fmt(n: float) -> str:
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{int(n)}"


# ── Chart ──────────────────────────────────────────────────────────────────────

# Minimum x separation, as a fraction of the plotted strike range, before two
# top-of-chart labels are considered to collide and the later one drops a row.
_LABEL_SEP_PCT = 0.07


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

    fig = plt.figure(figsize=(14, 8.8), facecolor=SURF)
    # Bottom at 0.33 rather than 0.30: the x tick labels plus "Strike Price"
    # need ~3% of figure height below the axes, and at 0.30 the label ran into
    # the separator and was clipped by the stats panel underneath.
    ax  = fig.add_axes([0.07, 0.33, 0.90, 0.58], facecolor=SURF)
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
    ax_r.set_ylabel("Cumulative GEX (sh per 1% move)", color=CUM_COL, fontsize=8)
    ax_r.tick_params(axis="y", colors=CUM_COL, labelsize=7)
    ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt(_per_pct(v))))
    for sp in ax_r.spines.values():
        sp.set_visible(False)

    # ── Key strike lines ───────────────────────────────────────────────────────

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    span = ymax - ymin or 1.0

    # Spot, Zero Gamma and the Call Wall all label the top of the chart, and
    # nothing stopped two of them sharing a row: spot 163.71 against a 165 call
    # wall put the price straight through the wall's box. Each label takes the
    # highest row still clear of every label already placed, where "clear"
    # means more than _LABEL_SEP_PCT of the x range away.
    taken: list[tuple[float, int]] = []      # (x, row)

    def _slot(x: float) -> float:
        min_sep = (xmax - xmin) * _LABEL_SEP_PCT
        row = 0
        while any(r == row and abs(x - px) < min_sep for px, r in taken):
            row += 1
        taken.append((x, row))
        return ymax - span * (0.02 + 0.13 * row)

    # Spot price — yellow dashed
    ax.axvline(stats["spot"], color=SPOT_COL, linewidth=1.4, linestyle="--", zorder=6, alpha=0.9)
    ax.text(stats["spot"] + bar_w * 0.3, _slot(stats["spot"]),
            f"${stats['spot']:.2f}", color=SPOT_COL, fontsize=8,
            va="top", ha="left", zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=SPOT_COL, alpha=0.85, lw=0.8))

    # Zero Gamma (Gamma Flip) — orange solid
    zg = stats.get("zero_gamma")
    if zg is not None:
        ax.axvline(zg, color=ZERO_G_COL, linewidth=1.6, zorder=6)
        ax.text(zg + bar_w * 0.3, _slot(zg),
                f"Zero Gamma\n${zg:.2f}", color=ZERO_G_COL, fontsize=7.5,
                va="top", ha="left", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=ZERO_G_COL, alpha=0.85, lw=0.8))

    # Call Wall — highest call GEX strike. None when no strike on that side
    # carries any GEX; _slot() must stay inside the guard, since it records
    # the slot it hands out and would be fed None.
    cw = stats["call_wall"]
    if cw is not None:
        ax.axvline(cw, color=CALL_COL, linewidth=0.8, linestyle=":", zorder=5, alpha=0.7)
        ax.text(cw, _slot(cw),
                f"Call Wall\n${cw:.0f}", color=CALL_COL, fontsize=7.5,
                va="top", ha="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=CALL_COL, alpha=0.85, lw=0.8))

    # Put Wall — most negative put GEX strike
    pw = stats["put_wall"]
    if pw is not None:
        ax.axvline(pw, color=PUT_COL, linewidth=0.8, linestyle=":", zorder=5, alpha=0.7)
        ax.text(pw, ymin + span * 0.02,
                f"Put Wall\n${pw:.0f}", color=PUT_COL, fontsize=7.5,
                va="bottom", ha="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=PUT_COL, alpha=0.85, lw=0.8))

    # ── Axis styling ───────────────────────────────────────────────────────────
    ax.set_xlim(xmin, xmax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt(_per_pct(v))))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}"))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", zorder=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xlabel("Strike Price", fontsize=9)
    ax.set_ylabel("Call / Put GEX (sh per 1% move)", fontsize=9)

    # ── Title ──────────────────────────────────────────────────────────────────
    ticker  = code.split(".")[-1]
    exp_str = _expiry_label(stats["expiries"])
    dir_str = "▲ Stabilizing" if stats["stable"] else "▼ Destabilizing"
    dir_col = CUM_COL if stats["stable"] else CALL_COL
    ax.set_title(
        f"{ticker}  —  Option GEX  |  Data: {stats['as_of']}  |  Spot: ${stats['spot']:.2f}"
        f"  |  Expiries: {exp_str}",
        color=TEXT_PRI, fontsize=10.5, fontweight="bold", pad=10,
    )
    # A saved PNG outlives any terminal it was printed beside, and a chart
    # missing half its chain is indistinguishable from a complete one by
    # eye -- the walls land on the same strikes. So the warning rides on the
    # image itself, in the title band where it cannot be missed, not in the
    # 7pt italic footnote nobody reads.
    note = coverage_note(stats.get("coverage"))
    if note:
        ax.text(0.5, 1.085, f"⚠  {note} — magnitudes understated",
                transform=ax.transAxes, color=CALL_COL, fontsize=9,
                fontweight="bold", ha="center", va="bottom", zorder=20)

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
    fig.add_artist(plt.Line2D([0.07, 0.98], [0.275, 0.275],
                               transform=fig.transFigure, color=GRID, linewidth=0.8))

    # ── Stats panel ────────────────────────────────────────────────────────────
    # Top stops below the separator, bottom above the footnote's own strip.
    ax2 = fig.add_axes([0.07, 0.085, 0.91, 0.175], facecolor=PAGE)
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
    _txt(COL[0], V3_Y,   f"Net GEX: {_fmt(_per_pct(stats['net_gex']))} sh/1%  {dir_str}",
         color=dir_col, fontsize=9, fontweight="bold")

    # Col 1: Walls
    _txt(COL[1], HEAD_Y, "Key Strikes",   color=TEXT_SEC, fontsize=9, fontweight="bold")
    _txt(COL[1], V1_Y,   f"Call Wall:  {_wall_str(stats, 'call')}",
         color=CALL_COL, fontsize=9)
    _txt(COL[1], V2_Y,   f"Put Wall:   {_wall_str(stats, 'put')}",
         color=PUT_COL,  fontsize=9)
    _txt(COL[1], V3_Y,   f"Expiries: {_expiry_label(stats['expiries'], 2)}",
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

    # Full width, on its own row: this is the line that says whether any of the
    # walls above are worth acting on, and it does not fit a quarter-width
    # column -- squeezed into one it printed over the next column's stats.
    hp, hl = stats["hedge_peak_pct"], stats["hedge_label"]
    hedge_col = {"negligible": TEXT_MUTED, "minor": TEXT_SEC,
                 "material": SPOT_COL, "dominant": CALL_COL}.get(hl, TEXT_MUTED)
    peak_k = stats["hedge_peak_strike"]
    peak_str = f"${peak_k:.0f}" if peak_k == peak_k else "n/a"
    fig.text(0.075, 0.048,
             f"Dealer hedging — strongest strike {peak_str}: "
             f"{_fmt(stats['hedge_peak'])} sh per 1% move"
             + (f"  =  {hp:.2f}% of 20d avg volume  →  {hl.upper()}" if hp == hp else "")
             + f"      ·      all strikes combined: {_fmt(stats['hedge_total'])} sh"
             + (f"  ({stats['hedge_total_pct']:.2f}%)" if hp == hp else ""),
             color=hedge_col, fontsize=8.5, va="bottom", ha="left")

    # On the figure, not inside ax2: sharing the panel meant it sat on the same
    # band as the "% 20d Avg Vol" row and the two printed over each other.
    fig.text(0.98, 0.016,
             "Dealers assumed net short  ·  GEX = γ × OI × 100 × spot × 1%  ·  "
             "Zero Gamma = BS-repriced dealer GEX flips sign vs. hypothetical spot",
             color=TEXT_MUTED, fontsize=7,
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

        expiries = get_near_expiry_dates(ctx, args.code, args.dte)
        if not expiries:
            print(f"  No expiries found within {args.dte} DTE. Try --dte 14 or 30.", flush=True)
            return
        print(f"  Expiries: {expiries}", flush=True)

        df, coverage = fetch_option_data(ctx, args.code, expiries)
        print(f"  Options loaded: {len(df)}", flush=True)
        note = coverage_note(coverage)
        if note:
            # Loud, and before the numbers it invalidates. Not an exception:
            # raising here would leave no chart at all, which reads as a
            # different failure entirely.
            print(f"  *** {note} -- every magnitude below is understated ***",
                  flush=True)

        df, by_strike = compute_gex(df, spot)
        # The FETCHED expiries, not the requested ones: the chart titles
        # itself from this list and would otherwise name expiries whose
        # contracts are not in the sum.
        stats = build_gex_stats(df, by_strike, spot, avg_vol,
                                coverage["expiries_ok"], coverage)

        zg_str = f"${stats['zero_gamma']:.2f}" if stats["zero_gamma"] else "none in range"
        print(f"\n  Net GEX    : {_fmt(_per_pct(stats['net_gex']))} sh/1%  "
              f"{'▲ Stabilizing' if stats['stable'] else '▼ Destabilizing'}", flush=True)
        print(f"  Zero Gamma : {zg_str}  (gamma flip strike)", flush=True)
        print(f"  Call Wall  : {_wall_str(stats, 'call')}", flush=True)
        print(f"  Put Wall   : {_wall_str(stats, 'put')}", flush=True)
        print(f"  ITM Call OI: {stats['call_oi']:,} contracts = {_fmt(stats['call_sh'])} sh"
              f"  ({stats['call_pct']:.1f}% of avg vol)", flush=True)
        print(f"  ITM Put  OI: {stats['put_oi']:,} contracts = {_fmt(stats['put_sh'])} sh"
              f"  ({stats['put_pct']:.1f}% of avg vol)", flush=True)
        print(f"  Hedging    : {_fmt(stats['hedge_peak'])} sh per 1% move at "
              f"${stats['hedge_peak_strike']:.0f} "
              f"({stats['hedge_peak_pct']:.2f}% of avg vol -- {stats['hedge_label']});"
              f"  all strikes {_fmt(stats['hedge_total'])} sh "
              f"({stats['hedge_total_pct']:.2f}%)", flush=True)

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
