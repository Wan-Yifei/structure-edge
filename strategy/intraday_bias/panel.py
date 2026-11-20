"""Build the day x hour panel, and nothing else.

THE ONE THING THAT MATTERS HERE
-------------------------------
``time_key`` is the END of the bar. CHANGELOG v0.92.0 is a forward leak
that came from forgetting it, and the first draft of this study's own
coverage table got it wrong a second time ("QQQ has 20:00 on 99.3% of
days" -- that stamp is the last minute of the 19:00-20:00 block). So an
hour block ``h`` is the bars stamped in ``(HH:00, HH+1:00]``: the bar
stamped 10:00 covers 09:59-10:00 and belongs to block 9.

Entry is the first bar's OPEN, exit is the last bar's CLOSE, which is a
strict 60-minute hold containing no information from outside the block.

SESSION LABELLING
-----------------
A block stamped at or after 20:00 belongs to the NEXT trading day, not
the calendar day it is stamped on. SOXL's overnight session opens Sunday
20:00; without this rule the panel grows 104 phantom "days" (570 instead
of 466) and every rolling window and intraday de-meaning group is
misaligned.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No stop, no target, no position sizing. The rule under test has no stop,
so R is undefined and inventing a denominator would add a free parameter
and a second forward-looking input (ATR). Returns are LOG returns in bps.
"""
from __future__ import annotations

import pathlib

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
KLINES_DB = ROOT / "db" / "backtest_klines.duckdb"

# -- locked panel conventions (PLAN section 1) ---------------------------
RTH_BARS = 390              # a full regular session, asserted not assumed
HALF_DAY_VOL_RATIO = 0.20   # same cut as backtest/orb.py, same sessions
NEXT_SESSION_HOUR = 20      # blocks stamped (20:00, 24:00] -> next day
MIN_BARS_PER_BLOCK = 30     # a "60-minute hold" that is 3 bars is a lie
SPAN_SLACK_MIN = 10         # how far past its own length a block may run


def min_bars(block_min: int = 60) -> int:
    """Half the block, so the rule is never priced off a fragment.

    A study that TRADES a block should demand the whole thing rather
    than this floor; this is the bound below which a cell is not an
    observation at all.
    """
    return max(2, block_min // 2)
WINSOR_FRAC = 0.005         # 0.5% each side, per hour, declared in advance
BAD_PRINT_LOG = 0.05        # bar-to-bar move flagged as a possible misprint


# -- loading -------------------------------------------------------------

def load_1m(code: str, db: pathlib.Path | None = None) -> pd.DataFrame:
    """Every 1-minute bar for `code`, all sessions, indexed on bar END."""
    con = duckdb.connect(str(db or KLINES_DB), read_only=True)
    try:
        k = con.execute(
            "SELECT time_key, open, high, low, close, volume FROM klines "
            "WHERE code=? AND ktype='1m' ORDER BY time_key", [code]).fetchdf()
    finally:
        con.close()
    k["t"] = pd.to_datetime(k.time_key)
    k = k.drop_duplicates("t").set_index("t").sort_index()
    return k.drop(columns=["time_key"])


def tradeable_days(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Trading dates with a complete, full-length regular session.

    Identical in construction to ``backtest/orb.py``'s ``sessions()`` so
    the two studies are run on the same calendar: 390 regular-hours bars
    and an afternoon/morning volume ratio above the half-day cut. An
    early close still carries 390 stamps because the pull used
    session=ALL -- the last 180 are after-hours bars, so the count alone
    cannot find them and the volume ratio has to.
    """
    hh = bars.index.hour + bars.index.minute / 60
    rth = bars[(hh > 9.5) & (hh <= 16.0)]
    keep = []
    for day, g in rth.groupby(rth.index.normalize()):
        if len(g) != RTH_BARS:
            continue
        gh = g.index.hour + g.index.minute / 60
        am = float(g.volume[gh <= 13.0].sum())
        pm = float(g.volume[gh > 13.0].sum())
        if am <= 0 or pm / am < HALF_DAY_VOL_RATIO:
            continue
        keep.append(day)
    return pd.DatetimeIndex(keep)


def assign_session(idx: pd.DatetimeIndex,
                   days: pd.DatetimeIndex) -> np.ndarray:
    """Which trading day each stamp belongs to; NaT when it belongs to none.

    Stamps at or after `NEXT_SESSION_HOUR` belong to the session of the
    FOLLOWING CALENDAR DAY, and are dropped when that day is not a
    tradeable one. Everything else maps to its own calendar date, and is
    dropped when that date is not tradeable (a half session, or a day
    whose regular hours are incomplete).

    "Following calendar day", not "next trading day", and the difference
    is not cosmetic. 99.17% of overnight bars are the night before the
    very next session either way. The other 0.83% are holiday weekends,
    where "next trading day" pulls Thursday night AND Sunday night into
    one Monday session -- a block 20 that spans 5,819 minutes and calls
    itself an hour. Those nights are dropped instead: the night before a
    day that does not trade is not the overnight leg of anything.

    CALLERS MUST PASS THE SHIFTED STAMP, not the raw bar END. The first
    version of this function was handed the raw stamp while the hour
    block was cut on the shifted one, and the two rulers disagreed on
    exactly one stamp a day -- the one at 20:00. That stamp is the last
    minute of block 19, but `hour >= 20` sent it to the NEXT session, so
    block 19 of every session became {yesterday 19:59} + {today 19:01 to
    19:59}: a 24-hour return wearing a one-hour label, mean +7.78 bps
    instead of +0.26. That is the third time in this project that bar-END
    has cost something, which is why `build_panel` now asserts the wall
    clock span of every block it produces.
    """
    d = idx.normalize()
    arr = days.to_numpy()
    out = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")

    own = np.isin(d.to_numpy(), arr)
    out[own] = d.to_numpy()[own]

    roll = idx.hour >= NEXT_SESSION_HOUR
    if roll.any():
        nxt = (d[roll] + pd.Timedelta(days=1)).to_numpy()
        tgt = np.where(np.isin(nxt, arr), nxt, np.datetime64("NaT"))
        out[roll] = tgt
    return out


# -- the panel -----------------------------------------------------------

def prepare(bars: pd.DataFrame,
            days: pd.DatetimeIndex | None = None,
            block_min: int = 60) -> pd.DataFrame:
    """Bar-level frame tagged with its session, its hour block, and the
    log step from the previous bar WITHIN the same block.

    The step is NaN on a block's first bar on purpose: a move across a
    block boundary belongs to neither block, and letting it leak into one
    would put a gap -- overnight, or across the open -- inside a
    "60-minute hold" and inside the Roll estimate of that hour's spread.
    """
    if days is None:
        days = tradeable_days(bars)
    b = bars.copy()
    # ONE ruler for both the hour block and the session: the stamp moved
    # back a minute, which is the time the bar's minute STARTED. Using the
    # raw stamp for either of them puts the 20:00 bar in a different
    # session from the block it belongs to.
    shifted = b.index - pd.Timedelta(minutes=1)
    b["session"] = assign_session(shifted, days)
    # `hour` is the BLOCK INDEX, which is the hour only at the default
    # 60-minute size. At 30 minutes it runs 0..47, and 09:30-10:00 is
    # block 19. The column keeps its name so that everything downstream
    # -- the audit, the wide panel, alpha -- works at either size.
    b["hour"] = (shifted.hour * 60 + shifted.minute) // block_min
    b = b[~pd.isna(b.session)]

    step = np.log(b.close.to_numpy())
    step = np.concatenate([[np.nan], np.diff(step)])
    b["step"] = step
    # a step that crosses a block boundary is not inside either block
    newblk = (b.session.to_numpy() != np.roll(b.session.to_numpy(), 1)) | \
             (b.hour.to_numpy() != np.roll(b.hour.to_numpy(), 1))
    b.loc[newblk, "step"] = np.nan
    return b


def build_panel(bars: pd.DataFrame,
                days: pd.DatetimeIndex | None = None,
                block_min: int = 60) -> pd.DataFrame:
    """One row per (session, hour) block.

    Columns
        ret        log(last close / first open) of the block
        n_bars     how many 1-minute bars the block actually has
        zero_vol   share of those bars with zero volume (stale prices)
        bad_print  the block contains a bar-to-bar log move > BAD_PRINT_LOG
        worst      the largest bar-to-bar move in the block, absolute
    """
    if "session" not in bars.columns:
        b = prepare(bars, days, block_min)
    else:
        b = bars
    b = b.copy()
    b["zero"] = (b.volume <= 0).astype(float)
    b["astep"] = b.step.abs()
    b["t"] = b.index
    g = b.groupby(["session", "hour"], sort=True)
    panel = pd.DataFrame({
        "n_bars": g.size(),
        "t_first": g.t.min(),
        "t_last": g.t.max(),
        "first_open": g.open.first(),
        "last_close": g.close.last(),
        "volume": g.volume.sum(),
        "zero_vol": g.zero.mean(),
        "worst": g.astep.max(),
    }).reset_index()
    panel["ret"] = np.log(panel.last_close / panel.first_open)
    panel["bad_print"] = panel.worst > BAD_PRINT_LOG
    panel = panel[panel.n_bars >= min_bars(block_min)]

    # A one-hour block cannot span more than an hour. This assertion is
    # here because it would have caught, in one second, a session-
    # labelling bug that turned block 19 into a 24-hour return with a
    # ten-times-too-large mean, and that no amount of plausible-looking
    # output revealed. 70 minutes, not 60, so that it fires on a real
    # defect rather than on an exchange's odd minute.
    span = (panel.t_last - panel.t_first).dt.total_seconds() / 60
    limit = block_min + SPAN_SLACK_MIN
    if (span > limit).any():
        bad = panel.loc[span.idxmax()]
        raise AssertionError(
            f"block wall-clock span {span.max():.0f} min > {limit} "
            f"(session {bad.session}, block {bad.hour}): the block cut "
            f"and the session cut are using different rulers")
    return panel.reset_index(drop=True)


def wide(panel: pd.DataFrame, hours: tuple[int, ...],
         column: str = "ret") -> pd.DataFrame:
    """sessions x hours matrix, keeping only days that have every hour.

    Dropping partial days rather than filling them is what makes the
    intraday de-meaning in `alpha` well defined: alpha is a contrast
    ACROSS the locked hour set on the same day, so a day missing one of
    them cannot contribute to any of the others either.
    """
    w = panel.pivot(index="session", columns="hour", values=column)
    missing = [h for h in hours if h not in w.columns]
    if missing:
        raise KeyError(f"hours absent from the panel: {missing}")
    return w[list(hours)].dropna(how="any")


def alpha(w: pd.DataFrame) -> pd.DataFrame:
    """Intraday de-meaning: alpha[d,h] = r[d,h] - mean_h'(r[d,h']).

    Sums to zero across the hour set BY CONSTRUCTION, every day. So "we
    found an hour with positive excess" carries no information at all --
    positive alphas exist in every market, including one with no hour
    effect whatever. Only the max-of-k critical value makes it a finding.

    The values depend on WHICH hours are in `w`. Change the set and every
    alpha changes, which is why the set is locked before the run.
    """
    return w.sub(w.mean(axis=1), axis=0)


def winsorise(w: pd.DataFrame, frac: float = WINSOR_FRAC) -> pd.DataFrame:
    """Per-hour two-sided winsorisation at `frac`.

    Declared in advance because single bad prints in this data can move
    an hour's mean by 59-65% of its own standard error (PLAN 2.2), which
    is enough to create or destroy a finding on its own. The unwinsorised
    version is always reported beside it; if they disagree, the finding
    is a print, not an hour.
    """
    lo = w.quantile(frac)
    hi = w.quantile(1 - frac)
    return w.clip(lower=lo, upper=hi, axis=1)


# -- reconciliation ------------------------------------------------------

def reconcile(b: pd.DataFrame, panel: pd.DataFrame,
              days: pd.DatetimeIndex | None = None,
              block_min: int = 60, tol: float = 1e-9) -> dict:
    """Audit the panel against the bars it was built from.

    THE FIRST VERSION OF THIS FUNCTION CHECKED NOTHING. It telescoped the
    blocks and compared the result to a buy-and-hold it computed from the
    panel itself, which is an algebraic identity: deleting 500 blocks,
    duplicating an hour, and shifting every session label were all tried,
    and all three passed with the total unchanged to the last digit. A
    check that cannot fail is worse than no check, because it is quoted.

    So there are now three, and each one can actually fail:

      bars      every bar assigned to a session is in exactly one cell
      label     each cell's session is the one its own first timestamp
                implies -- a shifted label changes no return and no
                total, so only this check sees it
      order     blocks telescope IN TIME ORDER -- sorting by hour instead
                is wrong the moment a session contains hours 20..23
      external  the telescoped total equals a buy-and-hold taken straight
                from `b`, by timestamp, never touching the panel

    Mutation-checked, since the point of this function is to fail: with
    500 blocks deleted, with hour 11 duplicated, and with every session
    label shifted by 60 rows, all three are rejected.
    """
    p = panel.sort_values("t_first")
    # -- bars: nothing dropped, nothing counted twice -------------------
    eligible = b.groupby(["session", "hour"]).size()
    floor = min_bars(block_min)
    eligible = eligible[eligible >= floor]
    if len(p) != len(eligible):
        raise AssertionError(
            f"{len(p)} cells in the panel, {len(eligible)} groups of "
            f">= {floor} bars in the data")
    if int(p.n_bars.sum()) != int(eligible.sum()):
        raise AssertionError(
            f"{int(p.n_bars.sum())} bars in the panel, "
            f"{int(eligible.sum())} in the eligible groups")

    # -- label: the session has to follow from the timestamp ------------
    if days is not None:
        want = assign_session(
            pd.DatetimeIndex(p.t_first) - pd.Timedelta(minutes=1), days)
        wrong = int((pd.Series(want).to_numpy()
                     != p.session.to_numpy()).sum())
        if wrong:
            raise AssertionError(
                f"{wrong} cells carry a session their own first bar does "
                f"not imply")

    # -- order: telescope along the clock, not along the hour label -----
    if not p.t_first.is_monotonic_increasing:
        raise AssertionError("blocks are not in time order")
    blocks = float(p.ret.sum())
    gaps = float(np.log(p.first_open.to_numpy()[1:]
                        / p.last_close.to_numpy()[:-1]).sum())

    # -- external: the comparison value never touches the panel ---------
    t0, t1 = p.t_first.iloc[0], p.t_last.iloc[-1]
    bh = float(np.log(b.close.loc[t1] / b.open.loc[t0]))
    if abs(blocks + gaps - bh) > tol:
        raise AssertionError(
            f"panel does not rebuild the bars: {blocks + gaps:.9f} vs "
            f"{bh:.9f} taken straight from {t0} .. {t1}")

    inside = float(p.groupby("session").ret.sum().sum()
                   + p.groupby("session").apply(
                       lambda g: float(np.log(
                           g.first_open.to_numpy()[1:]
                           / g.last_close.to_numpy()[:-1]).sum()),
                       include_groups=False).sum())
    return dict(sessions=int(p.session.nunique()), cells=len(p),
                bars=int(p.n_bars.sum()), blocks=blocks, gaps=gaps,
                inside=inside, bh=bh, t0=t0, t1=t1,
                max_span=float((p.t_last - p.t_first)
                               .dt.total_seconds().max() / 60))
