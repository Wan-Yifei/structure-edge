"""Session Value-Area reversal backtest engine -- long-only, session-scoped.

Strategy: after each session (premarket/regular/afterhours/overnight) starts,
wait `warmup_minutes`, freeze a volume profile (POC/VAL) from the bars since
session start, then go long on a VAL-touch-then-RSI-oversold reversal
(strategy/session_vp/reversal.py), targeting POC with the stop mirrored for
an exact 1:1 R:R (strategy/session_vp/profile.py computes POC/VAL).

Structurally unrelated to the SMC strategy in backtest/engine.py (no FVG
zones, no BOS/CHoCH trend, session-scoped instead of continuous) -- there is
no strategy-plug-in seam in engine.py's run_backtest(), so this is its own
top-level loop. Everything below the loop is reused directly from
backtest/engine.py: _find_exit() for exit simulation and BacktestResult
(generic over any params object with .to_dict() and any Trade-like objects
with .direction/.result/.r_multiple).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, _find_exit
from core.time_utils import minutes_since_session_start
from strategy.session_vp.profile import compute_value_area
from strategy.session_vp.reversal import compute_rsi, detect_val_reversal
from strategy.smc.fvg import compute_volume_profile


def _algo_version() -> str:
    """Return the most recent svp_v* git tag on HEAD.

    Mirrors backtest/engine.py's _algo_version() exactly (same rationale:
    version-stamp every trade ID so the same entry under a different algo
    version gets a distinct ID), but against a separate tag namespace --
    this is a structurally unrelated strategy family and must not share
    smc_v*'s tags, or compare_versions.py/trade-ID stamps would conflate
    the two.
    """
    import os, subprocess, pathlib
    env_ver = os.environ.get("SVP_ALGO_VERSION", "").strip()
    if env_ver:
        return env_ver
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--match", "svp_v*", "--abbrev=0"],
            cwd=pathlib.Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "svp_unknown"


ALGO_VERSION = _algo_version()


@dataclass
class SessionVPParams:
    warmup_minutes:       int   = 45     # wait this long after session start before freezing the profile
    va_pct:               float = 0.70   # value-area coverage target
    n_bins:               int   = 60     # profile resolution
    rsi_period:           int   = 6
    rsi_threshold:        float = 30.0
    tradeable_sessions:   tuple = ("premarket", "regular", "afterhours", "overnight")
    max_bars:             int   = 120    # holding cap passed to _find_exit
    min_val_poc_dist_pct: float = 0.001  # skip a session if (poc-val)/val is below this -- too tight/noisy a range for a meaningful 1:1 target
    entry_tf:             str   = "1m"   # single-timeframe strategy: profile + signal both computed off this TF
    trend_tf:             str   = "1m"   # always equal to entry_tf -- kept as its own field only because report.py/db.py's DB schema hard-require both columns (confirmed by running report.py against a real result CSV: it KeyErrors on a missing trend_tf column)
    target_rr:            float = 1.0    # tp stays anchored at POC (the market-structure target); sl_dist = (poc-entry)/target_rr, so target_rr<1 widens the stop (more room, lower R per win) while target_rr=1 reproduces the original exact-1:1 design
    val_proximity_pct:    float = 0.0    # widen the VAL-touch test to low <= val*(1+val_proximity_pct) -- 0.0 reproduces the original "price must actually reach VAL" behavior; >0 lets a signal fire before price gets all the way down to VAL

    def __post_init__(self) -> None:
        if isinstance(self.tradeable_sessions, list):
            self.tradeable_sessions = tuple(self.tradeable_sessions)
        self.trend_tf = self.entry_tf

    def label(self) -> str:
        sessions = "+".join(s[:3] for s in self.tradeable_sessions)
        return (
            f"warmup{self.warmup_minutes} va{self.va_pct:.2f} bins{self.n_bins}"
            f" rsi{self.rsi_period}<{self.rsi_threshold:.0f} maxb{self.max_bars}"
            f" mindist{self.min_val_poc_dist_pct:.4f} rr{self.target_rr:.2f}"
            f" prox{self.val_proximity_pct:.4f} [{sessions}]"
        )

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]
        d["tradeable_sessions"] = json.dumps(list(self.tradeable_sessions))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionVPParams":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        if "tradeable_sessions" in d and isinstance(d["tradeable_sessions"], str):
            d["tradeable_sessions"] = tuple(json.loads(d["tradeable_sessions"]))
        return cls(**d)


@dataclass
class Trade:
    direction:     str            # always "bull" -- long-only strategy
    entry_price:   float
    sl:            float
    tp:            float
    planned_rr:    float
    entry_time:    str
    exit_time:     str   = ""
    exit_price:    float = 0.0
    result:        str   = ""     # "win" | "loss" | "timeout"
    r_multiple:    float = 0.0
    trade_id:      str   = ""
    entry_ltf_bar: int   = 0
    fvg_top:       float = 0.0    # unused -- kept only so backtest.db.write_trades() (fixed field list) still works
    fvg_bottom:    float = 0.0    # unused, ditto
    session:       str   = ""     # which of the 4 sessions this trade belongs to (not persisted by write_trades -- the trades table has no such column; recoverable in-run via params.tradeable_sessions since a run covers one session by convention, see run_session_vp.py)

    def __post_init__(self) -> None:
        if not self.trade_id:
            key = f"{ALGO_VERSION}:{self.entry_time}:{self.direction}:{self.entry_price:.6f}:{self.sl:.6f}"
            self.trade_id = hashlib.sha256(key.encode()).hexdigest()[:8]


def _session_occurrence_ids(session_info: list[tuple[str, int] | None]) -> np.ndarray:
    """Group bars into contiguous session occurrences.

    A new occurrence starts whenever the session name changes, the elapsed-
    minutes-since-start resets to a lower value than the previous bar's (a
    new calendar occurrence of the same session type), or after a gap
    (bar not in any session). Returns an array of occurrence IDs, -1 for
    bars not in any session.
    """
    n = len(session_info)
    occurrence_id = np.full(n, -1, dtype=int)
    cur_id = -1
    prev_name: str | None = None
    prev_elapsed = -1
    for idx in range(n):
        info = session_info[idx]
        if info is None:
            prev_name, prev_elapsed = None, -1
            continue
        name, elapsed = info
        if name != prev_name or elapsed < prev_elapsed:
            cur_id += 1
        occurrence_id[idx] = cur_id
        prev_name, prev_elapsed = name, elapsed
    return occurrence_id


def precompute_session_context(klines: pd.DataFrame, schedule_sessions: dict) -> dict:
    """Precompute the parts of a session_vp backtest that don't depend on
    SessionVPParams at all -- session_info/occurrence_id/groups derive only
    from klines' timestamps and the schedule.

    A grid/random search runs thousands of SessionVPParams combos against
    the SAME klines, and each of these three steps is an O(len(klines))
    pure-Python loop. Recomputing them inside run_backtest_session_vp on
    every combo made a full year of 1-minute bars (~1000 session
    occurrences) take hours instead of minutes -- compute this once per
    klines/schedule pair (e.g. once in the CLI runner before dispatching to
    worker processes) and pass it into run_backtest_session_vp via
    `session_context` for every combo.
    """
    times = pd.to_datetime(klines["time_key"].astype(str).str[:16], format="%Y-%m-%d %H:%M")
    session_info = [
        minutes_since_session_start(t.to_pydatetime(), schedule_sessions) for t in times
    ]
    occurrence_id = _session_occurrence_ids(session_info)

    groups: dict[int, list[int]] = {}
    for idx, oid in enumerate(occurrence_id):
        if oid < 0:
            continue
        groups.setdefault(int(oid), []).append(idx)

    return {"session_info": session_info, "occurrence_id": occurrence_id, "groups": groups}


def run_backtest_session_vp(
    klines: pd.DataFrame,
    params: SessionVPParams,
    schedule_sessions: dict,
    session_context: dict | None = None,
) -> BacktestResult:
    """Backtest the session value-area reversal strategy over `klines`.

    klines: single-timeframe OHLCV, columns time_key/open/high/low/close/volume,
    sorted ascending, time_key as "YYYY-MM-DD HH:MM[...]" strings.
    schedule_sessions: config/schedule.json's "sessions" dict (session name ->
    {"start": "HH:MM", "end": "HH:MM", "enabled": bool}).
    session_context: precompute_session_context(klines, schedule_sessions)'s
    result, when the caller is running many combos against the same klines
    (see that function's docstring). Computed internally when omitted, so
    this function stays usable standalone (e.g. in tests) with just klines
    and params.

    One trade per session occurrence (first valid signal only -- no
    pyramiding in v1). Exit is simulated by _find_exit without an
    intraday-only cap, matching backtest/engine.py's default (non-
    intraday_only) behavior: a trade may close in a later session if it
    hasn't hit SL/TP/max_bars first.
    """
    result = BacktestResult(params=params, trades=[])
    n = len(klines)
    if n == 0:
        return result

    lows   = klines["low"].to_numpy(dtype=float)
    highs  = klines["high"].to_numpy(dtype=float)
    closes = klines["close"].to_numpy(dtype=float)

    if session_context is None:
        session_context = precompute_session_context(klines, schedule_sessions)
    session_info  = session_context["session_info"]
    groups        = session_context["groups"]

    # Computed once for the whole run (not per session occurrence) -- RSI only
    # depends on params.rsi_period, and recomputing it per occurrence (there
    # can be ~1000s per backtest) dominated runtime on anything but tiny inputs.
    rsi = compute_rsi(klines["close"], params.rsi_period)

    for idxs in groups.values():
        sess_name = session_info[idxs[0]][0]  # type: ignore[index]
        if sess_name not in params.tradeable_sessions:
            continue

        session_start = idxs[0]
        session_end   = idxs[-1]

        warmup_end = None
        for idx in idxs:
            if session_info[idx][1] >= params.warmup_minutes:  # type: ignore[index]
                warmup_end = idx
                break
        if warmup_end is None:
            continue  # session shorter than the warmup window

        profile_klines = klines.iloc[session_start : warmup_end + 1]
        edges, bin_vols = compute_volume_profile(profile_klines, n_bins=params.n_bins)
        va  = compute_value_area(edges, bin_vols, va_pct=params.va_pct)
        poc, val = va["poc"], va["val"]
        if poc <= 0 or val <= 0 or poc <= val:
            continue
        if (poc - val) / val < params.min_val_poc_dist_pct:
            continue

        # klines/rsi/lows/highs/closes are the FULL arrays (not just this
        # session's slice) so RSI has its normal warmup history from before
        # the session started -- see detect_val_reversal's "caller
        # responsibility" note. start_idx/end_idx bound the scan to just
        # this occurrence -- leaving end_idx open would rescan all the way
        # to the end of the whole klines array on every occurrence.
        signals = detect_val_reversal(
            klines, val,
            rsi_period=params.rsi_period,
            rsi_threshold=params.rsi_threshold,
            start_idx=warmup_end,
            end_idx=session_end,
            val_proximity_pct=params.val_proximity_pct,
            rsi=rsi, lows=lows, highs=highs, closes=closes,
        )
        if not signals:
            continue

        sig         = signals[0]  # one trade per session occurrence (v1: no pyramiding)
        entry_idx   = sig["entry_idx"]
        entry_price = sig["entry_price"]
        tp = poc
        tp_dist = tp - entry_price
        sl_dist = tp_dist / params.target_rr
        sl = entry_price - sl_dist
        if sl_dist <= 0:
            continue

        exit_bar, exit_price, outcome = _find_exit(
            lows, highs, closes,
            from_bar=entry_idx, sl=sl, tp=tp,
            direction="bull", max_bars=params.max_bars,
        )
        if outcome == "win":
            r_multiple = (tp - entry_price) / sl_dist
        elif outcome == "loss":
            r_multiple = -1.0
        else:
            r_multiple = (exit_price - entry_price) / sl_dist

        result.trades.append(Trade(
            direction     = "bull",
            entry_price   = entry_price,
            sl            = sl,
            tp            = tp,
            planned_rr    = params.target_rr,
            entry_time    = str(klines["time_key"].iloc[entry_idx]),
            exit_time     = str(klines["time_key"].iloc[exit_bar]),
            exit_price    = exit_price,
            result        = outcome,
            r_multiple    = r_multiple,
            entry_ltf_bar = entry_idx,
            session       = sess_name,
        ))

    return result
