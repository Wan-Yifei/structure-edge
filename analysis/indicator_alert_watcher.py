"""Open-ended indicator-threshold live-scan watcher -- shared by the live
signal scanner (analysis/signal_scanner.py).

Unlike the other watchers (session_vp/vah_soxs/val_kd_tp), this is not an
entry-signal detector -- it's a continuous status monitor: "is this rule's
indicator currently past its threshold?" A rule fires (repeatedly, once per
scan cycle, not just on the edge crossing) whenever the latest closed bar's
indicator value satisfies condition+threshold; the UI provides a Mute button
to pause a noisy rule until it goes negative again (see
db.signals.SignalsDB.upsert_indicator_alert_state's docstring for how mute
is protected from being raced by the next scan cycle).

INDICATOR_REGISTRY is the whole point of this module: adding a new
indicator (MACD, classic KDJ, ...) later is "write one function, register
one line" -- no schema change, no UI change, no scanner dispatch change.
Each registered function takes (klines, params: dict) and returns
dict[str, np.ndarray] -- the dict keys are the selectable "field" values a
rule's config can pick from.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

import numpy as np

from feeds.fetcher import fetch_klines
from strategy.session_vp.profile import compute_value_area
from strategy.session_vp.reversal import compute_rsi
from strategy.smc.fvg import compute_volume_profile
from strategy.smc.kd_trend import compute_kd

_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "scanner" / "indicator_alert_params.json"
_SCHEDULE_PATH = _ROOT / "config" / "schedule.json"


def _compute_rsi_series(klines, params: dict) -> dict[str, np.ndarray]:
    period = params.get("period", 6)
    return {"value": compute_rsi(klines["close"], period)}


def _compute_kd_series(klines, params: dict) -> dict[str, np.ndarray]:
    """The KD *channel* indicator already in this repo (strategy/smc/kd_trend.py)
    -- EMA-band midlines/spread, NOT the classic 0-100 KDJ stochastic. Registered
    so a future rule can watch e.g. field="spread" crossing a level."""
    fast = params.get("fast", 25)
    slow = params.get("slow", 90)
    kd = compute_kd(klines, fast=fast, slow=slow)
    return {col: kd[col].to_numpy(dtype=float) for col in
            ["up1", "lo1", "mid1", "up2", "lo2", "mid2", "spread", "width"]}


def _load_schedule_sessions() -> dict:
    with open(_SCHEDULE_PATH, encoding="utf-8") as f:
        return json.load(f)["sessions"]


def _compute_session_va_series(klines, params: dict) -> dict[str, np.ndarray]:
    """Distance from the latest price to the CURRENT session occurrence's
    POC / VAH / VAL.

    Params:
      session         only evaluate while this session is the active one
                      (e.g. "regular"); omit to follow whichever session the
                      latest bar falls in.
      warmup_minutes  freeze the profile over the session's first N minutes
                      (default 30, matching the session_vp strategy). Set 0
                      for a developing profile over the whole session so far
                      -- note that a developing VAH tracks price as price
                      makes new highs, so a "near VAH" rule on a rolling
                      profile self-triggers on breakouts; the frozen profile
                      gives a level that stays put while price approaches it.
      n_bins          profile resolution (default 50)
      va_pct          value-area coverage (default 0.70)

    Fields: poc/vah/val (the levels), price, plus per level
      dist_<lvl>_pct / dist_<lvl>_abs  unsigned gap -- "how close", pair with
                                       condition "below" for a proximity alert
      off_<lvl>_pct  / off_<lvl>_abs   signed gap (+ above / - below) -- pair
                                       with "above"/"below" 0 for a break alert
    OHLCV-based (strategy.smc.compute_volume_profile), no ticks.db dependency
    -- same choice strategy/session_vp makes, since the scanner runs on symbols
    and dates that may have no tick coverage at all.

    Every field is a full-length array of NaN with only the last bar filled:
    scan_indicator_alert() reads series[-1], and a genuine per-bar series
    would mean rebuilding the profile once per bar for a value nothing reads.
    """
    n = len(klines)
    nan_col = lambda: np.full(n, np.nan)   # noqa: E731 -- one-liner factory
    levels  = ("poc", "vah", "val")
    out = {"price": nan_col()}
    for lvl in levels:
        out[lvl] = nan_col()
        for pre in ("dist", "off"):
            for unit in ("pct", "abs"):
                out[f"{pre}_{lvl}_{unit}"] = nan_col()
    if n == 0:
        return out

    # Imported lazily: backtest.session_vp_engine pulls in the backtest stack,
    # which the other indicators here have no reason to load.
    from backtest.session_vp_engine import precompute_session_context

    ctx = precompute_session_context(klines, _load_schedule_sessions())
    last_oid = int(ctx["occurrence_id"][-1])
    if last_oid < 0:
        return out           # latest bar is outside every enabled session
    idxs = ctx["groups"][last_oid]
    session_name, _elapsed = ctx["session_info"][idxs[-1]]
    want = params.get("session")
    if want and session_name != want:
        return out           # a different session is active -- rule idles

    warmup = int(params.get("warmup_minutes", 30))
    if warmup > 0:
        end_i = next((i for i in idxs if ctx["session_info"][i][1] >= warmup), None)
        if end_i is None:
            return out       # still inside the warmup -- no frozen profile yet
    else:
        end_i = idxs[-1]     # developing profile over the session so far

    edges, bin_vols = compute_volume_profile(
        klines.iloc[idxs[0] : end_i + 1], n_bins=int(params.get("n_bins", 50)))
    if edges is None:
        return out           # degenerate price range (flat session)

    va = compute_value_area(edges, bin_vols, va_pct=float(params.get("va_pct", 0.70)))
    price = float(klines["close"].iloc[-1])
    out["price"][-1] = price
    for lvl in levels:
        level = float(va[lvl])
        if level <= 0:
            continue
        out[lvl][-1] = level
        out[f"off_{lvl}_abs"][-1]  = price - level
        out[f"off_{lvl}_pct"][-1]  = (price - level) / level * 100.0
        out[f"dist_{lvl}_abs"][-1] = abs(price - level)
        out[f"dist_{lvl}_pct"][-1] = abs(price - level) / level * 100.0
    return out


INDICATOR_REGISTRY = {
    "rsi": _compute_rsi_series,
    "kd":  _compute_kd_series,
    "session_va": _compute_session_va_series,
    # Add a new indicator by writing a _compute_xxx_series(klines, params) ->
    # dict[str, np.ndarray] function above and registering it here -- e.g.
    # "macd": _compute_macd_series. Nothing else in this file, the DB schema,
    # or the scanner UI needs to change.
}


def load_indicator_alert_config(path: pathlib.Path | None = None) -> dict[str, list[dict]]:
    """Load the per-symbol list of indicator-threshold rules.

    Returns {symbol: [{"indicator": ..., "tf": ..., "params": {...},
    "field": ..., "condition": "above"|"below", "threshold": ...}, ...]}.
    Top-level "_"-prefixed keys are ignored. Returns {} if the file doesn't
    exist yet.
    """
    p = path or _DEFAULT_CONFIG_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def rule_enabled(rule: dict) -> bool:
    """Whether a rule takes part in the scan at all.

    Distinct from the UI's Mute button: mute is a temporary silence that the
    scanner auto-lifts once the reading goes negative again, and a muted rule
    keeps being evaluated so its row stays live in the table. `enabled: false`
    takes the rule out of the scan entirely -- kept in the config, costing no
    fetch and producing no row, until it is switched back on.

    A rule with no "enabled" key is on, so configs written before this flag
    existed keep working untouched.
    """
    return bool(rule.get("enabled", True))


def rule_id(symbol: str, rule: dict) -> str:
    # "enabled" is deliberately not part of the identity -- toggling a rule
    # off and back on must land on the same row, not orphan the old one.
    params_json = json.dumps(rule.get("params", {}), sort_keys=True)
    return "|".join([
        symbol, rule["tf"], rule["indicator"], rule["field"],
        rule["condition"], str(rule["threshold"]), params_json,
    ])


def scan_indicator_alert(
    symbol: str,
    rule: dict,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> dict | None:
    """Evaluate one rule against the latest closed bar. Returns None when the
    indicator isn't registered, there isn't enough data, or the computed
    value is NaN (e.g. still inside the indicator's warmup window) -- never
    raises for an unknown indicator name, so a typo in the config just skips
    that rule instead of crashing the scan cycle.
    """
    compute_fn = INDICATOR_REGISTRY.get(rule["indicator"])
    if compute_fn is None:
        return None

    klines = fetch_klines(symbol, rule["tf"], start, end, force_refresh=force_refresh)
    if klines is None or klines.empty:
        return None

    params = rule.get("params", {})
    series_dict = compute_fn(klines, params)
    series = series_dict.get(rule["field"])
    if series is None or len(series) == 0:
        return None

    value = float(series[-1])
    if np.isnan(value):
        return None

    threshold = float(rule["threshold"])
    condition = rule["condition"]
    if condition == "above":
        is_positive = value > threshold
    elif condition == "below":
        is_positive = value < threshold
    else:
        return None

    return {
        "rule_id":      rule_id(symbol, rule),
        "symbol":       symbol,
        "tf":           rule["tf"],
        "indicator":    rule["indicator"],
        "params_json":  json.dumps(params, sort_keys=True),
        "field":        rule["field"],
        "condition":    condition,
        "threshold":    threshold,
        "value":        value,
        "is_positive":  is_positive,
        "last_bar_time": str(klines["time_key"].iloc[-1])[:19],
    }
