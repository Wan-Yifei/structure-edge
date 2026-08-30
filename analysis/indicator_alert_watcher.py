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
from strategy.session_vp.reversal import compute_rsi
from strategy.smc.kd_trend import compute_kd

_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "scanner" / "indicator_alert_params.json"


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


INDICATOR_REGISTRY = {
    "rsi": _compute_rsi_series,
    "kd":  _compute_kd_series,
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


def _rule_id(symbol: str, rule: dict) -> str:
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
        "rule_id":      _rule_id(symbol, rule),
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
