"""Startup defaults for the PyQtGraph viewer, read from a TOML file.

Every toggle and spinbox in `trade_viewer_qt.py` and `liq_hm_window.py`
used to carry its opening value as a literal in the widget-construction
code, so changing which overlays a chart opens with meant editing Python.
This module moves those values into `config/viewer_qt.toml`.

`DEFAULTS` below is the schema AND the fallback: it lists every key that
may appear in the file, with the value used when the file is absent,
unreadable, or silent about that key. Deleting the config file therefore
changes nothing.

The schema is also what makes a typo visible. A key the file sets that
`DEFAULTS` does not know is reported through `problems()` rather than
silently ignored -- a config whose misspelled keys do nothing is worse
than no config, because it looks like it worked.

This is startup state only. Nothing here is written back: toggling a
checkbox in a running window does not touch the file, and reopening the
window returns to what the file says.
"""
from __future__ import annotations

import pathlib
import tomllib

_CFG_PATH = pathlib.Path(__file__).parent.parent / "config" / "viewer_qt.toml"

# section -> key -> default. Types are taken from these values: a bool
# default accepts only a bool, a float default accepts an int too, and a
# list default accepts a list of strings.
DEFAULTS: dict[str, dict] = {
    # Toolbar checkboxes, by the key they are registered under in
    # TradeViewerQt._ind_checks.
    "indicators": {
        # SMC row
        "bos_choch": False,
        "fvg": False,
        "ob": True,
        # Indicators row
        "heatmap": True,
        "delta": True,
        "kd_band": False,
        "kd": False,
        "cvd": True,
        "adi": False,
        "dcvd": True,
        "absorb": True,
        "offex": True,
        "ema": False,
        "avwap": False,
        "vol": False,
        "chandelier": False,
        "atr_stop": False,
        "red_up": True,
        # Order-flow row: tick profile size filter and display mode
        "tick_s": True,
        "tick_m": True,
        "tick_l": True,
        "tick_vp": False,
        "tick_net": False,
        # Options row. Walls draw on the chart; Alerts fire notifications.
        # Drawing by default is harmless, starting a notification stream
        # unasked is not.
        "option_walls": True,
        "option_alerts": False,
    },
    # Which sessions feed the volume profile, and how many trading days
    # the profile and the CVD window cover.
    "session": {
        "segments": ["regular", "pre", "post", "night"],
        "range": "1d",
    },
    # Toolbar spinboxes.
    "numbers": {
        "redraw_secs": 0.5,
        "fvg_min_pct": 1.0,
        "ob_max_count": 4,
        "dcvd_lookback": 1,
        "avwap_sigma": 0,
        "profile_bins": 120,
        "poc_trail": 2,
        "poc_hysteresis_pct": 5,
        "tick_top_n": 3,
        "option_dte": 7,
        "option_wall_count": 4,
        "option_poll_secs": 15,
        "option_threshold": 500,
    },
    # The Liquidity Heatmap window (liq_hm_window.py).
    "liq_hm": {
        "bid_ask": True,
        "price_path": True,
        "iceberg": False,
        "spoof": False,
        "wall": True,
        "imbalance": True,
        "absorb": True,
        "gamma": 1.0,
        "min_vol": 0,
        "col_secs": 1,
        "max_cols": 240,
        "n_price": 100,
        "ice_min_ref": 3,
        "spoof_dur": 30,
        "wall_n": 2,
        "simb_levels": 3,
        "simb_ratio": 3.0,
        "simb_depth": 10,
        "absorb_min_vol": 6000,
    },
}

_SESSION_KEYS = ("regular", "pre", "post", "night")
_RANGE_KEYS = ("1d", "2d", "3d", "7d")

_loaded: dict[str, dict] | None = None
_problems: list[str] = []


def _typed(section: str, key: str, raw, fallback):
    """Return `raw` if it matches the fallback's type, else None."""
    if isinstance(fallback, bool):
        return raw if isinstance(raw, bool) else None
    if isinstance(fallback, float):
        return float(raw) if isinstance(raw, (int, float)) \
            and not isinstance(raw, bool) else None
    if isinstance(fallback, int):
        return raw if isinstance(raw, int) and not isinstance(raw, bool) \
            else None
    if isinstance(fallback, list):
        return list(raw) if isinstance(raw, list) and all(
            isinstance(x, str) for x in raw) else None
    if isinstance(fallback, str):
        return raw if isinstance(raw, str) else None
    return None


def _load() -> dict[str, dict]:
    global _loaded
    if _loaded is not None:
        return _loaded
    _problems.clear()
    out = {s: dict(d) for s, d in DEFAULTS.items()}
    if not _CFG_PATH.exists():
        _loaded = out
        return out
    try:
        with open(_CFG_PATH, "rb") as f:
            raw = tomllib.load(f)
    except Exception as exc:                          # noqa: BLE001
        # A broken file must not stop the chart from opening; it costs
        # the custom defaults and says so.
        _problems.append(f"{_CFG_PATH.name} unreadable ({exc}); "
                         f"built-in defaults used")
        _loaded = out
        return out
    for section, body in raw.items():
        if section not in DEFAULTS:
            _problems.append(f"unknown section [{section}]")
            continue
        if not isinstance(body, dict):
            _problems.append(f"[{section}] is not a table")
            continue
        for key, val in body.items():
            if key not in DEFAULTS[section]:
                _problems.append(f"unknown key [{section}] {key}")
                continue
            good = _typed(section, key, val, DEFAULTS[section][key])
            if good is None:
                _problems.append(
                    f"[{section}] {key} = {val!r} has the wrong type; "
                    f"keeping {DEFAULTS[section][key]!r}")
                continue
            out[section][key] = good
    # Values that are names rather than numbers: a misspelt one would
    # otherwise silently select nothing.
    bad = [s for s in out["session"]["segments"] if s not in _SESSION_KEYS]
    if bad:
        _problems.append(f"[session] unknown segments {bad}; "
                         f"valid: {list(_SESSION_KEYS)}")
        out["session"]["segments"] = [
            s for s in out["session"]["segments"] if s in _SESSION_KEYS]
    if out["session"]["range"] not in _RANGE_KEYS:
        _problems.append(f"[session] range = "
                         f"{out['session']['range']!r} is not one of "
                         f"{list(_RANGE_KEYS)}; keeping "
                         f"{DEFAULTS['session']['range']!r}")
        out["session"]["range"] = DEFAULTS["session"]["range"]
    _loaded = out
    return out


def reload(path: pathlib.Path | str | None = None) -> None:
    """Drop the cache, optionally pointing at a different file (tests)."""
    global _CFG_PATH, _loaded
    if path is not None:
        _CFG_PATH = pathlib.Path(path)
    _loaded = None


def get(section: str, key: str):
    """The configured value, or the built-in default."""
    return _load()[section][key]


def flag(key: str) -> bool:
    """Startup state of the `_ind_checks` checkbox registered under `key`."""
    return bool(_load()["indicators"][key])


def num(key: str):
    """Startup value of a toolbar spinbox."""
    return _load()["numbers"][key]


def liq(key: str):
    """Startup value of a Liquidity Heatmap control."""
    return _load()["liq_hm"][key]


def sessions() -> list[str]:
    return list(_load()["session"]["segments"])


def problems() -> list[str]:
    """Complaints about the config file, for the window to log on open."""
    _load()
    return list(_problems)


def path() -> pathlib.Path:
    return _CFG_PATH
