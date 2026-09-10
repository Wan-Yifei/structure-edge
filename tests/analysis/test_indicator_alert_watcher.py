"""Unit tests for analysis/indicator_alert_watcher.py — the `enabled` flag and
the session_va (session value-area proximity) indicator."""

import numpy as np
import pandas as pd

from analysis.indicator_alert_watcher import (
    INDICATOR_REGISTRY,
    rule_enabled,
    rule_id,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _session_klines(
    n: int = 90,
    start: str = "2026-09-09 09:31",
    heavy_first: int = 30,
) -> pd.DataFrame:
    """1-minute bars inside the regular session (time_key = bar END).

    Volume is concentrated in the first `heavy_first` bars so a warmup-frozen
    profile has a clearly different POC from a whole-session one.
    """
    times = pd.date_range(start, periods=n, freq="1min")
    close = np.concatenate([
        np.linspace(100.0, 101.0, n * 2 // 3),
        np.linspace(101.0, 100.6, n - n * 2 // 3),
    ])
    volume = np.where(np.arange(n) < heavy_first, 5000.0, 500.0)
    return pd.DataFrame({
        "time_key": times.strftime("%Y-%m-%d %H:%M:%S"),
        "open":     close,
        "high":     close + 0.03,
        "low":      close - 0.03,
        "close":    close,
        "volume":   volume,
    })


def _session_va(klines, **params) -> dict:
    return INDICATOR_REGISTRY["session_va"](klines, params)


_LEVELS = ("poc", "vah", "val")
_FIELDS = (
    ["price", *_LEVELS]
    + [f"{pre}_{lvl}_{unit}"
       for pre in ("dist", "off") for lvl in _LEVELS for unit in ("pct", "abs")]
)


# ── rule_enabled / rule_id ────────────────────────────────────────────────────

class TestRuleEnabled:
    def test_missing_key_is_enabled(self):
        """Configs written before the flag existed must keep scanning."""
        assert rule_enabled({"indicator": "rsi"}) is True

    def test_explicit_true_and_false(self):
        assert rule_enabled({"enabled": True}) is True
        assert rule_enabled({"enabled": False}) is False

    def test_rule_id_ignores_enabled(self):
        """Toggling a rule off and back on must land on the same DB row."""
        rule = {"indicator": "rsi", "tf": "1m", "params": {"period": 6},
                "field": "value", "condition": "below", "threshold": 30.0}
        assert rule_id("US.SOXL", {**rule, "enabled": True}) == \
               rule_id("US.SOXL", {**rule, "enabled": False})
        assert rule_id("US.SOXL", rule) == rule_id("US.SOXL", {**rule, "enabled": False})

    def test_rule_id_separates_params(self):
        """Two session_va rules differing only by session are distinct rules."""
        base = {"indicator": "session_va", "tf": "1m", "field": "dist_vah_pct",
                "condition": "below", "threshold": 0.15}
        a = rule_id("US.SOXL", {**base, "params": {"session": "regular"}})
        b = rule_id("US.SOXL", {**base, "params": {"session": "premarket"}})
        assert a != b


# ── session_va ────────────────────────────────────────────────────────────────

class TestSessionVAFields:
    def test_registered(self):
        assert "session_va" in INDICATOR_REGISTRY

    def test_returns_all_fields_full_length(self):
        kl = _session_klines()
        out = _session_va(kl, session="regular", warmup_minutes=30)
        assert set(out) == set(_FIELDS)
        assert all(len(v) == len(kl) for v in out.values())

    def test_only_last_bar_is_populated(self):
        """Documented contract: scan_indicator_alert reads series[-1] only."""
        out = _session_va(_session_klines(), session="regular", warmup_minutes=30)
        for name, arr in out.items():
            assert not np.isnan(arr[-1]), name
            assert np.all(np.isnan(arr[:-1])), name

    def test_levels_ordered_and_distances_consistent(self):
        out = _session_va(_session_klines(), session="regular", warmup_minutes=30)
        poc, vah, val = (float(out[k][-1]) for k in _LEVELS)
        price = float(out["price"][-1])
        assert val <= poc <= vah
        for lvl, level in zip(_LEVELS, (poc, vah, val)):
            assert out[f"off_{lvl}_abs"][-1] == price - level
            assert out[f"dist_{lvl}_abs"][-1] == abs(price - level)
            assert out[f"dist_{lvl}_pct"][-1] == abs(price - level) / level * 100.0

    def test_dist_is_unsigned_and_off_is_signed(self):
        """A price above VAH: dist stays positive, off goes positive; a price
        below VAL: dist stays positive, off goes negative."""
        out = _session_va(_session_klines(), session="regular", warmup_minutes=30)
        assert out["price"][-1] > out["vah"][-1]
        assert out["off_vah_abs"][-1] > 0
        assert out["dist_vah_abs"][-1] > 0
        assert out["off_val_abs"][-1] > 0

        kl = _session_klines()
        kl.loc[kl.index[-1], "close"] = 99.0     # drop the last close under VAL
        low = _session_va(kl, session="regular", warmup_minutes=30)
        assert low["off_val_abs"][-1] < 0
        assert low["dist_val_abs"][-1] > 0
        assert low["dist_val_abs"][-1] == abs(low["off_val_abs"][-1])


class TestSessionVAWindow:
    def test_frozen_and_rolling_differ(self):
        """The whole-session profile keeps absorbing bars after the warmup, so
        its levels drift away from the frozen ones."""
        kl = _session_klines()
        frozen  = _session_va(kl, session="regular", warmup_minutes=30)
        rolling = _session_va(kl, session="regular", warmup_minutes=0)
        assert float(rolling["vah"][-1]) != float(frozen["vah"][-1])

    def test_rolling_vah_tracks_a_breakout_but_frozen_does_not(self):
        """Why frozen is the default: on a developing profile VAH follows a
        breakout, so a 'near VAH' rule would chase price instead of waiting at
        a level. The warmup-frozen profile is unmoved by anything after it."""
        kl = _session_klines()
        kl2 = _session_klines()
        late = kl2.index[-10:]
        kl2.loc[late, ["high", "low", "open", "close"]] += 2.0   # breakout...
        kl2.loc[late, "volume"] = 20000.0                        # ...on volume

        rolling_before = _session_va(kl,  session="regular", warmup_minutes=0)
        rolling_after  = _session_va(kl2, session="regular", warmup_minutes=0)
        assert float(rolling_after["vah"][-1]) > float(rolling_before["vah"][-1])

        frozen_before = _session_va(kl,  session="regular", warmup_minutes=30)
        frozen_after  = _session_va(kl2, session="regular", warmup_minutes=30)
        assert float(frozen_after["vah"][-1]) == float(frozen_before["vah"][-1])

    def test_inside_warmup_returns_nan(self):
        out = _session_va(_session_klines(n=10), session="regular", warmup_minutes=30)
        assert np.all(np.isnan(out["dist_vah_pct"]))

    def test_rolling_works_inside_warmup_window(self):
        """warmup_minutes=0 has no warmup to wait out."""
        out = _session_va(_session_klines(n=10), session="regular", warmup_minutes=0)
        assert not np.isnan(out["dist_vah_pct"][-1])


class TestSessionVAGuards:
    def test_other_session_idles(self):
        out = _session_va(_session_klines(), session="premarket", warmup_minutes=30)
        assert np.all(np.isnan(out["dist_vah_pct"]))

    def test_no_session_param_follows_active_session(self):
        out = _session_va(_session_klines(), warmup_minutes=30)
        assert not np.isnan(out["dist_vah_pct"][-1])

    def test_empty_frame(self):
        out = _session_va(_session_klines().iloc[:0], session="regular")
        assert set(out) == set(_FIELDS)
        assert all(len(v) == 0 for v in out.values())

    def test_flat_price_range_returns_nan(self):
        """Degenerate profile (compute_volume_profile returns None)."""
        kl = _session_klines()
        for col in ("open", "high", "low", "close"):
            kl[col] = 100.0
        out = _session_va(kl, session="regular", warmup_minutes=30)
        assert np.all(np.isnan(out["dist_vah_pct"]))

    def test_outside_every_session_returns_nan(self):
        """03:00 is inside the overnight session; 'regular' must not match."""
        kl = _session_klines(start="2026-09-09 02:00")
        out = _session_va(kl, session="regular", warmup_minutes=30)
        assert np.all(np.isnan(out["dist_vah_pct"]))
