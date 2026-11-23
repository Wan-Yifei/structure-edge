"""A strike with no GEX must not be drawn as a wall.

The chart fades weaker-ranked walls, so a zero-GEX strike that gets padded
into the top-N arrives looking exactly like a genuine wall that happens to
rank low -- there is no visual channel left to say "we looked here and found
nothing". On a real SOXL chain 100 of 163 strikes carry put_gex == 0, so with
the default of 4 walls per side this was reachable, not theoretical.

These are pure-function tests: no Qt, no OpenD, no market hours.
"""
import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from strategy.option.gex import (          # noqa: E402
    build_gex_stats, compute_gex, top_gex_walls, _plot,
)


def _chain(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """(option_type, strike, gamma, open_interest) -> a chain frame."""
    return pd.DataFrame([
        {"code": f"T{i}", "option_type": t, "option_strike_price": k,
         "option_gamma": g, "option_open_interest": oi,
         "option_delta": 0.5, "option_implied_volatility": 30.0,
         "strike_time": "2026-11-27", "volume": 1.0}
        for i, (t, k, g, oi) in enumerate(rows)
    ])


def _built(rows, spot=100.0):
    df, by_strike = compute_gex(_chain(rows), spot)
    return df, by_strike, build_gex_stats(df, by_strike, spot, 1e6, ["2026-11-27"])


def test_all_zero_chain_yields_no_walls():
    """Nothing on either side carries GEX -> no walls, not $0 walls."""
    rows = [("CALL", 20.0, 0.0, 0), ("CALL", 30.0, 0.0, 0),
            ("PUT", 20.0, 0.0, 0), ("PUT", 30.0, 0.0, 0)]
    _, by_strike, stats = _built(rows)

    assert top_gex_walls(by_strike, 4) == ([], [])
    # None, not NaN: hedge_peak_strike next door uses a NaN sentinel, and the
    # `x == x` guard written for that does not catch None.
    assert stats["call_wall"] is None
    assert stats["put_wall"] is None
    assert stats["call_wall_v"] is None
    assert stats["put_wall_v"] is None


def test_partial_side_returns_fewer_not_padded():
    """Two live call strikes, four requested -> two returned, both non-zero."""
    rows = [("CALL", 90.0, 0.01, 500), ("CALL", 95.0, 0.02, 800),
            ("CALL", 100.0, 0.0, 0), ("CALL", 105.0, 0.0, 0),
            ("PUT", 90.0, 0.03, 900)]
    _, by_strike, _ = _built(rows)

    call_walls, put_walls = top_gex_walls(by_strike, 4)
    assert len(call_walls) == 2, call_walls
    assert all(w["gex"] != 0 for w in call_walls), call_walls
    assert len(put_walls) == 1, put_walls
    assert all(w["gex"] != 0 for w in put_walls), put_walls


def test_a_full_side_still_returns_four():
    """Guards the opposite mistake: filtering so hard that real walls vanish."""
    rows = [("CALL", 90.0 + i, 0.01 * (i + 1), 100 * (i + 1)) for i in range(6)]
    rows += [("PUT", 90.0 + i, 0.01 * (i + 1), 100 * (i + 1)) for i in range(6)]
    _, by_strike, stats = _built(rows)

    call_walls, put_walls = top_gex_walls(by_strike, 4)
    assert len(call_walls) == 4
    assert len(put_walls) == 4
    assert stats["call_wall"] is not None
    assert stats["put_wall"] is not None


def test_strongest_wall_is_the_one_reported():
    """The surviving filter must not change which strike wins."""
    rows = [("CALL", 90.0, 0.01, 100), ("CALL", 95.0, 0.09, 900),
            ("CALL", 100.0, 0.0, 0),
            ("PUT", 90.0, 0.08, 700), ("PUT", 95.0, 0.01, 100)]
    _, by_strike, stats = _built(rows)

    assert stats["call_wall"] == 95.0
    assert stats["put_wall"] == 90.0
    call_walls, _ = top_gex_walls(by_strike, 4)
    assert call_walls[0]["strike"] == 95.0      # strongest first, still


@pytest.mark.parametrize("rows", [
    [("CALL", 20.0, 0.0, 0), ("PUT", 20.0, 0.0, 0)],                 # both dead
    [("CALL", 20.0, 0.01, 50), ("PUT", 20.0, 0.0, 0)],               # put dead
    [("CALL", 20.0, 0.0, 0), ("PUT", 20.0, 0.01, 50)],               # call dead
])
def test_plot_survives_a_missing_wall(tmp_path, rows):
    """A None wall must not reach matplotlib.

    _slot() records every slot it hands out, so feeding it None raises inside
    the label layout rather than at the call site -- the failure this pins is
    a crash several frames deep, not a wrong number.
    """
    df, by_strike, stats = _built(rows, spot=20.0)
    out = tmp_path / "gex.png"
    _plot("US.TEST", by_strike, stats, str(out))
    assert out.exists() and out.stat().st_size > 0
