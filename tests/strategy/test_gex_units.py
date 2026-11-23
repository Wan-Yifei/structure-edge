"""Every GEX number a reader sees is "shares per 1% move", exactly once.

The stored `gex` column is gamma x OI x 100 x spot -- the hedge for a move of
one WHOLE spot. It used to be printed raw and labelled "sh", directly beside
a "% of 20d Avg Vol" column, so the obvious division came out 100x too large
(1,825,346 sh read as 0.8% of average volume; the true figure is 0.008%).

The fix scales at format time only. These tests pin BOTH halves of that:
the stored column must stay in the 100% unit, and the displayed value must
be the 1% one. Either test alone passes under a double-scaling or a
fold-into-compute_gex mistake; together they do not.
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt                     # noqa: E402

from strategy.option.gex import (                   # noqa: E402
    build_gex_stats, compute_gex, _fmt, _per_pct, _wall_str,
)

# One all-CALL strike, so nothing cancels against a put and the identity
# below is exact. gamma 0.05 x OI 100 x 100 shares x spot 50 = 25,000.
GAMMA, OI, SPOT = 0.05, 100, 50.0
RAW_GEX = GAMMA * OI * 100 * SPOT          # 25,000 -- per 100% move
PER_PCT = RAW_GEX * 0.01                   # 250    -- per 1% move


def _one_call_chain() -> pd.DataFrame:
    return pd.DataFrame([{
        "code": "T0", "option_type": "CALL", "option_strike_price": 50.0,
        "option_gamma": GAMMA, "option_open_interest": OI,
        "option_delta": 0.5, "option_implied_volatility": 30.0,
        "strike_time": "2026-11-27", "volume": 1.0,
    }])


def _built():
    df, by_strike = compute_gex(_one_call_chain(), SPOT)
    return df, by_strike, build_gex_stats(df, by_strike, SPOT, 1e6, ["2026-11-27"])


def test_the_stored_column_is_still_per_whole_spot():
    """Guards against folding the 0.01 into compute_gex().

    If it were folded there, the identity test below would still pass -- both
    sides would shrink together -- while every stored value silently changed
    unit. This is the test that notices.
    """
    df, _, _ = _built()
    assert float(df["gex"].iloc[0]) == RAW_GEX


def test_displayed_net_gex_equals_the_hedge_figure():
    """The number being fixed, tied to the one that was already right.

    hedge_total (build_gex_stats) has always been per-1%; on an all-call
    chain the displayed Net GEX must now equal it. Before the fix these two
    sat on the same panel 100x apart.
    """
    _, _, stats = _built()
    assert stats["hedge_total"] == PER_PCT
    assert _per_pct(stats["net_gex"]) == stats["hedge_total"]


def test_the_wall_string_is_scaled_too():
    _, _, stats = _built()
    out = _wall_str(stats, "call")
    assert "sh/1%" in out, out
    assert _fmt(PER_PCT) in out, out
    assert _fmt(RAW_GEX) not in out, out          # the unscaled value is gone


def test_both_axis_formatters_scale():
    """The tick labels are what a reader actually reads magnitudes off.

    They were the easiest place to miss, because they scale the axis rather
    than any named value.
    """
    from strategy.option.gex import _plot
    df, by_strike, stats = _built()
    _plot("US.TEST", by_strike, stats, None)
    fig = plt.gcf()
    ax = fig.axes[0]
    fmt = ax.yaxis.get_major_formatter()
    assert fmt(1_825_346, None) == _fmt(18_253.46), fmt(1_825_346, None)
    # the right-hand cumulative axis shares the same scaling
    twins = [a for a in fig.axes if a is not ax and a.get_ylabel()]
    assert twins, "no right-hand axis found"
    assert twins[0].yaxis.get_major_formatter()(1_825_346, None) == _fmt(18_253.46)
    plt.close(fig)


def test_the_hedge_line_was_not_touched():
    """build_gex_stats scales its own hedge figures; it must not be routed
    through _per_pct as well, which would halve-scale them to 2.5."""
    _, _, stats = _built()
    assert stats["hedge_total"] == PER_PCT
    assert stats["hedge_peak"] == PER_PCT
    assert stats["hedge_peak_strike"] == 50.0


def test_labels_say_which_unit():
    from strategy.option.gex import _plot
    df, by_strike, stats = _built()
    _plot("US.TEST", by_strike, stats, None)
    fig = plt.gcf()
    labels = [a.get_ylabel() for a in fig.axes if a.get_ylabel()]
    assert any("per 1% move" in l for l in labels), labels
    assert not any(l.endswith("(shares)") for l in labels), labels
    plt.close(fig)
