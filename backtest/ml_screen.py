"""Is there structure here that a linear model cannot reach?

Three models on one feature set, one walk-forward, one way of scoring:

    base    OLS on dr1 and x                       the two that already work
    inter   base + hand-built interaction terms    cheap nonlinearity
    gbm     gradient-boosted trees, all features   nonlinearity + interactions

A GRU is deliberately NOT here yet. The binding constraint is the sample,
not the architecture: ~20,000 rows that overlap so heavily their effective
count is closer to the 60 sessions, against a network with thousands of
parameters. If trees -- far better suited to this size, and able to find
any interaction a GRU could -- cannot beat two linear terms, a sequence
model will not either, and the install is saved. If they can, it earns the
next step.

SCORED THE ONLY WAY THAT MATTERS FOR TRADING. Not R-squared, not AUC: a
model at AUC 0.56 can be entirely untradeable. Each model's predictions
are ranked, the top slice is taken, and the actual net R of those bars is
compared against the validated rule AT THE SAME TRADE COUNT. Fewer better
trades and more worse ones are not comparable until the counts match.

NO LOOKAHEAD ANYWHERE. Each fold trains only on days before its test
block, and the cut-off that defines "top slice" is read off the TRAINING
predictions -- taking a quantile of the test predictions would let the
test set choose its own threshold.

Usage:
    uv run python -m backtest.ml_screen
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).parent.parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(_ROOT))

from backtest.absorption import outcomes                    # noqa: E402
from backtest.delta_atr_matrix import (CODE, END, MIN_SIGNED,  # noqa
                                       SPREAD, START, STOP, atr)
from backtest.walk_forward import folds                     # noqa: E402

WARMUP = 60
rng = np.random.default_rng(1013)

LEVELS = [("val", "VAL"), ("poc", "POC"), ("vah", "VAH"),
          ("ndlo", "筹码峰下界"), ("ndpk", "筹码峰峰值"), ("ndhi", "筹码峰上界")]

FEATS = (["dr1", "dr_prev", "flip", "x", "xmag", "pos", "prev_ret",
          "atr", "hour"]
         + [f"d_{k}" for k, _ in LEVELS]
         + [f"{k}_{s}" for k, _ in LEVELS
            for s in ("reclaim", "break", "approach")])


def build():
    from feeds import delta_bars
    from feeds.fetcher import fetch_klines
    from strategy.session_vp.profile import (bins_for_price,
                                             compute_value_area, find_nodes)
    from strategy.smc.fvg import compute_volume_profile

    k = fetch_klines(CODE, "1m", START, END)
    k["t"] = pd.to_datetime(k.time_key)
    k = k.sort_values("t").drop_duplicates("t").set_index("t")
    dl = delta_bars.load(CODE, START, END)
    k = k.join(dl[["buy", "sell"]], how="left")
    k[["buy", "sell"]] = k[["buy", "sell"]].fillna(0.0)

    parts = []
    for day, g in k.groupby(k.index.normalize()):
        hh = g.index.hour + g.index.minute / 60
        r = g[(hh > 9.5) & (hh <= 16.0)]
        if len(r) < 380:
            continue
        h, l, c = (r.high.to_numpy(float), r.low.to_numpy(float),
                   r.close.to_numpy(float))
        a = atr(h, l, c, 14)
        buy, sell = r.buy.to_numpy(float), r.sell.to_numpy(float)
        tot = buy + sell
        ok = tot >= MIN_SIGNED
        with np.errstate(invalid="ignore", divide="ignore"):
            dr = np.where(ok, (buy - sell) / tot, np.nan)
        tot_s = pd.Series(np.where(ok, tot, np.nan))
        scale = tot_s.rolling(60, min_periods=20).median().shift(1)
        x = (pd.Series(np.where(ok, buy - sell, np.nan)) / scale).to_numpy()

        n = len(r)
        lv = {key: np.full(n, np.nan) for key, _ in LEVELS}
        for i in range(WARMUP, n):
            up = r.iloc[:i + 1]
            nb = bins_for_price(float(up.low.min()), float(up.high.max()))
            if not nb:
                continue
            edges, vols = compute_volume_profile(up, n_bins=nb)
            if edges is None:
                continue
            va = compute_value_area(edges, vols, va_pct=0.70)
            lv["poc"][i], lv["val"][i], lv["vah"][i] = (
                va["poc"], va["val"], va["vah"])
            nodes = find_nodes(edges, vols)
            if not nodes:
                continue
            px = c[i]
            inside = [nd for nd in nodes if nd.lo <= px <= nd.hi]
            pick = (inside[0] if inside
                    else min(nodes, key=lambda z: abs(z.peak - px)))
            lv["ndlo"][i], lv["ndpk"][i], lv["ndhi"][i] = (
                pick.lo, pick.peak, pick.hi)

        d = pd.DataFrame(index=r.index)
        d["day"] = str(day.date())
        d["hour"] = r.index.hour
        d["atr"] = a
        d["dr1"] = dr
        d["x"] = x
        d["xmag"] = np.abs(x)
        rng_bar = np.where(h > l, h - l, np.nan)
        d["pos"] = (c - l) / rng_bar
        prev = np.full(n, np.nan)
        for i in range(5, n):
            if np.isfinite(a[i]) and a[i] > 0:
                prev[i] = (c[i] - c[i - 5]) / a[i]
        d["prev_ret"] = prev
        prev_c = np.r_[np.nan, c[:-1]]
        for key, _ in LEVELS:
            L = lv[key]
            with np.errstate(invalid="ignore", divide="ignore"):
                d[f"d_{key}"] = (c - L) / a
            d[f"{key}_reclaim"] = ((prev_c > L) & (l <= L) & (c > L)
                                   ).astype(float)
            d[f"{key}_break"] = ((prev_c <= L) & (c > L)).astype(float)
            d[f"{key}_approach"] = ((prev_c < L) & (h >= L) & (c <= L)
                                    ).astype(float)
        d["R"] = outcomes(h, l, c, a, +1)
        parts.append(d)
    return pd.concat(parts)


# ── models ───────────────────────────────────────────────────────────────────

def ols_fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def mat(df, cols):
    return np.column_stack([np.ones(len(df))]
                           + [df[c].to_numpy(float) for c in cols])


# The flip -- whether the previous bar carried the opposite sign -- is the
# one-bar memory the Markov test found, and the only memory it found. It
# was missing from the first run of this screen, which made "the models
# cannot beat two variables" a claim about a rule the models were never
# shown.
BASE = ["dr1", "x", "flip"]
INTER = BASE + ["xmag", "pos", "prev_ret", "dr_prev",
                "dr1_x", "dr1_pos", "x_pos", "dr1_flip", "dr1_val",
                "dr1_ndlo"]


def add_inter(df):
    df = df.copy()
    df["dr1_x"] = df.dr1 * df.x
    df["dr1_flip"] = df.dr1 * df.flip
    df["dr1_pos"] = df.dr1 * df.pos
    df["x_pos"] = df.x * df.pos
    df["dr1_val"] = df.dr1 * df.val_reclaim
    df["dr1_ndlo"] = df.dr1 * df.ndlo_approach
    return df


def run_fold(tr, te):
    """Returns {model: predictions on te} plus the training predictions."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    out, cut = {}, {}
    tr_i, te_i = add_inter(tr), add_inter(te)
    for name, cols, frames in (("base", BASE, (tr, te)),
                               ("inter", INTER, (tr_i, te_i))):
        a, b = frames
        beta = ols_fit(mat(a, cols), a.R.to_numpy())
        out[name] = mat(b, cols) @ beta
        cut[name] = mat(a, cols) @ beta

    g = HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.05,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    g.fit(tr[FEATS].to_numpy(float), tr.R.to_numpy())
    out["gbm"] = g.predict(te[FEATS].to_numpy(float))
    cut["gbm"] = g.predict(tr[FEATS].to_numpy(float))
    return out, cut


def net_of(sub):
    if sub.empty:
        return np.nan
    g = sub.groupby("day").agg(n=("R", "size"), sr=("R", "sum"))
    n, sr = g.n.to_numpy(float), g.sr.to_numpy()
    return sr.sum() / n.sum() if n.sum() else np.nan


def ci_of(sub, reps=3000):
    if len(sub) < 50:
        return np.nan, np.nan
    g = sub.groupby("day").agg(n=("R", "size"), sr=("R", "sum"))
    n, sr = g.n.to_numpy(float), g.sr.to_numpy()
    if len(n) < 5:
        return np.nan, np.nan
    i = rng.integers(0, len(n), size=(reps, len(n)))
    t = n[i].sum(1)
    ok = t > 0
    v = sr[i].sum(1)[ok] / t[ok]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    d = build().sort_index()
    d["dr_prev"] = d.groupby("day")["dr1"].shift(1)
    d["flip"] = ((d.dr_prev < 0) & (d.dr1 >= 0)).astype(float)
    d = d.dropna(subset=["R"] + FEATS)
    days = sorted(d.day.unique())
    fl = folds(days, anchored=True)
    print(f"{len(d):,} 根特征齐全的 bar / {len(days)} 个交易日，{len(fl)} 折")
    print("只做多。止损 1 ATR，目标 1.5 ATR，30 根时间止损，价差已扣\n")

    pred = {m: [] for m in ("base", "inter", "gbm")}
    keep = []
    for tr_days, te_days in fl:
        tr = d[d.day.isin(tr_days)]
        te = d[d.day.isin(te_days)]
        if tr.empty or te.empty:
            continue
        out, cut = run_fold(tr, te)
        keep.append(te)
        for m in pred:
            # The cut-off comes from TRAINING predictions: letting the test
            # set pick its own quantile would be the test choosing its own
            # exam. Two slices, matched to the two validated rules.
            pred[m].append(pd.DataFrame({
                "p": out[m],
                "q17": out[m] >= np.nanquantile(cut[m], 1 - 0.17),
                "q49": out[m] >= np.nanquantile(cut[m], 1 - 0.49),
            }, index=te.index))
    te_all = pd.concat(keep)
    base_n = len(te_all)

    print(f"{'':<26}{'笔数':>8}{'占比':>7}{'胜率':>8}{'每笔 R':>9}"
          f"{'95% 区间':>22}")

    def line(label, sub):
        if len(sub) < 50:
            print(f"{label:<26}{len(sub):>8}   样本太少")
            return
        lo, hi = ci_of(sub)
        star = "  *" if lo > 0 else ""
        print(f"{label:<26}{len(sub):>8}{len(sub)/base_n:>7.1%}"
              f"{100*(sub.R > 0).mean():>8.1f}%{net_of(sub):>+9.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]{star}")

    line("全部 bar（无规则）", te_all)
    print("\n  —— 约 17% 的笔数，对标 dr1≥0.4 ——")
    line("  规则 dr1≥0.4", te_all[te_all.dr1 >= 0.4])
    for m in ("base", "inter", "gbm"):
        sel = pd.concat([p.q17 for p in pred[m]])
        line(f"  模型 {m}", te_all[sel.reindex(te_all.index).fillna(False)])

    print("\n  —— 约 49% 的笔数，对标 dr1≥0 ——")
    line("  规则 dr1≥0", te_all[te_all.dr1 >= 0])
    for m in ("base", "inter", "gbm"):
        sel = pd.concat([p.q49 for p in pred[m]])
        line(f"  模型 {m}", te_all[sel.reindex(te_all.index).fillna(False)])

    print("\n  * = 95% 区间（按交易日重采样）整个在 0 以上")
    print("  模型要赢，得在同等笔数下明显高过那一行规则——否则多出来的")
    print("  复杂度没有换到任何东西。")


if __name__ == "__main__":
    main()
