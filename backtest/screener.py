"""Strategy fitness screener.

Fetches HTF klines for a basket of stocks and computes market-structure
features that predict whether the SMC strategy is likely to work well.

Key features
------------
fvg_fill_rate     % of detected FVGs that eventually get filled (higher = price
                  reliably returns to retest gaps — the core of this strategy).
fvg_freq_per100   FVGs per 100 HTF bars (signal density).
mean_fvg_width    Mean FVG width as % of price (gap quality).
kd_clarity_pct    % of HTF bars where |smooth_width| / ATR >= 0.036 (the p25
                  threshold), i.e. bars inside a clearly directional segment.
atr_pct           Mean ATR / close × 100 (normalised volatility).

Scoring
-------
Weighted sum normalised to 0–100.  Higher = more suitable for the strategy.
Weights are calibrated so that SNDK scores near the top.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from feeds.fetcher import fetch_klines
from strategy.smc import detect_fvg
from strategy.smc.kd_trend import compute_kd

_DEFAULT_CFG   = pathlib.Path(__file__).parent.parent / "config" / "schedule.json"
_DEFAULT_START = "2025-01-01"
_DEFAULT_END   = datetime.today().strftime("%Y-%m-%d")
_HTF           = "15m"
_KD_FAST, _KD_SLOW, _KD_SMOOTH, _KD_ATR_PERIOD = 15, 60, 3, 14
_KD_ATR_THR    = 0.036   # p25 threshold — same as kd_trend() default

# Scoring weights (must sum to 1.0)
_W = {
    "fvg_bounce_rate":  0.45,
    "kd_clarity_pct":   0.25,
    "atr_pct":          0.20,
    "fvg_freq_per100":  0.10,
}

_FVG_LOOKAHEAD = 30   # bars to scan after FVG formation


# ── Feature computation ───────────────────────────────────────────────────────

def _fvg_touch_stats(
    htf: pd.DataFrame,
    fvgs: list[dict],
    lookahead: int = _FVG_LOOKAHEAD,
) -> tuple[float, float, float]:
    """Scan bars after each FVG to measure touch/bounce/overfill rates.

    Returns (touch_rate, bounce_rate, overfill_rate) as fractions of all FVGs.
    - touch_rate:    wick entered the zone within `lookahead` bars
    - bounce_rate:   of touched FVGs, close stayed on the trend side (good entry)
    - overfill_rate: of touched FVGs, close crossed through the far side (bad)
    """
    lows   = htf["low"].values.astype(float)
    highs  = htf["high"].values.astype(float)
    closes = htf["close"].values.astype(float)
    n      = len(htf)

    n_touched = n_bounce = n_overfill = 0

    for f in fvgs:
        idx  = f["idx"]
        bull = f["direction"] == "bull"
        bot, top = f["bottom"], f["top"]

        for j in range(idx + 1, min(idx + 1 + lookahead, n)):
            wick = lows[j] if bull else highs[j]
            # Check if wick entered the zone
            entered = (wick <= top) if bull else (wick >= bot)
            if not entered:
                continue

            n_touched += 1
            cls = closes[j]
            if bull:
                if cls >= bot:
                    n_bounce += 1    # closed above FVG bottom → bounce
                else:
                    n_overfill += 1  # closed below FVG bottom → over-fill
            else:
                if cls <= top:
                    n_bounce += 1    # closed below FVG top → bounce
                else:
                    n_overfill += 1  # closed above FVG top → over-fill
            break  # only count first touch per FVG

    n_fvgs = len(fvgs)
    if n_fvgs == 0:
        return 0.0, 0.0, 0.0
    touch_rate    = n_touched  / n_fvgs
    bounce_rate   = n_bounce   / n_touched if n_touched > 0 else 0.0
    overfill_rate = n_overfill / n_touched if n_touched > 0 else 0.0
    return touch_rate, bounce_rate, overfill_rate


def _compute_features(htf: pd.DataFrame, code: str) -> dict:
    if len(htf) < 100:
        return {"code": code, "n_bars": len(htf), "error": "too few bars"}

    # FVG features
    fvgs   = detect_fvg(htf, min_gap_pct=0.001)
    n_fvgs = len(fvgs)
    if n_fvgs > 0:
        fvg_freq       = n_fvgs / len(htf) * 100
        mean_fvg_width = float(np.mean([(f["top"] - f["bottom"]) / f["bottom"] * 100
                                         for f in fvgs]))
        touch_rate, bounce_rate, overfill_rate = _fvg_touch_stats(htf, fvgs)
    else:
        fvg_freq = mean_fvg_width = 0.0
        touch_rate = bounce_rate = overfill_rate = 0.0

    # KD features
    kd       = compute_kd(htf, fast=_KD_FAST, slow=_KD_SLOW, atr_period=_KD_ATR_PERIOD)
    smooth_w = kd["width"].rolling(_KD_SMOOTH, min_periods=1).mean()
    atr_s    = kd["atr"].replace(0.0, np.nan)
    ratio    = smooth_w.abs() / atr_s
    kd_clarity = float((ratio >= _KD_ATR_THR).mean() * 100)
    atr_pct    = float((atr_s / htf["close"]).mean() * 100)

    # Daily dollar volume (avg across trading days, in millions)
    htf2 = htf.copy()
    htf2["_date"] = pd.to_datetime(htf2["time_key"]).dt.date
    htf2["_dv"]   = htf2["close"].astype(float) * htf2["volume"].astype(float)
    avg_dollar_vol_m = float(htf2.groupby("_date")["_dv"].sum().mean()) / 1e6

    return {
        "code":               code,
        "n_bars":             len(htf),
        "fvg_touch_rate":     round(touch_rate * 100, 1),
        "fvg_bounce_rate":    round(bounce_rate * 100, 1),
        "fvg_overfill_rate":  round(overfill_rate * 100, 1),
        "fvg_freq_per100":    round(fvg_freq, 2),
        "mean_fvg_width_pct": round(mean_fvg_width, 3),
        "kd_clarity_pct":     round(kd_clarity, 1),
        "atr_pct":            round(atr_pct, 3),
        "avg_dollar_vol_m":   round(avg_dollar_vol_m, 1),
    }


def _score(df: pd.DataFrame) -> pd.Series:
    """Normalise each feature to [0, 1] then apply weights."""
    scores = pd.Series(0.0, index=df.index)

    def _norm(col: pd.Series) -> pd.Series:
        lo, hi = col.min(), col.max()
        return (col - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=col.index)

    # Higher bounce rate → better (price respects the FVG and reverses)
    scores += _W["fvg_bounce_rate"] * _norm(df["fvg_bounce_rate"])

    # KD clarity: moderate is best — peaks around 40-60%
    clarity_norm = 1.0 - (df["kd_clarity_pct"] - 50.0).abs() / 50.0
    scores += _W["kd_clarity_pct"] * clarity_norm.clip(0.0, 1.0)

    # ATR%: moderate is best — peaks around 1.0%
    atr_norm = 1.0 - (df["atr_pct"] - 1.0).abs() / 1.0
    scores += _W["atr_pct"] * atr_norm.clip(0.0, 1.0)

    # FVG frequency: higher is better (more opportunities), but normalised
    scores += _W["fvg_freq_per100"] * _norm(df["fvg_freq_per100"])

    return (scores * 100).round(1)


# ── HTML report ───────────────────────────────────────────────────────────────

_CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;background:#0b1120;color:#cdd6f4;margin:0;padding:20px}
h1{color:#5c9cf5;font-size:1.4em;margin-bottom:4px}
.meta{color:#666;font-size:.8em;margin-bottom:20px}
table{border-collapse:collapse;width:100%;font-size:.82em}
th{background:#131f30;color:#5c9cf5;padding:8px 10px;text-align:right;border-bottom:2px solid #1e2d42;white-space:nowrap}
th:first-child,th:nth-child(2){text-align:left}
td{padding:6px 10px;border-bottom:1px solid #1e2d42;text-align:right}
td:first-child,td:nth-child(2){text-align:left}
tr.sndk{background:#131f30}
tr:hover{background:#1a2840}
.score-hi{color:#26a69a;font-weight:bold}
.score-lo{color:#ef5350}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:.75em}
.desc{margin-top:28px;color:#888;font-size:.78em;line-height:1.7em}
.desc b{color:#cdd6f4}
"""

_FEAT_DESCS = {
    "fvg_touch_rate":     ("FVG touch rate (%)",    "% of FVGs where price wick entered the zone within 30 bars.  Higher → gaps are revisited."),
    "fvg_bounce_rate":    ("FVG bounce rate (%)",   "Of touched FVGs: % where close stayed on the trend side (valid bounce).  Higher → FVG acts as real support/resistance.  Key scoring metric."),
    "fvg_overfill_rate":  ("FVG over-fill rate (%)", "Of touched FVGs: % where close crossed through the far side (over-refilling).  Lower is better."),
    "fvg_freq_per100":    ("FVG / 100 bars",        "Fair Value Gaps detected per 100 HTF bars.  More = more entry opportunities."),
    "mean_fvg_width_pct": ("Mean FVG width (%)",    "Average gap size as % of price."),
    "kd_clarity_pct":     ("KD clarity (%)",        "% of bars inside a clearly directional KD segment (|smooth_w|/ATR ≥ 0.036).  Moderate (40-60%) is ideal."),
    "atr_pct":            ("ATR (%)",               "Mean ATR / close × 100.  Normalised volatility.  Sweet spot ~1.0%."),
    "score":              ("Score",                 "Weighted fitness score (0-100).  Weights: bounce_rate 45%, KD clarity 25%, ATR 20%, FVG freq 10%."),
    "avg_dollar_vol_m":   ("Avg Daily DV ($M)",    "Average daily dollar volume (price × volume summed across 15m bars per day, averaged over the period, in millions).  Higher = more liquid / actively traded."),
}


def _cell_color(col: str, val: float) -> str:
    """Return a CSS color string for a table cell based on feature direction."""
    if col == "fvg_bounce_rate":
        pct = val / 100.0
        r = int(239 * (1 - pct) + 38 * pct)
        g = int(83  * (1 - pct) + 166 * pct)
        b = int(80  * (1 - pct) + 154 * pct)
    elif col == "fvg_touch_rate":
        pct = val / 100.0
        r = int(239 * (1 - pct) + 38 * pct)
        g = int(83  * (1 - pct) + 166 * pct)
        b = int(80  * (1 - pct) + 154 * pct)
    elif col == "fvg_overfill_rate":
        pct = val / 100.0  # lower is better → invert
        r = int(38  * (1 - pct) + 239 * pct)
        g = int(166 * (1 - pct) + 83  * pct)
        b = int(154 * (1 - pct) + 80  * pct)
    elif col == "kd_clarity_pct":
        # Peaks at 50, falls off at 0 and 100
        t = 1 - abs(val - 50) / 50
        r = int(239 * (1 - t) + 38 * t)
        g = int(83  * (1 - t) + 166 * t)
        b = int(80  * (1 - t) + 154 * t)
    elif col == "atr_pct":
        # Peaks at 1.0
        t = max(0.0, 1 - abs(val - 1.0))
        r = int(239 * (1 - t) + 38 * t)
        g = int(83  * (1 - t) + 166 * t)
        b = int(80  * (1 - t) + 154 * t)
    elif col == "fvg_freq_per100":
        t = max(0.0, 1 - abs(val - 4.0) / 4.0)
        r = int(239 * (1 - t) + 38 * t)
        g = int(83  * (1 - t) + 166 * t)
        b = int(80  * (1 - t) + 154 * t)
    elif col == "score":
        t = val / 100.0
        r = int(239 * (1 - t) + 38 * t)
        g = int(83  * (1 - t) + 166 * t)
        b = int(80  * (1 - t) + 154 * t)
    elif col == "avg_dollar_vol_m":
        # log scale: $1M→dim, $10M→mid, $100M→bright green
        t = min(1.0, math.log10(max(val, 0.1) + 1) / math.log10(101))
        r = int(239 * (1 - t) + 38 * t)
        g = int(83  * (1 - t) + 166 * t)
        b = int(80  * (1 - t) + 154 * t)
    else:
        return ""
    return f"color:rgb({r},{g},{b})"


def _build_html(df: pd.DataFrame, start: str, end: str) -> str:
    """Render the ranked screener DataFrame as a self-contained HTML string.

    Leaves two placeholder tokens — ``{corr_table}`` and ``{vol_corr_table}``
    — that the caller replaces with the correlation matrix HTML fragments.
    """
    display_cols = ["rank", "code", "n_bars", "score",
                    "fvg_touch_rate", "fvg_bounce_rate", "fvg_overfill_rate",
                    "fvg_freq_per100", "kd_clarity_pct", "atr_pct", "avg_dollar_vol_m"]
    headers = ["#", "Code", "Bars", "Score",
               "Touch%", "Bounce%", "Overfill%",
               "FVG/100bar", "KD clarity%", "ATR%", "AvgDV($M)"]

    rows_html = ""
    for _, row in df.iterrows():
        is_sndk = row["code"] == "US.SNDK"
        tr_cls  = ' class="sndk"' if is_sndk else ""
        cells   = ""
        for col, hdr in zip(display_cols, headers):
            val = row.get(col, "")
            if col == "rank":
                cells += f"<td>{int(val)}</td>"
            elif col == "code":
                label = str(val).replace("US.", "")
                badge = ' <span class="badge" style="background:#1e3a5f;color:#80cbc4">ref</span>' if is_sndk else ""
                cells += f"<td><b>{label}</b>{badge}</td>"
            elif col == "n_bars":
                cells += f"<td>{int(val):,}</td>"
            elif isinstance(val, float):
                style = _cell_color(col, val)
                cells += f'<td style="{style}">{val:.1f}</td>'
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr{tr_cls}>{cells}</tr>\n"

    header_html = "".join(f"<th>{h}</th>" for h in headers)

    desc_rows = "".join(
        f"<tr><td><b>{lbl}</b></td><td>{desc}</td></tr>"
        for col, (lbl, desc) in _FEAT_DESCS.items()
    )

    sndk_row = df[df["code"] == "US.SNDK"]
    sndk_note = ""
    if not sndk_row.empty:
        s = sndk_row.iloc[0]
        sndk_note = (f"<p>SNDK reference: touch={s['fvg_touch_rate']:.1f}%  "
                     f"bounce={s['fvg_bounce_rate']:.1f}%  "
                     f"overfill={s['fvg_overfill_rate']:.1f}%  "
                     f"freq={s['fvg_freq_per100']:.2f}  "
                     f"clarity={s['kd_clarity_pct']:.1f}%  "
                     f"ATR={s['atr_pct']:.3f}%  "
                     f"AvgDV={s.get('avg_dollar_vol_m', float('nan')):.1f}M  "
                     f"score={s['score']:.1f}</p>")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SMC Strategy Screener</title>
<style>{_CSS}</style>
</head><body>
<h1>SMC Strategy Fitness Screener</h1>
<div class="meta">HTF: {_HTF} &nbsp;|&nbsp; {start} – {end} &nbsp;|&nbsp;
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
{sndk_note}
<table>
<thead><tr>{header_html}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
{{corr_table}}
{{vol_corr_table}}
<div class="desc">
<table style="width:auto;font-size:1em">
<tr><th style="text-align:left;width:160px">Feature</th><th style="text-align:left">Description</th></tr>
{desc_rows}
</table>
</div>
</body></html>"""


# ── Correlation matrix ────────────────────────────────────────────────────────

def _compute_corr(htf_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of 15m log returns."""
    returns = {}
    for code, htf in htf_map.items():
        label = code.replace("US.", "")
        s = htf.set_index("time_key")["close"].astype(float)
        returns[label] = np.log(s / s.shift(1)).dropna()

    aligned = pd.DataFrame(returns).dropna()
    return aligned.corr()


def _corr_cell_color(val: float) -> str:
    """Red = high correlation, green/teal = low/negative correlation."""
    t = (val + 1) / 2  # map [-1, 1] → [0, 1]
    r = int(38  * (1 - t) + 239 * t)
    g = int(166 * (1 - t) + 83  * t)
    b = int(154 * (1 - t) + 80  * t)
    return f"color:rgb({r},{g},{b})"


def _compute_vol_corr(htf_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of log(daily dollar volume)."""
    dvols: dict[str, pd.Series] = {}
    for code, htf in htf_map.items():
        label = code.replace("US.", "")
        htf2 = htf.copy()
        htf2["_date"] = pd.to_datetime(htf2["time_key"]).dt.date
        htf2["_dv"]   = htf2["close"].astype(float) * htf2["volume"].astype(float)
        daily_dv = htf2.groupby("_date")["_dv"].sum()
        dvols[label] = np.log(daily_dv.replace(0, np.nan)).dropna()
    aligned = pd.DataFrame(dvols).dropna()
    return aligned.corr()


def _corr_table_html(
    corr: pd.DataFrame,
    title: str = "Return Correlation Matrix (15m log returns)",
    subtitle: str = "Green/teal = low correlation (diversified). Red = high correlation (moves together).",
) -> str:
    labels = list(corr.columns)
    header = "<th></th>" + "".join(f"<th>{l}</th>" for l in labels)
    rows = ""
    for row_label in labels:
        cells = f"<td><b>{row_label}</b></td>"
        for col_label in labels:
            val = corr.loc[row_label, col_label]
            if row_label == col_label:
                cells += "<td style='color:#444'>—</td>"
            else:
                style = _corr_cell_color(val)
                cells += f'<td style="{style}">{val:.2f}</td>'
        rows += f"<tr>{cells}</tr>\n"

    return f"""<h2 style="color:#5c9cf5;font-size:1.1em;margin-top:32px">
{title}</h2>
<p style="color:#888;font-size:.78em">{subtitle}</p>
<table style="font-size:.75em">
<thead><tr>{header}</tr></thead>
<tbody>{rows}</tbody>
</table>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def run_screener(
    codes: list[str],
    start: str,
    end: str,
    out_path: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Fetch HTF klines for each code, compute features, score, and write HTML report.

    Args:
        codes:    List of moomoo stock codes to screen.
        start:    Start date 'YYYY-MM-DD'.
        end:      End date 'YYYY-MM-DD'.
        out_path: Override output HTML path; defaults to a timestamped file in results/.

    Returns:
        DataFrame of scored and ranked results (one row per code).
    """
    records  = []
    htf_map  = {}   # code → DataFrame, kept for correlation computation

    for code in codes:
        print(f"  {code} ...", end=" ", flush=True)
        try:
            htf = fetch_klines(code, _HTF, start, end)
            feat = _compute_features(htf, code)
            htf_map[code] = htf
            print(f"{feat.get('n_bars', '?')} bars")
        except Exception as exc:
            print(f"ERROR: {exc}")
            feat = {"code": code, "n_bars": 0, "error": str(exc)}
        records.append(feat)

    df = pd.DataFrame(records)
    if "error" in df.columns:
        ok = df[df["error"].isna()].copy()
    else:
        ok = df.copy()

    if ok.empty:
        print("No valid results.")
        return df

    ok["score"] = _score(ok)
    ok = ok.sort_values("score", ascending=False).reset_index(drop=True)
    ok.insert(0, "rank", range(1, len(ok) + 1))

    # Correlation matrices
    corr          = _compute_corr(htf_map)
    vol_corr      = _compute_vol_corr(htf_map)
    corr_html     = _corr_table_html(corr)
    vol_corr_html = _corr_table_html(
        vol_corr,
        title="Dollar-Volume Correlation Matrix (log daily DV)",
        subtitle="Stocks with similar trading-activity patterns. Green/teal = less correlated liquidity regime.",
    )

    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = pathlib.Path(__file__).parent / "results" / f"{ts}_screener" / "screener.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = (_build_html(ok, start, end)
            .replace("{corr_table}", corr_html)
            .replace("{vol_corr_table}", vol_corr_html))
    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport saved → {out_path}")

    cols = ["rank", "code", "score", "fvg_touch_rate", "fvg_bounce_rate",
            "fvg_overfill_rate", "fvg_freq_per100", "kd_clarity_pct", "atr_pct",
            "avg_dollar_vol_m", "n_bars"]
    print(ok[[c for c in cols if c in ok.columns]].to_string(index=False))
    return ok


def main() -> None:
    """CLI entry point: parse arguments and run the strategy fitness screener."""
    ap = argparse.ArgumentParser(description="SMC strategy fitness screener")
    ap.add_argument("--config", default=str(_DEFAULT_CFG),
                    help="schedule.json with 'targets' list (default: config/schedule.json)")
    ap.add_argument("--codes", nargs="+", help="Override stock list")
    ap.add_argument("--start", default=_DEFAULT_START)
    ap.add_argument("--end",   default=_DEFAULT_END)
    ap.add_argument("--out",   help="Output HTML path")
    args = ap.parse_args()

    if args.codes:
        codes = args.codes
    else:
        cfg = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        codes = cfg.get("targets", [])

    print(f"Screening {len(codes)} stocks  ({args.start} – {args.end})\n")
    out = pathlib.Path(args.out) if args.out else None
    run_screener(codes, args.start, args.end, out)


if __name__ == "__main__":
    main()
