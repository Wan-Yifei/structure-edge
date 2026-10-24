"""Regenerate rates_c{0,1}.js from the question set they describe.

These two files hold the base rates the drill page quotes beside each
question -- "n=186, to +2ATR 47%, 95% [40%, 54%]". They had no generator:
they were produced once, by hand, and so went stale while the question set
grew. On 2026-10-20 the page was quoting 101 sessions and 713 questions
against a set that held 107 sessions and 761, after a backfill doubled it.

Computed from the question file itself rather than from the tick store, so
the base rate describes exactly the population the questions are drawn
from and cannot drift from it again. That makes it descriptive, not
predictive, which is what the page says it is: "同类位置，不是对这一笔的
预测".

The rate throughout is P(the 2:1 barrier is reached first), the `lab`
field, because that is the number the page prints. lab11 is used only for
the unweighted-entry figures.

    uv run python -m strategy.drill_rates --confirm 1
    uv run python -m strategy.drill_rates --all
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

_ROOT = pathlib.Path(__file__).parent.parent
DRILL = _ROOT / "strategy" / "drill"

# Mirrors DR_DECISION in strategy/drill/index.html. The two have to agree:
# the page decides the colour, this file counts how that colour has done.
DR_DECISION = 0.4


def light_of(q: dict) -> str | None:
    """Which rule light a question would show. None when the bar has too
    little aggressor volume for delta_r to exist.

    Green wants a flip behind the move, not a flip on the bar itself: 0.4
    reached one bar after the turn measured the same as 0.4 on the turn
    (40.0% vs 40.0% on c1), while 0.4 with no turn behind it measured the
    set's own base rate.
    """
    s = q.get("drs1")
    m = q.get("mark")
    if not s or m is None:
        return None
    now = s[m]
    if now is None:
        return None
    if now < 0:
        return "red"
    p1 = s[m - 1] if m >= 1 else None
    p2 = s[m - 2] if m >= 2 else None
    flip = (p1 is not None and p1 < 0) or (
        p1 is not None and p2 is not None
        and p2 < 0 and 0 <= p1 < DR_DECISION)
    return "green" if (now >= DR_DECISION and flip) else "amber"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Score interval. The page prints these next to small n, where the
    normal approximation runs past 0 and 1."""
    if not n:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def stat(qs: list[dict], key: str = "lab") -> list | None:
    """[n, rate, lo, hi], rounded the way the existing files are."""
    v = [q[key] for q in qs if q.get(key) is not None]
    if not v:
        return None
    k = sum(1 for x in v if x > 0)
    lo, hi = wilson(k, len(v))
    return [len(v), round(k / len(v), 4), round(lo, 4), round(hi, 4)]


def read_set(path: pathlib.Path) -> list[dict]:
    t = path.read_text(encoding="utf-8")
    return json.loads(t[t.index("=") + 1:].rstrip().rstrip(";"))


def build(qs: list[dict], confirm: int, code: str) -> dict:
    labelled = [q for q in qs if q.get("lab") is not None]
    days = sorted({q["day"] for q in labelled})

    # Two spaces before delta_r: the page builds this key by string
    # concatenation and a single space would silently miss every bucket.
    buckets: dict[str, list] = {"_all": stat(labelled)}
    for kind in sorted({q["kind"] for q in labelled}):
        same = [q for q in labelled if q["kind"] == kind]
        buckets[kind] = stat(same)
        for sign, name in ((True, "≥ 0"), (False, "< 0")):
            sub = [q for q in same if q.get("dr1") is not None
                   and ((q["dr1"] >= 0) == sign)]
            s = stat(sub)
            if s:
                buckets[f"{kind}  delta_r {name}"] = s

    pa: dict[str, list] = {}
    for call in sorted({q.get("call") for q in labelled if q.get("call")}):
        s = stat([q for q in labelled if q.get("call") == call])
        if s:
            pa[call] = s
    pa["_all"] = stat(labelled)

    # cross carries no interval in the existing files; the page prints the
    # rate alone, so emitting four values here would be dead weight.
    cross: dict[str, list] = {}
    for call in sorted({q.get("call") for q in labelled if q.get("call")}):
        for sign, mark in ((True, "+"), (False, "-")):
            sub = [q for q in labelled
                   if q.get("call") == call and q.get("dr1") is not None
                   and ((q["dr1"] >= 0) == sign)]
            s = stat(sub)
            if s:
                cross[f"{call}|{mark}"] = [s[0], s[1]]

    dr: dict[str, list] = {}
    for col in ("dr1", "dr5", "dr15"):
        for sign, name in ((True, "≥0"), (False, "<0")):
            sub = [q for q in labelled if q.get(col) is not None
                   and ((q[col] >= 0) == sign)]
            s = stat(sub)
            if s:
                dr[f"{col}|{name}"] = s

    # Mean R from entering every question, at each reward ratio: a 1:1 pays
    # +1 or -1 so the mean is 2p-1, a 2:1 pays +2 or -1 so it is 3p-1. The
    # page loads this and currently reads it nowhere; kept so the file shape
    # does not change under it if that changes.
    a = stat(labelled, "lab")
    a11 = stat(labelled, "lab11")
    allin = {}
    if a11:
        allin["1:1"] = round(2 * a11[1] - 1, 4)
    if a:
        allin["2:1"] = round(3 * a[1] - 1, 4)

    # What each rule light has actually paid on this set. Mean R is left
    # to the page as 3p-1: every question that survived generation reached
    # one barrier or the other, so a 2:1 trade pays exactly +2 or -1 and
    # the mean is a function of the rate, not a second measurement.
    light: dict[str, list] = {}
    for name in ("green", "red", "amber"):
        s = stat([q for q in labelled if light_of(q) == name])
        if s:
            light[name] = s

    return {"confirm": confirm, "n": len(labelled), "sessions": len(days),
            "period": [days[0], days[-1]], "code": code,
            "buckets": buckets, "pa": pa, "cross": cross, "dr": dr,
            "allin": allin, "light": light}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--confirm", type=int, choices=(0, 1))
    ap.add_argument("--all", action="store_true", help="both settings")
    ap.add_argument("--code", default="US.SOXL")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.confirm is None and not args.all:
        ap.error("给 --confirm 0/1 或 --all")

    for conf in ((0, 1) if args.all else (args.confirm,)):
        src = DRILL / f"questions_c{conf}.js"
        if not src.exists():
            print(f"c{conf}: 没有 {src.name}，跳过", flush=True)
            continue
        qs = read_set(src)
        data = build(qs, conf, args.code)
        out = DRILL / f"rates_c{conf}.js"
        out.write_text(
            f"window.RATES_c{conf}=" +
            json.dumps(data, ensure_ascii=False, separators=(",", ":")) +
            ";\n", encoding="utf-8")
        b = data["buckets"]["_all"]
        print(f"c{conf}: {len(qs)} 题 → n={data['n']}，"
              f"{data['sessions']} 个交易日 "
              f"{data['period'][0]}..{data['period'][1]}，"
              f"到 +2ATR {b[1]:.1%} [{b[2]:.1%}, {b[3]:.1%}]  → {out.name}",
              flush=True)


if __name__ == "__main__":
    main()
