"""Read the daily PowerPoint journal, and crop its screenshots for reading.

The deck is the most complete record there is: one title slide per trading day,
a counts table, then one slide per trade carrying a title of the form

    "3 SOXL, long, open & close, WIN"

and a Chinese note saying which session it was, which profile level was traded
and what went wrong.  `trades` pulls that out.

What the deck alone cannot give is the R/R design -- that lives only as pixels
in the moomoo capture pasted on each slide.  `crop` reduces a ~2000x1440
capture to the three regions that carry data (the timeframe selector, the price
chart with the R/R tool on it, and the capture timestamp in the corner), which
is what makes the small numbers legible without reading a full-size image.  The
numbers themselves are transcribed by hand into data/screens.csv.

The trade index is per day and is **not** chronological: the day session is
grouped by instrument, so slide 6 of 10/08 is the 11:52 SOXL trade while slides
7 and 8 are the SOXS trades that came before it.  The capture timestamp is what
resolves that, which is the other reason to keep it in the crop.
"""

from __future__ import annotations

import pathlib
import re
from datetime import datetime

import pandas as pd
from pptx import Presentation
from PIL import Image

_PICTURE = 13                      # MSO_SHAPE_TYPE.PICTURE
_DAY = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")
# Six September slides read "- SOXL, long, open & close, -": a setup that was
# planned and written up but never entered, which is why those days show more
# planned than executed trades in their summary table. They are kept, with no
# index and no result, so the plan survives even though no fill exists.
_TITLE = re.compile(r"^(\d+|-)\s+(SOXL|SOXS),\s*(\w+),\s*(.*?),\s*(WIN|FAIL|-)\s*$")

# Fractions of the full capture.  Stable across the capture sizes in the deck.
_CHART = (0.0, 0.10, 1.0, 0.66)    # header tail, timeframe row, price chart
_STAMP = (0.78, 0.945, 1.0, 1.0)   # "2026/10/07 23:54:49"
_WIDTH = 1240


def trades(pptx: pathlib.Path) -> pd.DataFrame:
    """One row per trade slide."""
    rows: list[dict] = []
    day = None
    for i, s in enumerate(Presentation(str(pptx)).slides):
        texts = [sh.text_frame.text.strip() for sh in s.shapes
                 if sh.has_text_frame and sh.text_frame.text.strip()]
        if not texts:
            continue
        head = texts[0]
        if _DAY.fullmatch(head):
            day = datetime.strptime(head, "%m/%d/%Y").date()
            continue
        m = _TITLE.match(head.replace("\n", " "))
        if m and day:
            idx, res = m.group(1), m.group(5)
            rows.append({
                "day": day,
                "idx": int(idx) if idx != "-" else None,
                "sym": m.group(2), "pos_dir": m.group(3),
                "result": res if res != "-" else "NOT TAKEN",
                "slide": i + 1,
                "pics": sum(1 for sh in s.shapes if sh.shape_type == _PICTURE),
                "note": (texts[1] if len(texts) > 1 else "").replace("\n", " "),
            })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    # The hand-written index repeats on three days -- 09/04 has two trades
    # numbered 2, 09/21 two numbered 3, 09/23 two 15s and three 21s. Deck order
    # is unambiguous, so `seq` is what everything downstream orders and keys on
    # while `idx` stays as the number actually written on the slide.
    t = t.sort_values(["day", "slide"]).reset_index(drop=True)
    t["seq"] = t.groupby("day").cumcount() + 1
    return t


def extract_images(pptx: pathlib.Path, out: pathlib.Path) -> int:
    """Write every picture as out/slideNN_K.png.  Returns how many."""
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, s in enumerate(Presentation(str(pptx)).slides):
        for k, sh in enumerate(sh for sh in s.shapes if sh.shape_type == _PICTURE):
            (out / f"slide{i + 1:02d}_{k}.{sh.image.ext}").write_bytes(sh.image.blob)
            n += 1
    return n


def crop(src: pathlib.Path, dst: pathlib.Path) -> int:
    """Stack the chart region and the timestamp strip into one small image.

    Every picture is cropped, not just the first on each slide: some slides
    lead with a LiqHm capture and keep the moomoo chart as the second or third.
    """
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("slide*.png")):
        im = Image.open(f).convert("RGB")
        W, H = im.size
        ch = im.crop((int(_CHART[0] * W), int(_CHART[1] * H),
                      int(_CHART[2] * W), int(_CHART[3] * H)))
        st = im.crop((int(_STAMP[0] * W), int(_STAMP[1] * H),
                      int(_STAMP[2] * W), int(_STAMP[3] * H)))
        scale = _WIDTH / ch.width
        ch = ch.resize((_WIDTH, int(ch.height * scale)), Image.LANCZOS)
        st = st.resize((int(st.width * scale * 1.6), int(st.height * scale * 1.6)),
                       Image.LANCZOS)
        out = Image.new("RGB", (_WIDTH, ch.height + st.height + 6), (0, 0, 0))
        out.paste(ch, (0, 0))
        out.paste(st, (_WIDTH - st.width, ch.height + 6))
        out.save(dst / f"{f.stem}.png", optimize=True)
        n += 1
    return n
