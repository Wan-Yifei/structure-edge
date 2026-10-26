"""A chunked candle picture must draw exactly what a full rebuild draws.

CandlestickItem used to bake all 2641 bars into one QPicture on every
set_data -- 557ms, repeated on every live refresh although only the
final bar had moved. It now bakes complete runs of CHUNK bars once,
keeps the remainder of history in a `part` picture, and rebakes only the
final bar per frame.

The whole value of that is reuse, and the whole risk is reusing
something stale. So the central test here is not that the pieces exist:
it is that a sequence of incremental updates paints the same pixels as
one from-scratch build of the same final data. Pixel equality catches a
chunk that was kept when it should have been dropped, which no amount of
inspecting the chunk list would.

The rest pin the invalidation rules one at a time, by counting bakes:

  - the default (dirty_bars=None) reuses nothing, as before
  - a dirty bar drops its chunk and every later picture
  - a config change (heatmap, colour scheme) drops everything
  - a steady live tick rebakes one bar and nothing else
  - a new bar rebakes `part`, and `part` becomes a chunk at the boundary

Off-exchange prints really do arrive late and land in an older bucket
(doc/ORDER_FLOW_GUIDE.md), so "only the last bar can change" is not a
safe assumption and is not one the item makes -- test_a_late_print_in_an
_old_bar covers exactly that case.
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtGui import QImage, QPainter        # noqa: E402
from PyQt6.QtWidgets import QApplication        # noqa: E402

import analysis.trade_viewer_qt as tv           # noqa: E402
from core.time_utils import candle_start        # noqa: E402

CHUNK = tv.CandlestickItem.CHUNK


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def make(n, seed=7):
    """n 1m bars plus a tick bucket per bar."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-10-23 09:31", periods=n, freq="1min")
    base = 100 + np.cumsum(rng.normal(0, 0.05, n))
    kl = pd.DataFrame({
        "time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
        "open": base, "close": base + rng.normal(0, 0.03, n),
        "high": base + 0.12, "low": base - 0.12,
        "volume": rng.integers(500, 5000, n).astype(float),
    })
    buckets = {}
    for j in range(n):
        bk = candle_start(t[j].to_pydatetime() - timedelta(minutes=1), 1)
        buckets[bk] = {
            float(round(p, 2)): {"buy": int(rng.integers(0, 400)),
                                 "sell": int(rng.integers(0, 400)),
                                 "neutral": int(rng.integers(0, 900))}
            for p in np.linspace(base[j] - 0.1, base[j] + 0.1, 7)
        }
    return kl, buckets


def shot(item, w=520, h=360):
    """Paint the item into an image and return the raw pixels."""
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    r = item.boundingRect()
    if r.width() > 0 and r.height() > 0:
        p.translate(0, h)
        p.scale(w / r.width(), -h / r.height())
        p.translate(-r.x(), -r.y())
    item.paint(p)
    p.end()
    return bytes(img.constBits().asarray(img.sizeInBytes()))


def count_bakes(item, monkeypatch):
    """Records (a, b) of every picture baked until the list is read."""
    baked = []
    real = type(item)._bake

    def counting(self, a, b):
        baked.append((a, b))
        return real(self, a, b)

    monkeypatch.setattr(type(item), "_bake", counting)
    return baked


class TestIncrementalEqualsFullRebuild:
    def test_a_live_tail_paints_what_a_full_build_paints(self, qapp):
        """The test that matters: walk the item forward one bar at a
        time, reusing chunks, and compare with one build of the same
        final frame."""
        n = CHUNK * 2 + 37
        kl, buckets = make(n)

        live = tv.CandlestickItem()
        start = CHUNK + 5
        live.set_data(kl.iloc[:start].reset_index(drop=True), buckets, 1)
        shot(live)
        for k in range(start + 1, n + 1):
            # Every step claims only the newest bar moved, which is what
            # the live path will claim.
            live.set_data(kl.iloc[:k].reset_index(drop=True), buckets, 1,
                          dirty_bars={k - 1})
            shot(live)

        whole = tv.CandlestickItem()
        whole.set_data(kl, buckets, 1)

        assert shot(live) == shot(whole), "增量渲染和整体重建画出来的不一样"

    def test_a_late_print_in_an_old_bar(self, qapp):
        """A print landing in a bar inside a frozen chunk must repaint
        it. Off-exchange prints arrive late, so this is a real case."""
        n = CHUNK * 2 + 10
        kl, buckets = make(n)

        live = tv.CandlestickItem()
        live.set_data(kl, buckets, 1)
        shot(live)

        victim = 7                       # inside chunk 0
        bk = list(buckets)[victim]
        buckets[bk] = dict(buckets[bk])
        buckets[bk][float(round(kl["open"].iloc[victim], 2))] = {
            "buy": 99999, "sell": 0, "neutral": 0}

        live.set_data(kl, buckets, 1, dirty_bars={victim})

        whole = tv.CandlestickItem()
        whole.set_data(kl, buckets, 1)
        assert shot(live) == shot(whole), "旧桶的迟到成交没有重画"

    def test_reporting_the_wrong_dirty_bar_is_visible(self, qapp):
        """Guards the test above: if the pixels were insensitive to that
        bucket, the previous test would pass for free."""
        n = CHUNK * 2 + 10
        kl, buckets = make(n)
        live = tv.CandlestickItem()
        live.set_data(kl, buckets, 1)
        before = shot(live)

        victim = 7
        bk = list(buckets)[victim]
        buckets[bk] = dict(buckets[bk])
        buckets[bk][float(round(kl["open"].iloc[victim], 2))] = {
            "buy": 99999, "sell": 0, "neutral": 0}
        live.set_data(kl, buckets, 1, dirty_bars={n - 1})   # wrong bar
        assert shot(live) == before, "这个改动本来就画不出来，上一条测试是空的"


class TestWhatGetsRebaked:
    def test_the_default_reuses_nothing(self, qapp, monkeypatch):
        kl, buckets = make(CHUNK * 2 + 5)
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1)
        shot(item)
        baked = count_bakes(item, monkeypatch)
        item.set_data(kl, buckets, 1)          # dirty_bars omitted
        shot(item)
        assert len(baked) == 4, "默认应当整幅重建（2 个整块 + part + 末根）"

    def test_a_steady_tick_rebakes_one_bar(self, qapp, monkeypatch):
        kl, buckets = make(CHUNK * 2 + 5)
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1)
        shot(item)
        baked = count_bakes(item, monkeypatch)
        item.set_data(kl, buckets, 1, dirty_bars={len(kl) - 1})
        shot(item)
        n = len(kl)
        assert baked == [(n - 1, n)], f"只该重烤末根，实际 {baked}"

    def test_a_dirty_old_bar_drops_its_chunk_and_the_rest(self, qapp,
                                                          monkeypatch):
        kl, buckets = make(CHUNK * 3 + 5)
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1)
        shot(item)
        baked = count_bakes(item, monkeypatch)
        item.set_data(kl, buckets, 1, dirty_bars={CHUNK + 3})   # chunk 1
        shot(item)
        starts = [a for a, _ in baked]
        assert CHUNK in starts and CHUNK * 2 in starts, "块 1 之后没有重烤"
        assert 0 not in starts, "块 0 不该动"

    def test_a_config_change_drops_everything(self, qapp, monkeypatch):
        kl, buckets = make(CHUNK * 2 + 5)
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1, show_heatmap=True)
        shot(item)
        baked = count_bakes(item, monkeypatch)
        item.set_data(kl, buckets, 1, show_heatmap=False,
                      dirty_bars=set())
        shot(item)
        assert len(baked) == 4, "改了 heatmap 开关还在复用旧图"

    def test_a_different_series_drops_everything(self, qapp, monkeypatch):
        kl, buckets = make(CHUNK * 2 + 5)
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1)
        shot(item)
        other = kl.iloc[10:].reset_index(drop=True)     # window moved
        baked = count_bakes(item, monkeypatch)
        item.set_data(other, buckets, 1, dirty_bars=set())
        shot(item)
        assert any(a == 0 for a, _ in baked), "换了窗口起点还在复用"


class TestChunkBookkeeping:
    def test_part_is_promoted_at_the_boundary(self, qapp):
        kl, buckets = make(CHUNK * 2 + 2)
        item = tv.CandlestickItem()
        # One bar short of the boundary: 1 chunk, part holds the rest.
        item.set_data(kl.iloc[:CHUNK].reset_index(drop=True), buckets, 1)
        shot(item)
        assert len(item._chunks) == 0
        assert item._part_rng == (0, CHUNK - 1)
        # One past it: the first CHUNK bars are now a frozen chunk.
        item.set_data(kl.iloc[:CHUNK + 1].reset_index(drop=True), buckets, 1)
        shot(item)
        assert len(item._chunks) == 1
        assert item._part_rng == (CHUNK, CHUNK)

    def test_a_new_bar_rebakes_part_not_the_chunks(self, qapp, monkeypatch):
        kl, buckets = make(CHUNK * 2 + 20)
        item = tv.CandlestickItem()
        item.set_data(kl.iloc[:-1].reset_index(drop=True), buckets, 1)
        shot(item)
        baked = count_bakes(item, monkeypatch)
        item.set_data(kl, buckets, 1, dirty_bars={len(kl) - 1})
        shot(item)
        n = len(kl)
        assert (CHUNK * 2, n - 1) in baked, "新 K 线出现时 part 没有吸收前一根"
        assert (n - 1, n) in baked
        assert not any(a == 0 for a, _ in baked), "整块被无谓重烤了"

    def test_a_single_bar_frame(self, qapp):
        kl, buckets = make(1)
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1)
        assert shot(item) is not None
        assert item._chunks == [] and item._part is None

    def test_an_empty_frame_paints_nothing(self, qapp):
        item = tv.CandlestickItem()
        item.set_data(pd.DataFrame({"time_key": [], "open": [], "high": [],
                                    "low": [], "close": [], "volume": []}),
                      {}, 1)
        blank = QImage(40, 40, QImage.Format.Format_ARGB32)
        blank.fill(0)
        assert shot(item, 40, 40) == bytes(
            blank.constBits().asarray(blank.sizeInBytes()))


class TestOffScreenChunksAreSkipped:
    """paint() replays only the chunks the view can see.

    The pixel tests above paint through a bare QPainter, where
    viewRect() is None and every chunk replays -- so none of them
    exercises the culling at all. These do.
    """

    @pytest.mark.parametrize("a,b,x0,x1,want", [
        (0, 128, 0.0, 10.0, True),       # view inside the range
        (0, 128, 200.0, 300.0, False),   # view entirely to the right
        (256, 384, 0.0, 10.0, False),    # view entirely to the left
        (0, 128, 127.4, 300.0, True),    # view starts on the last bar
        (0, 128, 127.6, 300.0, False),   # ...and just past it
        (128, 256, -50.0, 127.5, True),  # view ends on the first bar
        (128, 256, -50.0, 127.4, False),  # ...and just short of it
    ])
    def test_the_overlap_test(self, a, b, x0, x1, want):
        assert tv.CandlestickItem._spans(a, b, x0, x1) is want

    def test_culling_does_not_change_the_picture(self, qapp, monkeypatch):
        """Under a real view, a narrow X range must paint exactly what
        an unculled replay of the same range paints."""
        import pyqtgraph as pg

        n = CHUNK * 3 + 20
        kl, buckets = make(n)

        def grab(cull):
            pw = pg.PlotWidget()
            pw.resize(400, 300)
            item = tv.CandlestickItem()
            item.set_data(kl, buckets, 1)
            pw.addItem(item)
            pw.setXRange(CHUNK + 10, CHUNK + 60, padding=0)
            pw.setYRange(float(kl["low"].min()), float(kl["high"].max()),
                         padding=0)
            if not cull:
                monkeypatch.setattr(tv.CandlestickItem, "_spans",
                                    staticmethod(lambda *a: True))
            img = QImage(400, 300, QImage.Format.Format_ARGB32)
            img.fill(0)
            p = QPainter(img)
            pw.render(p)
            p.end()
            monkeypatch.undo()
            return bytes(img.constBits().asarray(img.sizeInBytes()))

        assert grab(True) == grab(False), "剔除改变了画面"

    def test_a_narrow_view_replays_few_chunks(self, qapp):
        """Counting the overlap decisions, since the saving is the whole
        point and the pixels are identical either way."""
        n = CHUNK * 8
        played = [i for i in range(n // CHUNK)
                  if tv.CandlestickItem._spans(i * CHUNK, (i + 1) * CHUNK,
                                               CHUNK * 3.0, CHUNK * 3.0 + 50)]
        assert played == [3], f"窄视野下仍会重放 {played}"


class TestDirectionOutline:
    """With the heatmap on, the body is repainted in buy/sell colours
    and the bar's own direction is shown nowhere. An outline in the
    bull/bear colour puts it back.

    The bars here are built with no wicks (high == body top, low ==
    body bottom), because wicks are drawn in that same colour -- with
    them the colour would be on screen whether or not the outline were
    drawn, and the test would pass for free.
    """

    @staticmethod
    def _flat(bull):
        """One wickless bar, plus a tick bucket that covers its body."""
        o, c = (100.0, 101.0) if bull else (101.0, 100.0)
        t = pd.date_range("2026-10-23 09:31", periods=1, freq="1min")
        kl = pd.DataFrame({
            "time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
            "open": [o], "close": [c],
            "high": [max(o, c)], "low": [min(o, c)],
            "volume": [1000.0]})
        bk = candle_start(t[0].to_pydatetime() - timedelta(minutes=1), 1)
        buckets = {bk: {float(p): {"buy": 100, "sell": 40, "neutral": 10}
                        for p in np.linspace(100.0, 101.0, 6)}}
        return kl, buckets

    @staticmethod
    def _colours(item, w=120, h=120):
        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(0)
        p = QPainter(img)
        r = item.boundingRect()
        p.translate(0, h)
        p.scale(w / r.width(), -h / r.height())
        p.translate(-r.x(), -r.y())
        item.paint(p)
        p.end()
        return {img.pixel(x, y) & 0xFFFFFF
                for x in range(w) for y in range(h)}

    def test_a_bull_body_keeps_its_colour_under_the_heatmap(self, qapp):
        kl, buckets = self._flat(True)
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1, show_heatmap=True)
        assert 0x26A69A in self._colours(item), "涨的方向看不出来了"

    def test_a_bear_body_keeps_its_colour_under_the_heatmap(self, qapp):
        kl, buckets = self._flat(False)
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1, show_heatmap=True)
        assert 0xEF5350 in self._colours(item), "跌的方向看不出来了"

    def test_the_colour_scheme_is_honoured(self, qapp):
        """red_up swaps them, and the outline must swap with them."""
        kl, buckets = self._flat(True)
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1, show_heatmap=True, red_up=True)
        cols = self._colours(item)
        assert 0xEF5350 in cols and 0x26A69A not in cols

    def test_a_bar_with_no_ticks_keeps_its_solid_body(self, qapp):
        """Not an outline test -- it passes with or without one. The
        outline is only drawn where the heatmap took the body over, and
        a bar with no bucket never does, so this pins that such a bar
        still gets a solid body in its own colour."""
        kl, _ = self._flat(True)
        bare = tv.CandlestickItem()
        bare.set_data(kl, {}, 1, show_heatmap=True)
        assert 0x26A69A in self._colours(bare)


class TestTheHeatmapDeltaSwitch:
    """Far out, the heatmap's per-level bins are a smear, so the body
    carries the bar's delta sign instead. The pictures outlive a zoom,
    so the only honest place to decide this is paint() -- and crossing
    the line is the one moment what was baked stops being valid.

    What these pin is the COST: no data is recomputed, and zooming
    within either mode rebakes nothing. The flip itself is one full
    bake, which is the price of not keeping two sets of pictures.
    """

    @staticmethod
    def _frame(n=300):
        from core.time_utils import candle_start
        rng = np.random.default_rng(5)
        c = 100 + np.cumsum(rng.normal(0, 0.05, n))
        t0 = pd.Timestamp("2026-10-26 09:31")
        rows, buckets = [], {}
        for i in range(n):
            end = t0 + timedelta(minutes=i + 1)
            o = c[i - 1] if i else c[0]
            hi, lo = max(o, c[i]) + 0.05, min(o, c[i]) - 0.05
            rows.append((end.strftime("%Y-%m-%d %H:%M"), o, hi, lo, c[i],
                         4000.0))
            bk = candle_start(end - timedelta(minutes=1), 1)
            buckets[bk] = {float(c[i]): {"buy": 300.0, "sell": 200.0,
                                         "neutral": 0.0}}
        kl = pd.DataFrame(rows, columns=["time_key", "open", "high", "low",
                                         "close", "volume"])
        return kl, buckets, rng.normal(0, 500, n)

    @staticmethod
    def _look(item, span, px):
        """Pretend the view shows `span` bars across `px` pixels."""
        from PyQt6.QtCore import QRectF

        class _VB:
            def width(self_inner):
                return px

        item.viewRect = lambda: QRectF(0.0, 0.0, float(span), 1.0)
        item.getViewBox = lambda: _VB()

    def _paint(self, item):
        from PyQt6.QtGui import QPicture
        pic = QPicture()
        p = QPainter(pic)
        item.paint(p)
        p.end()

    def _item(self, qapp, **kw):
        kl, buckets, delta = self._frame()
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1, delta=delta, **kw)
        return item

    def test_far_out_the_body_carries_delta(self, qapp):
        item = self._item(qapp)
        self._look(item, span=300, px=600)        # 2 px per bar
        self._paint(item)
        assert item._far is True

    def test_zoomed_in_the_heatmap_stays(self, qapp):
        item = self._item(qapp)
        self._look(item, span=30, px=600)         # 20 px per bar
        self._paint(item)
        assert item._far is False

    def test_the_line_is_the_constant(self, qapp):
        """Just either side of _HM_MIN_PX, so the threshold cannot drift
        from the one the comment and the tooltip describe."""
        for px_per_bar, want in ((tv._HM_MIN_PX - 0.5, True),
                                 (tv._HM_MIN_PX + 0.5, False)):
            item = self._item(qapp)
            self._look(item, span=100, px=100 * px_per_bar)
            self._paint(item)
            assert item._far is want, f"{px_per_bar} px/bar -> {item._far}"

    def test_zooming_within_a_mode_rebakes_nothing(self, qapp, monkeypatch):
        item = self._item(qapp)
        self._look(item, span=300, px=600)
        self._paint(item)                          # first bake
        n = []
        orig = item._bake
        monkeypatch.setattr(item, "_bake",
                            lambda a, b: (n.append(1), orig(a, b))[1])
        for span in (280, 260, 240):               # still far out
            self._look(item, span=span, px=600)
            self._paint(item)
        assert n == [], f"rebaked {len(n)} picture(s) on a plain zoom"

    def test_crossing_rebakes_every_picture_once(self, qapp, monkeypatch):
        """Counted exactly, not just "more than zero": part and last are
        rebaked whatever happens, so a flip that forgot to drop the
        CHUNKS would still look like it had done something."""
        item = self._item(qapp)
        self._look(item, span=300, px=600)
        self._paint(item)
        expected = len(item._chunks) + (item._part is not None) + 1
        assert len(item._chunks) >= 2, "the frame must span several chunks"
        n = []
        orig = item._bake
        monkeypatch.setattr(item, "_bake",
                            lambda a, b: (n.append(1), orig(a, b))[1])
        self._look(item, span=20, px=600)          # cross into heatmap
        self._paint(item)
        assert len(n) == expected, (
            f"crossing baked {len(n)} of {expected} pictures -- the "
            f"chunks were kept in the wrong mode")
        self._paint(item)                          # same view again
        assert len(n) == expected, "a second paint rebaked again"

    def test_without_delta_nothing_switches(self, qapp):
        """The old behaviour, unchanged: no delta, no second mode."""
        kl, buckets, _ = self._frame()
        item = tv.CandlestickItem()
        item.set_data(kl, buckets, 1)
        self._look(item, span=300, px=600)
        self._paint(item)
        assert item._far is False

    def test_with_the_heatmap_off_nothing_switches(self, qapp):
        """There is nothing to replace, and the body already shows its
        own direction at full strength."""
        item = self._item(qapp, show_heatmap=False)
        self._look(item, span=300, px=600)
        self._paint(item)
        assert item._far is False
