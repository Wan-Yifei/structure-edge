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
