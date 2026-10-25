"""The delta labels are pooled, not rebuilt on every render.

Attributing TextItem constructions by call site put 1440 of a default
render's 1453 at _draw_delta: one label per bar, built, added to the
scene, and removed again on the next render. That churn was the +739ms
the per-feature audit charged to `delta` and the 8-11 garbage
collections a render that made every timing jitter.

The labels are now created once and reused. Reuse is only correct if the
items stay in the scene and every mutable thing about them is refreshed,
so these drive the real widget and read the placed items back:

  - a redraw must construct nothing once the pool is warm
  - the pool must still be in the scene afterwards; an item that was
    silently removed would reuse fine and draw nothing
  - a shorter frame must hide the tail, or it shows the previous
    render's numbers at the previous render's prices
  - text, position and fill must all follow the new frame

The fill is checked against _qc(_UP, 180) -- the exact expression the
per-label mkBrush used -- because _UP and _DOWN are the heatmap's gold
and purple, not the obvious green and red, and prebuilding the wrong
two brushes would recolour every label without failing anything else.
"""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

import pyqtgraph as pg                     # noqa: E402
from PyQt6.QtWidgets import QApplication   # noqa: E402

import analysis.trade_viewer_qt as tv      # noqa: E402

N_MAX = 16
LOW0 = 100.0


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="module")
def viewer(qapp):
    v = tv.TradeViewerQt()
    v.resize(1200, 800)
    v.show()
    for _ in range(10):
        qapp.processEvents()
    yield v
    v.close()


@pytest.fixture(autouse=True)
def fresh_pool(viewer):
    """Each test starts with an empty pool, module-scoped widget or not."""
    for item in viewer._delta_items:
        viewer._plot_c.removeItem(item)
    viewer._delta_items.clear()
    viewer._delta_used = 0
    viewer._bk_sig = None
    yield


def frame(n, deltas=None):
    """n bars off one shared series, so bar i is bar i in every frame."""
    t = pd.date_range("2026-10-23 09:31", periods=N_MAX, freq="1min")[:n]
    kl = pd.DataFrame({
        "time_key": t.strftime("%Y-%m-%d %H:%M:%S"),
        "low": LOW0 + np.arange(n) * 0.01,
    })
    if deltas is None:
        deltas = [(i + 1) * 100 for i in range(n)]
    keys = tv.bar_start_keys(kl["time_key"], 1)
    buckets = {}
    for i, bk in enumerate(keys):
        d = deltas[i]
        # Split into buy/sell so the sum the drawer takes is the delta.
        # Integers, as the tick store holds them: the under-1000 branch
        # prints the delta raw, so floats here would read "+100.0".
        buckets[bk] = {LOW0 + i * 0.01: {"buy": int(max(d, 0)),
                                         "sell": int(max(-d, 0))}}
    return kl, buckets


def draw(viewer, n, deltas=None):
    kl, buckets = frame(n, deltas)
    viewer._clear_delta_items()
    viewer._draw_delta(kl, buckets, 1)
    return kl


def count_new(monkeypatch):
    """Counts TextItem constructions until the returned list is read."""
    made = []
    real = pg.TextItem.__init__

    def counting(self, *a, **kw):
        made.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(pg.TextItem, "__init__", counting)
    return made


class TestThePoolIsReused:
    def test_a_redraw_constructs_nothing(self, viewer, monkeypatch):
        draw(viewer, 12)
        assert len(viewer._delta_items) == 12
        made = count_new(monkeypatch)
        draw(viewer, 12)
        assert made == [], f"重绘又新建了 {len(made)} 个 TextItem"

    def test_the_same_objects_come_back(self, viewer):
        draw(viewer, 12)
        first = list(viewer._delta_items)
        draw(viewer, 12)
        assert viewer._delta_items == first
        assert all(a is b for a, b in zip(viewer._delta_items, first))

    def test_it_grows_to_the_high_water_mark_then_stops(self, viewer,
                                                        monkeypatch):
        draw(viewer, 5)
        assert len(viewer._delta_items) == 5
        made = count_new(monkeypatch)
        draw(viewer, 12)
        assert len(made) == 7, "只应补上差额"
        made.clear()
        draw(viewer, 12)
        assert made == []

    def test_the_pool_stays_in_the_scene(self, viewer):
        """A pooled item that was removed would be reused happily and
        draw nothing at all."""
        draw(viewer, 8)
        draw(viewer, 8)
        assert all(it.scene() is not None for it in viewer._delta_items)

    def test_clearing_hides_without_dropping_the_pool(self, viewer):
        draw(viewer, 8)
        viewer._clear_delta_items()
        assert len(viewer._delta_items) == 8
        assert viewer._delta_used == 0
        assert not any(it.isVisible() for it in viewer._delta_items)
        assert all(it.scene() is not None for it in viewer._delta_items)


class TestStaleLabelsAreNotLeftShowing:
    def test_a_shorter_frame_hides_the_tail(self, viewer):
        draw(viewer, 12)
        draw(viewer, 4)
        assert viewer._delta_used == 4
        assert all(it.isVisible() for it in viewer._delta_items[:4])
        assert not any(it.isVisible() for it in viewer._delta_items[4:]), \
            "上一帧的标签还留在图上"

    def test_a_bar_whose_delta_is_zero_does_not_consume_a_slot(self, viewer):
        """Zero-delta bars are skipped, so the pool index is not the bar
        index -- the tail trim has to follow _delta_used, not len(klines)."""
        draw(viewer, 6, deltas=[100, 0, 200, 0, 0, 300])
        assert viewer._delta_used == 3
        assert [it.textItem.toPlainText()
                for it in viewer._delta_items[:3]] == ["+100", "+200", "+300"]


class TestEveryMutableThingIsRefreshed:
    def test_text_follows_the_new_frame(self, viewer):
        draw(viewer, 3, deltas=[1200, 1300, 1400])
        draw(viewer, 3, deltas=[-250, 400, 900])
        assert [it.textItem.toPlainText()
                for it in viewer._delta_items[:3]] == ["-250", "+400", "+900"]

    def test_position_follows_the_new_frame(self, viewer):
        kl = draw(viewer, 5)
        for i, it in enumerate(viewer._delta_items[:5]):
            assert it.pos().x() == pytest.approx(i)
            assert it.pos().y() == pytest.approx(kl["low"].iloc[i] * 0.9997)

    def test_fill_follows_the_sign(self, viewer):
        draw(viewer, 2, deltas=[500, -500])
        up, dn = viewer._delta_items[0], viewer._delta_items[1]
        assert up.fill.color() == tv._qc(tv._UP, 180)
        assert dn.fill.color() == tv._qc(tv._DOWN, 180)

    def test_a_reused_label_repaints_when_the_sign_flips(self, viewer):
        """The fill is an attribute with no setter; assigning it is only
        equivalent to the old per-label brush if it actually changes."""
        draw(viewer, 1, deltas=[500])
        assert viewer._delta_items[0].fill.color() == tv._qc(tv._UP, 180)
        draw(viewer, 1, deltas=[-500])
        assert viewer._delta_items[0].fill.color() == tv._qc(tv._DOWN, 180)

    def test_the_two_brushes_are_the_heatmap_colours(self, viewer):
        """Pinned separately from the drawing path: the brushes are built
        once at import, and green/red would be a plausible wrong guess."""
        assert tv._DELTA_UP_BRUSH.color() == tv._qc("#ffa005", 180)
        assert tv._DELTA_DN_BRUSH.color() == tv._qc("#ab47bc", 180)
