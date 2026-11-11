"""The live K-line push, folded into the chart between fetches.

The viewer has always subscribed the K-line SubType (moomoo requires it
before request_history_kline returns intraday bars) and never consumed it, so
OpenD's bar stream was decoded by the SDK and thrown away while the candles
waited on a 15s poll. Measured on US.SOXL over two minutes: 98 pushes, a
median 0.81s apart, 95 of them the FORMING bar moving rather than a closed
one. The redraw timer already runs at 0.5s, so consuming this costs nothing.

Three things have to hold, and each has tests here:

  * the callback thread never touches self._klines -- the GUI thread renders
    from that frame for ~200ms at a stretch and the fetcher thread holds it as
    its top-up cache, so a write from the subscription thread would race both.
    Pushes are staged; the GUI thread folds them in;
  * which of two observations of one bar is later is decided by VOLUME, not by
    anyone's clock, because volume only grows within a bar. That is what stops
    a fetch landing mid-bar from rolling the forming candle backwards to the
    snapshot it took a second earlier;
  * a bar arriving from the stream has to move the view exactly as a bar
    arriving from a fetch does, or the newest candle walks off the right edge
    and the stale xMax limit clips it.

Nothing here opens a subscription: _ingest_kline is called directly, which is
why it lives on the window rather than inside the handler class.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication        # noqa: E402

import analysis.trade_viewer_qt as tv           # noqa: E402

COLS = ["time_key", "open", "high", "low", "close", "volume", "turnover"]
K1M = str(tv.TIMEFRAME_MAP["1m"][0])
K5M = str(tv.TIMEFRAME_MAP["5m"][0])
K1D = str(tv.TIMEFRAME_MAP["1d"][0])


def frame(n: int = 3, day: bool = False) -> pd.DataFrame:
    """A frame shaped like the one the viewer actually holds.

    volume is FLOAT, as in production: both the push and the history fetch
    read moomoo's hpVolume, a protobuf double, and nothing in DataFetcher
    casts the column down. An int64 fixture made a dtype test pass against a
    condition that does not exist on the live chart.

    turnover is here because the push does not carry every column the fetch
    does -- it is the stand-in for whatever the frame has and the stream
    does not.
    """
    if day:
        keys = [f"2026-11-{9+i:02d} 00:00:00" for i in range(n)]
    else:
        keys = [f"2026-11-11 09:{30+i:02d}:00" for i in range(n)]
    rows = [(k, 54.0, 54.5, 53.8, 54.2, 1000.0 + i, 5.4e4) for i, k in
            enumerate(keys)]
    return pd.DataFrame(rows, columns=COLS)


def push(time_key: str, close: float = 55.0, volume: float = 9999.0,
         *, code: str = "US.SOXL", k_type: str = K1M, **kw) -> dict:
    """One row shaped like CurKlinePush.unpack_rsp builds them."""
    row = {"code": code, "name": "SOXL", "k_type": k_type,
           "time_key": time_key,
           "open": 54.0, "high": 55.5, "low": 53.5, "close": close,
           "volume": volume, "turnover": 6.1e4, "last_close": 53.9,
           "pe_ratio": 0.0, "turnover_rate": 0.0}
    row.update(kw)
    return row


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
    for _ in range(20):
        qapp.processEvents()
    yield v
    v.close()
    qapp.processEvents()


@pytest.fixture()
def live(viewer):
    """A viewer in Live mode holding a 3-bar SOXL 1m frame."""
    viewer._mode_combo.setCurrentText("Live")
    viewer._klines = frame()
    viewer._live_kl_key = ("US.SOXL", "1m")
    viewer._pending_kl = {}
    viewer._last_kl_push = None
    viewer._tick_seq = 0
    viewer._drawn_tick_seq = 0
    yield viewer
    viewer._pending_kl = {}
    viewer._last_kl_push = None
    viewer._live_kl_key = None
    viewer._klines = None


@pytest.fixture()
def painted(live, monkeypatch):
    """Counts repaints and right-edge follows instead of doing either."""
    calls = {"render": 0, "follow": []}
    monkeypatch.setattr(type(live), "_render",
                        lambda self, k, t: calls.__setitem__(
                            "render", calls["render"] + 1))
    monkeypatch.setattr(type(live), "_follow_right_edge",
                        lambda self, p, lo, hi: calls["follow"].append(p))
    return calls


# ── the fold itself ──────────────────────────────────────────────────────────

class TestFoldingOneBar:

    def test_the_forming_bar_is_updated_in_place(self):
        df, grew = tv._fold_pushed_bar(frame(), push("2026-11-11 09:32:00",
                                                     close=61.75, volume=2000))
        assert grew is False
        assert len(df) == 3
        assert df["close"].iloc[-1] == pytest.approx(61.75)
        assert df["volume"].iloc[-1] == 2000

    def test_a_new_bar_is_appended(self):
        df, grew = tv._fold_pushed_bar(frame(), push("2026-11-11 09:33:00"))
        assert grew is True
        assert len(df) == 4
        assert str(df["time_key"].iloc[-1]) == "2026-11-11 09:33:00"

    def test_a_bar_older_than_the_last_is_dropped(self):
        """The stream is about the bar being formed. A real late correction to
        an older bar comes back through _merge_klines on the next fetch."""
        base = frame()
        df, grew = tv._fold_pushed_bar(base, push("2026-11-11 09:30:00",
                                                  close=99.0))
        assert grew is False
        assert df is base

    def test_an_empty_frame_is_left_alone(self):
        empty = pd.DataFrame(columns=COLS)
        df, grew = tv._fold_pushed_bar(empty, push("2026-11-11 09:33:00"))
        assert df is empty and grew is False

    def test_no_frame_at_all(self):
        assert tv._fold_pushed_bar(None, push("2026-11-11 09:33:00")) == (
            None, False)

    def test_an_empty_push_row(self):
        base = frame()
        assert tv._fold_pushed_bar(base, {}) == (base, False)

    def test_a_row_without_a_time_key(self):
        base = frame()
        df, grew = tv._fold_pushed_bar(base, {"close": 9.0, "volume": 5})
        assert df is base and grew is False


class TestVolumeDecidesWhichObservationIsLater:
    """Within one bar volume only grows, so the larger figure was taken later.
    No clock is trusted: not the push's, not the fetch's, not this machine's
    (which free-runs at -39 ms/h anyway)."""

    def test_a_smaller_volume_is_an_older_observation_and_is_dropped(self):
        base = frame()                       # last bar volume 1002
        df, _ = tv._fold_pushed_bar(base, push("2026-11-11 09:32:00",
                                               close=99.0, volume=500))
        assert df is base, "a stale push overwrote a fresher bar"

    def test_an_equal_volume_still_applies(self):
        """A tie is the same instant seen twice; taking the push is harmless
        and keeps the common no-trade-since case from needing a special path."""
        df, _ = tv._fold_pushed_bar(frame(), push("2026-11-11 09:32:00",
                                                  close=61.0, volume=1002))
        assert df["close"].iloc[-1] == pytest.approx(61.0)

    def test_a_non_numeric_volume_is_not_an_error(self):
        base = frame()
        df, grew = tv._fold_pushed_bar(base, push("2026-11-11 09:32:00",
                                                  volume="n/a"))
        assert df is base and grew is False


class TestItNeverMutatesTheCallersFrame:
    """The fetcher thread holds this very frame as its top-up cache while the
    GUI thread folds. An in-place edit here is a cross-thread write."""

    def test_the_input_object_is_untouched_on_an_update(self):
        base = frame()
        before = base.copy()
        tv._fold_pushed_bar(base, push("2026-11-11 09:32:00", close=61.75))
        pd.testing.assert_frame_equal(base, before)

    def test_the_input_object_is_untouched_on_an_append(self):
        base = frame()
        before = base.copy()
        tv._fold_pushed_bar(base, push("2026-11-11 09:33:00"))
        pd.testing.assert_frame_equal(base, before)

    def test_a_changed_frame_is_a_different_object(self):
        base = frame()
        df, _ = tv._fold_pushed_bar(base, push("2026-11-11 09:32:00",
                                               close=61.75))
        assert df is not base

    def test_an_unchanged_frame_is_the_same_object(self):
        """_drain_pushed_klines reads identity to decide whether to repaint."""
        base = frame()
        df, _ = tv._fold_pushed_bar(base, push("2026-11-11 09:00:00"))
        assert df is base

    def test_only_the_last_row_is_written_on_a_duplicate_index(self):
        """Every production path hands this a RangeIndex, so writing by label
        would work today -- and would quietly write every row sharing the
        label the first time one does not."""
        base = frame()
        base.index = [0, 1, 0]
        df, _ = tv._fold_pushed_bar(base, push("2026-11-11 09:32:00",
                                               close=61.75))
        assert list(df["close"]) == [pytest.approx(54.2),
                                     pytest.approx(54.2),
                                     pytest.approx(61.75)]


class TestDtypes:
    """Asserting on the VALUE proves nothing here -- 140.0 == 140 either way.
    These assert on the column, which is what a later concat or a formatted
    readout would trip over."""

    def test_the_frame_keeps_its_dtypes_on_an_update(self):
        base = frame()
        df, _ = tv._fold_pushed_bar(base, push("2026-11-11 09:32:00",
                                               volume=2000.0))
        assert df.dtypes.to_dict() == base.dtypes.to_dict()

    def test_the_frame_keeps_its_dtypes_on_an_append(self):
        """A column the push does not carry must not be filled with None --
        that turns a numeric column into object and changes the schema."""
        base = frame()
        df, _ = tv._fold_pushed_bar(base, push("2026-11-11 09:33:00",
                                               volume=140.0))
        assert df.dtypes.to_dict() == base.dtypes.to_dict()

    def test_a_fractional_volume_is_not_truncated(self):
        """hpVolume is a protobuf double. Casting it to int would silently
        drop the fraction, and the column is float64 anyway."""
        df, _ = tv._fold_pushed_bar(frame(), push("2026-11-11 09:32:00",
                                                  volume=1403.75))
        assert df["volume"].iloc[-1] == pytest.approx(1403.75)

    def test_the_new_bar_carries_only_the_frames_columns(self):
        df, _ = tv._fold_pushed_bar(frame(), push("2026-11-11 09:33:00"))
        assert list(df.columns) == COLS, "push-only fields leaked in"

    def test_a_column_the_push_lacks_is_carried_over(self):
        df, _ = tv._fold_pushed_bar(frame(), push("2026-11-11 09:33:00"))
        assert df["turnover"].iloc[-1] == pytest.approx(5.4e4)


class TestAMalformedPushCannotKillTheApp:
    """These run inside a QTimer slot, and an exception escaping a PyQt6 slot
    aborts the process. A bad push has to cost a dropped bar, not the viewer.
    The first version guarded only the volume COMPARISON, so the append branch
    had no guard at all and the update branch passed by accident."""

    @pytest.mark.parametrize("field,bad", [
        ("volume", float("nan")), ("volume", "N/A"), ("volume", object()),
        ("close", "N/A"), ("open", object()), ("high", "x"), ("low", None),
        # A NaN PRICE is the main reason the finiteness check exists: it does
        # not crash anything, it silently poisons every indicator downstream.
        # Proving the check on volume alone left half of it unproven.
        ("close", float("nan")), ("high", float("inf")),
        ("low", float("-inf")), ("open", float("nan")),
        # float() of a huge Python int raises OverflowError, which is neither
        # TypeError nor ValueError. Unreachable from a protobuf double, but
        # the guard's promise is that a malformed push costs a bar and not
        # the process, and an escape here aborts it.
        ("close", 10 ** 400), ("volume", 10 ** 400),
    ])
    def test_an_uncoercible_update(self, field, bad):
        base = frame()
        row = push("2026-11-11 09:32:00")
        row[field] = bad
        df, grew = tv._fold_pushed_bar(base, row)
        assert grew is False
        pd.testing.assert_frame_equal(base, frame())

    @pytest.mark.parametrize("field,bad", [
        ("volume", float("nan")), ("volume", "N/A"),
        ("close", "N/A"), ("open", object()),
        ("close", float("nan")), ("high", float("inf")),
        ("close", 10 ** 400),
    ])
    def test_an_uncoercible_append(self, field, bad):
        base = frame()
        row = push("2026-11-11 09:33:00")
        row[field] = bad
        df, grew = tv._fold_pushed_bar(base, row)
        assert grew is False
        assert len(df) == 3


class TestDailyBars:
    """The push gives a 1d bar a bare "YYYY-MM-DD"; the fetched frame carries
    "YYYY-MM-DD 00:00:00" because DataFetcher normalises it. Compared raw they
    never match, so on 1d the fold did nothing -- and at a day boundary the
    bare key sorted ABOVE the normalised one and appended a row in a format
    nothing downstream parses."""

    def test_todays_bar_is_recognised(self):
        base = frame(day=True)            # last key 2026-11-11 00:00:00
        df, grew = tv._fold_pushed_bar(base, push("2026-11-11", close=61.75,
                                                  k_type=K1D, volume=5e6))
        assert grew is False
        assert len(df) == 3, "a same-day push appended a duplicate bar"
        assert df["close"].iloc[-1] == pytest.approx(61.75)

    def test_a_new_day_is_appended_normalised(self):
        df, grew = tv._fold_pushed_bar(frame(day=True),
                                       push("2026-11-12", k_type=K1D))
        assert grew is True
        assert str(df["time_key"].iloc[-1]) == "2026-11-12 00:00:00"

    def test_an_intraday_key_is_left_alone(self):
        df, _ = tv._fold_pushed_bar(frame(), push("2026-11-11 09:32:00",
                                                  close=61.75))
        assert str(df["time_key"].iloc[-1]) == "2026-11-11 09:32:00"


# ── staging, on the callback thread ──────────────────────────────────────────

class TestIngest:

    def test_a_push_is_staged_not_applied(self, live):
        before = live._klines
        live._ingest_kline([push("2026-11-11 09:32:00")], "US.SOXL")
        assert live._pending_kl, "nothing was staged"
        assert live._klines is before, "the callback thread touched _klines"

    def test_several_pushes_for_one_bar_collapse_to_the_latest(self, live):
        for v in (100, 200, 300):
            live._ingest_kline([push("2026-11-11 09:32:00", volume=v)],
                               "US.SOXL")
        assert len(live._pending_kl) == 1
        assert live._pending_kl["2026-11-11 09:32:00"]["volume"] == 300

    def test_it_bumps_the_paint_sequence(self, live):
        live._ingest_kline([push("2026-11-11 09:32:00")], "US.SOXL")
        assert live._tick_seq == 1

    def test_a_filtered_push_buys_no_repaint(self, live):
        """Bumping unconditionally would repaint for a push we then drop."""
        live._ingest_kline([push("2026-11-11 09:32:00")], "US.QQQ")
        assert live._tick_seq == 0
        assert live._pending_kl == {}

    def test_an_empty_payload(self, live):
        live._ingest_kline([], "US.SOXL")
        assert live._tick_seq == 0

    def test_a_row_without_a_time_key_is_not_staged(self, live):
        live._ingest_kline([{"close": 1.0}], "US.SOXL")
        assert live._pending_kl == {}
        assert live._tick_seq == 0

    def test_nothing_is_staged_without_a_live_binding(self, live):
        """Historical mode: _live_kl_key is None and a late push must not be
        kept for whatever is shown next."""
        live._live_kl_key = None
        live._ingest_kline([push("2026-11-11 09:32:00")], "US.SOXL")
        assert live._pending_kl == {}


# ── draining, on the GUI thread ──────────────────────────────────────────────

class TestDrain:

    def test_it_folds_and_reports_the_change(self, live):
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75)],
                           "US.SOXL")
        changed, appended = live._drain_pushed_klines()
        assert (changed, appended) == (True, False)
        assert live._klines["close"].iloc[-1] == pytest.approx(61.75)

    def test_a_new_bar_reports_appended(self, live):
        live._ingest_kline([push("2026-11-11 09:33:00")], "US.SOXL")
        assert live._drain_pushed_klines() == (True, True)
        assert len(live._klines) == 4

    def test_the_staging_is_emptied(self, live):
        live._ingest_kline([push("2026-11-11 09:32:00")], "US.SOXL")
        live._drain_pushed_klines()
        assert live._pending_kl == {}

    def test_two_bars_are_folded_oldest_first(self, live):
        live._ingest_kline([push("2026-11-11 09:34:00", close=70.0),
                            push("2026-11-11 09:33:00", close=60.0)],
                           "US.SOXL")
        live._drain_pushed_klines()
        assert len(live._klines) == 5
        assert str(live._klines["time_key"].iloc[-1]) == "2026-11-11 09:34:00"
        assert live._klines["close"].iloc[-1] == pytest.approx(70.0)

    def test_nothing_staged_is_not_a_change(self, live):
        assert live._drain_pushed_klines() == (False, False)

    def test_a_push_for_the_previous_timeframe_is_dropped(self, live):
        """The toolbar switches instantly; a 5m push can still be in flight
        when the frame already belongs to 1m."""
        live._ingest_kline([push("2026-11-11 09:32:00", close=99.0,
                                 k_type=K5M)], "US.SOXL")
        assert live._drain_pushed_klines() == (False, False)
        assert live._klines["close"].iloc[-1] != pytest.approx(99.0)

    def test_a_push_for_another_symbol_is_dropped(self, live):
        live._pending_kl = {"2026-11-11 09:32:00":
                            push("2026-11-11 09:32:00", close=99.0,
                                 code="US.QQQ")}
        assert live._drain_pushed_klines() == (False, False)

    def test_nothing_is_folded_outside_live_mode(self, live):
        """blockSignals, or setCurrentText runs _on_mode_changed, which stops
        live and clears the staging -- and then this passes whether the guard
        is there or not. The first version of this test did exactly that."""
        live._ingest_kline([push("2026-11-11 09:32:00", close=99.0)],
                           "US.SOXL")
        live._mode_combo.blockSignals(True)
        live._mode_combo.setCurrentText("Historical")
        try:
            assert live._drain_pushed_klines() == (False, False)
            assert live._klines["close"].iloc[-1] != pytest.approx(99.0)
        finally:
            live._mode_combo.setCurrentText("Live")
            live._mode_combo.blockSignals(False)

    def test_it_remembers_the_bar_it_folded(self, live):
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75)],
                           "US.SOXL")
        live._drain_pushed_klines()
        key, row = live._last_kl_push
        assert key == ("US.SOXL", "1m")
        assert row["close"] == pytest.approx(61.75)

    def test_a_dropped_push_is_not_remembered(self, live):
        live._ingest_kline([push("2026-11-11 09:32:00", volume=1)], "US.SOXL")
        live._drain_pushed_klines()          # lower volume -> no change
        assert live._last_kl_push is None


# ── the redraw tick ──────────────────────────────────────────────────────────

class TestTheRedrawTick:

    def test_a_pushed_bar_causes_a_repaint(self, live, painted):
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75)],
                           "US.SOXL")
        live._drawn_tick_seq = live._tick_seq      # no NEW ticks, only a bar
        live._on_redraw_tick()
        assert painted["render"] == 1

    def test_nothing_new_means_no_repaint(self, live, painted):
        live._drawn_tick_seq = live._tick_seq
        live._on_redraw_tick()
        assert painted["render"] == 0

    def test_an_appended_bar_follows_the_right_edge(self, live, painted):
        live._ingest_kline([push("2026-11-11 09:33:00")], "US.SOXL")
        live._on_redraw_tick()
        assert painted["follow"] == [3], \
            "the bar count from BEFORE the append must be passed"

    def test_an_updated_bar_does_not_move_the_view(self, live, painted):
        """Only a NEW bar scrolls. Re-centring on every forming-bar update
        would fight the user roughly once a second."""
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75)],
                           "US.SOXL")
        live._on_redraw_tick()
        assert painted["follow"] == []

    def test_no_frame_yet(self, live, painted):
        live._klines = None
        live._on_redraw_tick()
        assert painted["render"] == 0


# ── a fetch landing on top of a pushed bar ───────────────────────────────────

class TestWhenAFetchLands:
    """_on_data_ready replaces the whole frame with a snapshot the fetch took
    a second or two earlier. Without re-applying the push, the live candle
    steps backwards once every Refresh cycle."""

    @staticmethod
    def _result(df, code="US.SOXL", tf="1m", historical=False):
        return {"klines": df, "code": code, "tf": tf, "date_str": "",
                "historical": historical, "ticks": {}, "warmup": None,
                "smc_signals": [], "fvg_gaps": [], "ob_blocks": [],
                "new_tick_days": {}}

    def test_the_forming_bar_is_not_rolled_back(self, live, monkeypatch):
        monkeypatch.setattr(type(live), "_render", lambda self, k, t: None)
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75,
                                 volume=2500)], "US.SOXL")
        live._drain_pushed_klines()

        stale = frame()                      # last bar still at volume 1002
        live._on_data_ready(self._result(stale))

        assert live._klines["close"].iloc[-1] == pytest.approx(61.75)
        assert live._klines["volume"].iloc[-1] == 2500

    def test_a_genuinely_newer_fetch_wins(self, live, monkeypatch):
        monkeypatch.setattr(type(live), "_render", lambda self, k, t: None)
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75,
                                 volume=2500)], "US.SOXL")
        live._drain_pushed_klines()

        fresh = frame()
        fresh.loc[fresh.index[-1], "volume"] = 9000
        fresh.loc[fresh.index[-1], "close"] = 62.5
        live._on_data_ready(self._result(fresh))

        assert live._klines["close"].iloc[-1] == pytest.approx(62.5)

    def test_a_push_from_another_chart_is_discarded(self, live, monkeypatch):
        monkeypatch.setattr(type(live), "_render", lambda self, k, t: None)
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75)],
                           "US.SOXL")
        live._drain_pushed_klines()

        live._on_data_ready(self._result(frame(), code="US.QQQ"))

        assert live._last_kl_push is None
        assert live._klines["close"].iloc[-1] != pytest.approx(61.75)

    def test_a_historical_fetch_discards_it(self, live, monkeypatch):
        monkeypatch.setattr(type(live), "_render", lambda self, k, t: None)
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75)],
                           "US.SOXL")
        live._drain_pushed_klines()

        live._on_data_ready(self._result(frame(), historical=True))

        assert live._last_kl_push is None
        assert live._klines["close"].iloc[-1] != pytest.approx(61.75)


class TestFollowingTheRightEdge:
    """Extracted from _on_data_ready so a bar from the stream moves the view
    exactly as a bar from a fetch does. Driving it only through the redraw
    tick left its own guards unproven -- these call it directly."""

    @pytest.fixture()
    def watched(self, live, monkeypatch):
        seen: dict = {"limits": [], "xrange": []}
        monkeypatch.setattr(live._plot_c.vb, "setLimits",
                            lambda **kw: seen["limits"].append(kw))
        monkeypatch.setattr(live._plot_c, "setXRange",
                            lambda a, b, **kw: seen["xrange"].append((a, b)))
        monkeypatch.setattr(type(live), "_autorange_price_subplots",
                            lambda self, a, b: None)
        monkeypatch.setattr(type(live), "_autorange_cumulative_subplots",
                            lambda self, a, b: None)
        return seen

    def test_it_raises_the_pan_limit_when_the_frame_grew(self, live, watched):
        live._follow_right_edge(2, 0.0, 3.0)      # 3 bars now, 2 before
        assert watched["limits"] == [{"xMax": 8, "maxXRange": 13}]

    def test_it_recentres_on_the_newest_bar(self, live, watched):
        live._follow_right_edge(2, 0.0, 3.0)
        assert watched["xrange"] == [(0.5, 3.5)]

    def test_it_leaves_a_panned_view_alone(self, live, watched):
        """Panning left to read history is deliberate and must survive the
        next bar. The pan LIMIT still has to be raised, or the stale xMax
        clips the newest candle once the user comes back."""
        live._follow_right_edge(2, 0.0, -5.0)
        assert watched["xrange"] == []
        assert watched["limits"], "the limit must still be raised"

    def test_it_does_nothing_when_the_frame_did_not_grow(self, live, watched):
        live._follow_right_edge(3, 0.0, 3.0)      # 3 bars before and after
        assert watched["limits"] == [] and watched["xrange"] == []

    def test_it_does_nothing_from_an_empty_frame(self, live, watched):
        """prev_n == 0 is the first load, where _reset_view owns the view."""
        live._follow_right_edge(0, 0.0, 3.0)
        assert watched["limits"] == []

    def test_it_does_nothing_in_historical(self, live, watched):
        live._mode_combo.blockSignals(True)
        live._mode_combo.setCurrentText("Historical")
        try:
            live._follow_right_edge(2, 0.0, 3.0)
        finally:
            live._mode_combo.setCurrentText("Live")
            live._mode_combo.blockSignals(False)
        assert watched["limits"] == [], \
            "a historical chart got scrolled to its right edge"

    def test_no_frame_at_all(self, live, watched):
        live._klines = None
        live._follow_right_edge(2, 0.0, 3.0)
        assert watched["limits"] == []


class TestTheHandlerItself:
    """The one layer every other test here skipped, and the one that was
    broken: CurKlineHandlerBase.on_recv_rsp returns a DATAFRAME it builds from
    its own col_list, not the list of dicts CurKlinePush.unpack_rsp hands it.
    The first version wrote `if not data`, which raises on a DataFrame; OpenD's
    push thread caught and logged it, so the chart saw nothing and the log got
    a traceback roughly once a second.

    parse_rsp_pb is stubbed, so the SDK's own frame-building code runs for
    real and only the protobuf is faked.
    """

    @pytest.fixture()
    def handler(self, live, monkeypatch):
        seen: list = []
        monkeypatch.setattr(type(live), "_ingest_kline",
                            lambda self, rows, code: seen.append((rows, code)))
        h = tv._KlinePushHandler(live, "US.SOXL")
        h.seen = seen
        return h

    @staticmethod
    def _stub(monkeypatch, content):
        monkeypatch.setattr(tv.CurKlineHandlerBase, "parse_rsp_pb",
                            classmethod(lambda cls, pb: (tv.RET_OK, content)))

    def test_it_delivers_rows_the_viewer_can_read(self, handler, monkeypatch):
        self._stub(monkeypatch, [push("2026-11-11 09:32:00", close=61.75)])
        handler.on_recv_rsp(object())

        assert handler.seen, "the push never reached the viewer"
        rows, code = handler.seen[0]
        assert code == "US.SOXL"
        assert isinstance(rows, list)
        assert isinstance(rows[0], dict), \
            "iterating this would yield column names, not bars"
        assert rows[0]["time_key"] == "2026-11-11 09:32:00"
        assert rows[0]["close"] == pytest.approx(61.75)
        assert rows[0]["k_type"] == K1M

    def test_an_empty_push_is_not_an_error(self, handler, monkeypatch):
        self._stub(monkeypatch, [])
        handler.on_recv_rsp(object())
        assert handler.seen == []

    def test_a_failed_parse_is_not_an_error(self, handler, monkeypatch):
        from moomoo import RET_ERROR
        monkeypatch.setattr(tv.CurKlineHandlerBase, "parse_rsp_pb",
                            classmethod(lambda cls, pb: (RET_ERROR, "boom")))
        handler.on_recv_rsp(object())
        assert handler.seen == []

    def test_two_bars_in_one_message_both_arrive(self, handler, monkeypatch):
        self._stub(monkeypatch, [push("2026-11-11 09:32:00"),
                                 push("2026-11-11 09:33:00")])
        handler.on_recv_rsp(object())
        rows, _ = handler.seen[0]
        assert len(rows) == 2


class TestWiringTheHandlerUp:
    """_start_live is where the handler gets registered, and nothing was
    testing it: deleting that line passed the whole suite. That is the same
    shape as the bug this change already shipped once -- the only untested
    layer was the broken one."""

    class _Ctx:
        def __init__(self):
            self.handlers: list = []
            self.subs: list = []

        def subscribe(self, code, types, **kw):
            self.subs.append((code, list(types)))
            return tv.RET_OK, ""

        def set_handler(self, h):
            self.handlers.append(h)
            return tv.RET_OK

    def _run(self, live, tf: str):
        ctx = self._Ctx()
        was_tf = live._tf_combo.currentText()
        live._ctx = ctx
        live._code_edit.setText("US.SOXL")
        live._tf_combo.blockSignals(True)
        live._tf_combo.setCurrentText(tf)
        try:
            live._start_live()
        finally:
            live._refresh_timer.stop()
            live._redraw_timer.stop()
            # The viewer is module-scoped: leaving the toolbar on "4h" and the
            # subscription fields set would be waiting for whoever adds the
            # next test to this file.
            live._tf_combo.setCurrentText(was_tf)
            live._tf_combo.blockSignals(False)
            live._live_code = ""
            live._live_kl_sub = None
            live._ctx = None
        return ctx

    def test_the_bar_stream_is_consumed(self, live):
        ctx = self._run(live, "1m")
        assert any(isinstance(h, tv._KlinePushHandler) for h in ctx.handlers), \
            "the K-line subscription is paid for and nothing reads it"

    def test_the_handler_is_bound_to_the_subscribed_symbol(self, live):
        ctx = self._run(live, "1m")
        h = next(h for h in ctx.handlers
                 if isinstance(h, tv._KlinePushHandler))
        assert h._code == "US.SOXL"

    def test_the_kline_subtype_is_subscribed(self, live):
        ctx = self._run(live, "1m")
        assert tv.SubType.K_1M in ctx.subs[0][1]

    def test_no_handler_for_a_timeframe_with_no_push_subtype(self, live):
        """4h has no SubType in the SDK, so there is no stream to consume."""
        ctx = self._run(live, "4h")
        assert not any(isinstance(h, tv._KlinePushHandler)
                       for h in ctx.handlers)


class TestTheRememberedPushIsReleased:
    """_last_kl_push is re-applied on every fetch. Held past the bar it
    belongs to, a downward volume correction from the exchange would be
    overwritten once per Refresh cycle with nothing to clear it."""

    @staticmethod
    def _result(df):
        return {"klines": df, "code": "US.SOXL", "tf": "1m", "date_str": "",
                "historical": False, "ticks": {}, "warmup": None,
                "smc_signals": [], "fvg_gaps": [], "ob_blocks": [],
                "new_tick_days": {}}

    def test_it_is_dropped_once_its_bar_is_no_longer_the_last(
            self, live, monkeypatch):
        monkeypatch.setattr(type(live), "_render", lambda self, k, t: None)
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75,
                                 volume=2500.0)], "US.SOXL")
        live._drain_pushed_klines()
        assert live._last_kl_push is not None

        live._on_data_ready(self._result(frame(4)))   # 09:33 is now the last

        assert live._last_kl_push is None, \
            "a pin on a closed bar outlives every later correction"

    def test_an_empty_fetch_releases_it_without_raising(self, live,
                                                        monkeypatch):
        """The empty-frame arm of the release guard. _on_data_ready is a slot,
        and an IndexError from .iloc[-1] there aborts the process. The full
        fetch refuses to emit an empty frame, but the top-up path filters by
        time_key after merging and can in principle filter to nothing."""
        monkeypatch.setattr(type(live), "_render", lambda self, k, t: None)
        live._ingest_kline([push("2026-11-11 09:32:00", volume=2500.0)],
                           "US.SOXL")
        live._drain_pushed_klines()

        live._on_data_ready(self._result(pd.DataFrame(columns=COLS)))

        assert live._last_kl_push is None
        assert len(live._klines) == 0

    def test_it_is_kept_while_its_bar_is_still_forming(self, live, monkeypatch):
        monkeypatch.setattr(type(live), "_render", lambda self, k, t: None)
        live._ingest_kline([push("2026-11-11 09:32:00", close=61.75,
                                 volume=2500.0)], "US.SOXL")
        live._drain_pushed_klines()

        live._on_data_ready(self._result(frame()))

        assert live._last_kl_push is not None
        assert live._klines["close"].iloc[-1] == pytest.approx(61.75)


class TestStoppingLive:
    def test_staged_pushes_are_dropped(self, live):
        """Nothing drains them once the redraw timer is stopped, and on the
        next Live start they would belong to the previous chart."""
        live._ingest_kline([push("2026-11-11 09:32:00")], "US.SOXL")
        live._stop_live()
        assert live._pending_kl == {}
        assert live._last_kl_push is None
