"""Editing the targets in the scheduler must reach a running collector.

It did not. Measured on 2026-10-26: config saved 21:58, collectors
running since 20:19, and OpenD's subscription table listed only
`US.SOXL` -- QQQ had been in the config, and visible in the UI, for
sixteen hours without one tick. Both collectors read the file once in
main() and then kept the list in a closure.

These pin the reload itself, and the refusals that matter more than the
happy path: a half-written file and an empty list must never be obeyed,
because unsubscribing everything is worse than being briefly stale.
"""
import json
import pathlib
import sys
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.collector_targets import TargetReloader       # noqa: E402

TICKER = "TICKER"          # a stand-in for SubType.TICKER; never inspected


class _Log:
    def __init__(self):
        self.lines = []

    def warning(self, fmt, *a):
        self.lines.append(fmt % a if a else fmt)

    info = warning


class _Ctx:
    """Records what was asked of it; `ret` drives the next subscribe."""

    def __init__(self, ret=0, raise_on=None):
        self.subs, self.unsubs = [], []
        self.ret = ret
        self.raise_on = raise_on

    def subscribe(self, codes, types, **kw):
        if self.raise_on == "subscribe":
            raise RuntimeError("boom")
        self.subs.append((list(codes), list(types), kw))
        return self.ret, "ok" if self.ret == 0 else "refused"

    def unsubscribe(self, codes, types):
        if self.raise_on == "unsubscribe":
            raise RuntimeError("boom")
        self.unsubs.append((list(codes), list(types)))
        return 0, "ok"


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "schedule.json"

    def write(targets, raw=None):
        p.write_text(raw if raw is not None
                     else json.dumps({"targets": targets}), encoding="utf-8")
        # mtime has a coarse resolution on some filesystems; the reloader
        # compares it, so a same-second rewrite must still register.
        st = p.stat()
        import os
        os.utime(p, (st.st_atime, st.st_mtime + 10))
        return p

    write(["US.SOXL"])
    return p, write


class TestTheReload:
    def test_a_fresh_reloader_does_nothing(self, cfg):
        """main() has just subscribed; the first poll must not redo it."""
        p, _ = cfg
        codes, ctx = ["US.SOXL"], _Ctx()
        r = TargetReloader(p, codes, TICKER, _Log())
        assert r.poll(ctx) is None
        assert ctx.subs == [] and ctx.unsubs == []

    def test_an_added_code_is_subscribed(self, cfg):
        p, write = cfg
        codes, ctx = ["US.SOXL"], _Ctx()
        r = TargetReloader(p, codes, TICKER, _Log())
        write(["US.SOXL", "US.QQQ"])
        line = r.poll(ctx)
        assert ctx.subs and ctx.subs[0][0] == ["US.QQQ"], \
            "only the new code should be subscribed"
        assert codes == ["US.SOXL", "US.QQQ"]
        assert "US.QQQ" in line

    def test_the_list_is_mutated_in_place(self, cfg):
        """_try_subscribe, the rebuild and the retention prune all read
        this same object -- replacing it would leave them on the old."""
        p, write = cfg
        codes = ["US.SOXL"]
        held = codes
        r = TargetReloader(p, codes, TICKER, _Log())
        write(["US.SOXL", "US.QQQ"])
        r.poll(_Ctx())
        assert held is codes and "US.QQQ" in held

    def test_a_removed_code_is_unsubscribed(self, cfg):
        p, write = cfg
        codes, ctx = ["US.SOXL", "HK.02800"], _Ctx()
        r = TargetReloader(p, codes, TICKER, _Log())
        write(["US.SOXL"])
        r.poll(ctx)
        assert ctx.unsubs and ctx.unsubs[0][0] == ["HK.02800"]
        assert codes == ["US.SOXL"]

    def test_subscribe_kwargs_are_passed_through(self, cfg):
        """TICKER needs extended_time and session=ALL; without them the
        new code would only stream during regular hours."""
        p, write = cfg
        codes, ctx = ["US.SOXL"], _Ctx()
        r = TargetReloader(p, codes, TICKER, _Log(),
                           {"subscribe_push": True, "extended_time": True,
                            "session": "ALL"})
        write(["US.SOXL", "US.QQQ"])
        r.poll(ctx)
        assert ctx.subs[0][2] == {"subscribe_push": True,
                                  "extended_time": True, "session": "ALL"}

    def test_an_unchanged_list_costs_one_stat(self, cfg):
        """A rewrite that does not change the targets is not a reload."""
        p, write = cfg
        codes, ctx = ["US.SOXL"], _Ctx()
        r = TargetReloader(p, codes, TICKER, _Log())
        write(["US.SOXL"])
        assert r.poll(ctx) is None
        assert ctx.subs == [] and ctx.unsubs == []


class TestWhatItRefuses:
    def test_an_empty_list_is_refused(self, cfg):
        """Obeying it would unsubscribe everything. A collector that is
        briefly out of date beats one that has stopped collecting."""
        p, write = cfg
        codes, ctx = ["US.SOXL"], _Ctx()
        log = _Log()
        r = TargetReloader(p, codes, TICKER, log)
        write([])
        assert r.poll(ctx) is None
        assert codes == ["US.SOXL"]
        assert ctx.unsubs == []
        assert any("no targets" in m for m in log.lines)

    def test_the_empty_warning_is_said_once(self, cfg):
        p, write = cfg
        log = _Log()
        r = TargetReloader(p, ["US.SOXL"], TICKER, log)
        for _ in range(3):
            write([])
            r.poll(_Ctx())
        assert sum("no targets" in m for m in log.lines) == 1

    def test_a_half_written_file_is_refused(self, cfg):
        """The scheduler rewrites the file whole, so a poll can land in
        the middle of it."""
        p, write = cfg
        codes, ctx = ["US.SOXL"], _Ctx()
        r = TargetReloader(p, codes, TICKER, _Log())
        write(None, raw='{"targets": ["US.SOXL", "US.Q')
        assert r.poll(ctx) is None
        assert codes == ["US.SOXL"] and ctx.unsubs == []

    def test_a_vanished_file_is_refused(self, cfg):
        p, write = cfg
        codes = ["US.SOXL"]
        r = TargetReloader(p, codes, TICKER, _Log())
        p.unlink()
        assert r.poll(_Ctx()) is None
        assert codes == ["US.SOXL"]

    def test_targets_of_the_wrong_shape_are_refused(self, cfg):
        p, write = cfg
        codes = ["US.SOXL"]
        r = TargetReloader(p, codes, TICKER, _Log())
        write(None, raw='{"targets": "US.QQQ"}')
        assert r.poll(_Ctx()) is None
        assert codes == ["US.SOXL"]


class TestWhenTheFeedRefuses:
    def test_a_refused_subscribe_is_retried(self, cfg):
        """OpenD may refuse for a reason that does not last, so the
        reloader must not record the file as applied."""
        p, write = cfg
        codes = ["US.SOXL"]
        r = TargetReloader(p, codes, TICKER, _Log())
        write(["US.SOXL", "US.QQQ"])

        bad = _Ctx(ret=-1)
        r.poll(bad)
        assert codes == ["US.SOXL"], "a refused code must not be claimed"

        good = _Ctx()
        r.poll(good)                       # same file, no further edit
        assert good.subs and good.subs[0][0] == ["US.QQQ"]
        assert codes == ["US.SOXL", "US.QQQ"]

    def test_a_throwing_subscribe_is_survived(self, cfg):
        p, write = cfg
        codes = ["US.SOXL"]
        log = _Log()
        r = TargetReloader(p, codes, TICKER, log)
        write(["US.SOXL", "US.QQQ"])
        r.poll(_Ctx(raise_on="subscribe"))
        assert codes == ["US.SOXL"]
        assert any("error" in m for m in log.lines)

    def test_a_failed_unsubscribe_still_drops_the_code(self, cfg):
        """We no longer want it; a refused unsubscribe only wastes quota,
        and keeping it in the list would have _try_subscribe add it back."""
        p, write = cfg
        codes = ["US.SOXL", "HK.02800"]
        r = TargetReloader(p, codes, TICKER, _Log())
        write(["US.SOXL"])
        r.poll(_Ctx(raise_on="unsubscribe"))
        assert codes == ["US.SOXL"]

    def test_a_clean_apply_is_not_repeated(self, cfg):
        p, write = cfg
        codes = ["US.SOXL"]
        r = TargetReloader(p, codes, TICKER, _Log())
        write(["US.SOXL", "US.QQQ"])
        r.poll(_Ctx())
        again = _Ctx()
        assert r.poll(again) is None
        assert again.subs == []


class TestTheWatchdogActuallyPolls:
    """The unit tests above would all pass with the reloader wired to
    nothing. These run the real watchdog loop once."""

    @pytest.mark.parametrize("mod_name,sub_attr", [
        ("analysis.tick_collector", "TICKER"),
        ("analysis.order_book_collector", "ORDER_BOOK"),
    ])
    def test_the_loop_reloads(self, cfg, mod_name, sub_attr):
        pytest.importorskip("moomoo")
        import importlib
        mod = importlib.import_module(mod_name)

        p, write = cfg
        codes = ["US.SOXL"]
        stop = threading.Event()

        class Ctx(_Ctx):
            def subscribe(self, c, t, **kw):
                out = super().subscribe(c, t, **kw)
                stop.set()          # one pass is all this needs
                return out

        ctx = Ctx()

        # timeout 0 makes check_interval 0, so the loop turns at once and
        # the stub's subscribe stops it. Run it in a thread so a wiring
        # bug fails the test instead of hanging the suite.
        #
        # The edit lands AFTER the thread starts: the reloader is built
        # inside the watchdog and treats the file as it finds it as
        # already applied, which is the point -- main() has just
        # subscribed from that same file.
        state = {"last_tick_time": None, "last_update_time": None,
                 "session_count": 0}
        t = threading.Thread(
            target=mod._watchdog,
            args=(state, 0, stop, [ctx], codes,
                  "127.0.0.1", 11111, None, None, str(p)),
            daemon=True)
        t.start()
        time.sleep(0.05)
        write(["US.SOXL", "US.QQQ"])
        t.join(timeout=5)
        assert not t.is_alive(), "the watchdog did not finish"
        assert ctx.subs and ctx.subs[0][0] == ["US.QQQ"], \
            f"{mod_name} did not apply the new target"
        assert codes == ["US.SOXL", "US.QQQ"]
        assert sub_attr in str(ctx.subs[0][1][0]), \
            "subscribed with the wrong subtype"
