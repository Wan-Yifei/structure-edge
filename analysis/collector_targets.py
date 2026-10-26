"""Pick up a changed target list without restarting a collector.

Both collectors read the config once in main() and then keep the code
list in a closure, so editing the targets in the scheduler changed the
file and nothing else. Measured on 2026-10-26: the config was saved at
21:58 while the collectors had been running since 20:19, and OpenD's
own subscription table listed only the one code they had started with --
QQQ had been in the config, and in the UI, for sixteen hours without a
single tick.

The watchdog already wakes every 60s, and already owns `codes` as a
mutable list that `_try_subscribe`, `_rebuild_ctx` and the retention
prune all read through. So the reload is a diff applied to that same
list: nothing else has to learn about it.

Threading: `codes` is mutated from the watchdog thread and read from the
watchdog thread. The main thread subscribes once at startup and does not
touch it afterwards, so no lock is needed -- but that is the invariant
this relies on.
"""
from __future__ import annotations

import json
import pathlib


class TargetReloader:
    """Applies the config's target list to a live subscription.

    Constructed after the collector has subscribed, so the file's current
    state counts as already applied: the first poll must not re-subscribe
    what main() just did.
    """

    def __init__(self, config_path, codes: list, sub_type, log,
                 sub_kwargs: dict | None = None) -> None:
        self._path = pathlib.Path(config_path)
        self._codes = codes                   # mutated in place, by design
        self._sub_type = sub_type
        self._log = log
        self._kwargs = sub_kwargs or {}
        self._mtime = self._stat()
        self._warned_empty = False

    def _stat(self) -> float | None:
        try:
            return self._path.stat().st_mtime
        except OSError:
            return None

    def _read(self) -> list | None:
        """The config's targets, or None when the file cannot be trusted.

        A missing file, a partial write (the scheduler rewrites it whole)
        or an empty list all return None. Empty is refused rather than
        obeyed: a collector that unsubscribes everything because a save
        was caught halfway is worse than one that is briefly out of date.
        """
        try:
            with open(self._path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            return None
        want = cfg.get("targets")
        if not isinstance(want, list):
            return None
        want = [str(c).strip() for c in want if str(c).strip()]
        if not want:
            if not self._warned_empty:
                self._log.warning(
                    "Config lists no targets — keeping %s", self._codes)
                self._warned_empty = True
            return None
        self._warned_empty = False
        return want

    def poll(self, ctx) -> str | None:
        """Bring the live subscription in line with the config.

        Returns a line worth logging, or None when nothing changed.

        The stored mtime only advances on a clean apply. A subscribe that
        fails therefore leaves the reloader looking at an out-of-date
        file, and the next poll retries -- which is what should happen
        when OpenD refuses a code for a reason that may not last.
        """
        m = self._stat()
        if m == self._mtime:
            return None
        want = self._read()
        if want is None:
            return None

        dropped = [c for c in self._codes if c not in want]
        added = [c for c in want if c not in self._codes]
        if not dropped and not added:
            self._mtime = m
            return None

        clean = True
        if dropped:
            try:
                ctx.unsubscribe(dropped, [self._sub_type])
            except Exception as exc:
                # Dropped either way: we no longer want these, and a
                # refused unsubscribe only wastes quota. The handler may
                # keep seeing pushes until OpenD catches up.
                self._log.warning("Unsubscribe %s failed: %s", dropped, exc)
            for c in dropped:
                self._codes.remove(c)

        if added:
            ok = False
            try:
                ret, msg = ctx.subscribe(added, [self._sub_type],
                                         **self._kwargs)
                # moomoo.RET_OK is 0; compared literally so this module
                # stays importable without the SDK, which is what lets
                # the tests drive it with a stub context.
                ok = ret == 0
                if not ok:
                    self._log.warning("Subscribe %s failed: %s", added, msg)
            except Exception as exc:
                self._log.warning("Subscribe %s error: %s", added, exc)
            if ok:
                self._codes.extend(added)
            else:
                clean = False

        if clean:
            self._mtime = m
        return (f"targets reloaded: +{added or '[]'} -{dropped or '[]'} "
                f"-> {', '.join(self._codes)}")
