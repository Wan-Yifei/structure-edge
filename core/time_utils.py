"""Pure time-alignment utilities — no I/O, no external dependencies."""

from datetime import datetime


def candle_start(dt: datetime, candle_minutes: int = 15) -> datetime:
    """Align *dt* to the start of its candle window.

    Supports any candle size:
      <= 60 min  : floor to the nearest candle boundary within the hour
      240 min (4h): floor to the nearest 4-hour boundary (00/04/08/12/16/20)
      1440 min (1d): floor to midnight (start of the calendar day)
    """
    if candle_minutes >= 1440:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if candle_minutes > 60:
        total_minutes = dt.hour * 60 + dt.minute
        floored = (total_minutes // candle_minutes) * candle_minutes
        return dt.replace(hour=floored // 60, minute=floored % 60,
                          second=0, microsecond=0)
    m = (dt.minute // candle_minutes) * candle_minutes
    return dt.replace(minute=m, second=0, microsecond=0)


def _parse_hm(s: str) -> int:
    """'HH:MM' -> minutes since midnight."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def session_for_timestamp(dt: datetime, sessions: dict) -> str | None:
    """Return which session in *sessions* the clock time of *dt* falls in.

    *sessions* maps name -> {"start": "HH:MM", "end": "HH:MM", "enabled": bool},
    matching config/schedule.json's "sessions" block -- passed in rather than
    loaded here so this stays a pure function with no I/O.

    A session whose end is numerically <= its start (e.g. overnight
    20:00-04:00) is treated as crossing midnight: it covers [start, 24:00)
    unioned with [00:00, end). Getting this wrong (checking start <= mins <
    end literally, without the wraparound case) silently excludes every
    post-midnight timestamp from an overnight-style session -- exactly the
    bug found and fixed in trade_viewer_qt.py's _filter_sessions(), which
    this function deliberately does not import (that method is GUI-coupled
    and instance-bound) so as not to re-introduce a second, divergent copy.
    """
    mins = dt.hour * 60 + dt.minute
    for name, cfg in sessions.items():
        if not cfg.get("enabled", True):
            continue
        lo = _parse_hm(cfg["start"])
        hi = _parse_hm(cfg["end"])
        if hi <= lo:
            if mins >= lo or mins < hi:
                return name
        elif lo <= mins < hi:
            return name
    return None


def minutes_since_session_start(dt: datetime, sessions: dict) -> tuple[str, int] | None:
    """Return (session_name, minutes elapsed since that session started).

    None if *dt* doesn't fall in any enabled session (see session_for_timestamp).
    Handles the same midnight-wraparound case: for the post-midnight portion
    of an overnight-style session, elapsed time is measured from the
    *previous* day's start clock time, not from 00:00.
    """
    name = session_for_timestamp(dt, sessions)
    if name is None:
        return None
    mins = dt.hour * 60 + dt.minute
    lo = _parse_hm(sessions[name]["start"])
    hi = _parse_hm(sessions[name]["end"])
    if hi <= lo and mins < hi:
        elapsed = (24 * 60 - lo) + mins
    else:
        elapsed = mins - lo
    return name, elapsed
