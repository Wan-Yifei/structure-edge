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
