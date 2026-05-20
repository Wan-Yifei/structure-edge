"""Pure time-alignment utilities — no I/O, no external dependencies."""

from datetime import datetime


def candle_start(dt: datetime, candle_minutes: int = 15) -> datetime:
    """Align *dt* to the start of its candle window.

    Works for any candle size where candle_minutes <= 60
    (i.e. 1m, 5m, 15m, 30m, 60m).  For 1-hour candles the minute is
    floored to 0 and the hour is preserved by the datetime object itself.
    """
    m = (dt.minute // candle_minutes) * candle_minutes
    return dt.replace(minute=m, second=0, microsecond=0)
