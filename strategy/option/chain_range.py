"""Group expiry dates into the fewest get_option_chain range calls.

Pure stdlib ON PURPOSE, and in its own module for the same reason. Both
callers need it -- option_tick_collector.pick_contracts and gex's wall fetch
-- but neither can take the other's imports: gex pulls in moomoo AND
matplotlib at module scope, while the collector defers every moomoo import
into its functions so the scheduler can import it at module scope (see the
note beside that import in analysis/scheduler.py). Putting twenty lines of
date arithmetic in either one would have dragged a plotting library into a
supervisor process.

WHY RANGES AT ALL
-----------------
`get_option_chain` is limited to 10 calls per 30 seconds, ACCOUNT-WIDE, and
the limit is shared by every underlying being collected and by the viewer.
One call per expiry is what that budget cannot afford:

  * QQQ lists DAILIES, so a 7-day DTE is five expiries. The viewer polls the
    option walls every 15 s -- ten calls per thirty seconds, exactly the
    limit, before anything else in the account touches a chain. A second
    collector, a 16:50 reconcile or a second window then pushes it over.
  * The refusal is not loud. gex.fetch_option_data records what came back
    and the chart shows "PARTIAL: 2/5 expiries", which is the only thing on
    screen that betrays it -- the wall magnitudes barely move (measured
    660/600 before and after), so three expiries can vanish from a GEX sum
    with the status bar still reading "updated". Intermittently, which is
    what sitting exactly on a limit looks like.
"""

from __future__ import annotations

import logging
from datetime import date

log = logging.getLogger(__name__)

# The documented limit on one start..end range is 30 days; 28 leaves room
# for the API counting the endpoints differently than we do.
#
# Chunked by SPAN, not by count: eight SOXL weeklies reach 49 days apart
# while eight QQQ dailies span about eleven, so fixed-size slices would ask
# for a range the API refuses for one and waste a call on the other.
CHAIN_SPAN_DAYS = 28


def chunk_by_span(expiries: list[str], span_days: int) -> list[list[str]]:
    """Group expiry dates into runs no wider than `span_days`.

    Sorted here rather than trusted from the caller. Both callers happen to
    pass sorted input today, but out of order the spans come out as
    start > end -- measured, a ladder arriving reversed produced
    `start='2027-01-15', end='2026-12-04'`, which costs the whole ladder.
    A function with a precondition that cheap should keep it itself.
    """
    out: list[list[str]] = []
    for e in sorted(expiries):
        # Skipped, not raised. GetOptionExpirationDate substitutes a single
        # space for a missing strikeTime (quote_query.py:3576 -- `optional`
        # in the proto, unlike the distance field beside it) and
        # get_near_expiry_dates appends it unchecked on its `dist` branch.
        # The option CHAIN response is a different field and gives "" there
        # (quote_query.py:2138), so do not read this as covering both.
        # The per-expiry loop this replaced asked OpenD for `start=' '`,
        # got nothing back and carried on one expiry short; raising here
        # would turn that into no session at all, since contracts are
        # picked once per run.
        try:
            when = date.fromisoformat(e)
        except ValueError:
            log.warning("skipping malformed expiry %r from the ladder", e)
            continue
        if out and (when - date.fromisoformat(out[-1][0])).days <= span_days:
            out[-1].append(e)
        else:
            out.append([e])
    return out
