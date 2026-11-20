"""Hour-of-day panel construction and the statistics it needs.

Reusable layer for the intraday-bias study. The loop and the reporting
live in ``backtest/intraday_bias.py``; the method is fixed by
``strategy/intraday_bias/PLAN.md`` and must not be changed by whatever
the numbers turn out to be.

Analysis only: nothing here writes to a database, issues a trade ID, or
touches the ``smc_v`` / ``svp_v`` tags.
"""
