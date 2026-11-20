"""Initial balance: the opening range, and what the day does after it.

"Initial balance" is the auction-market term for the high-low range of
the session's opening period -- here the first 30 minutes. It is NOT
implied volatility, and it is NOT the `IVB` the user calls
`strategy/session_vp/`. Spell it out everywhere; two of those three
names differ by one letter and mean different things.

Panel machinery is shared with the hour-of-day study and imported from
`strategy.intraday_bias.panel`, which is generic and parameterised by
block size.
"""
