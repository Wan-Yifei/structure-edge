"""Intraday momentum: the first half hour's sign, traded in the last.

The rule is Gao, Han, Li and Zhou's, published in 2018 on a sample that
ends in 2013. Nothing in it is ours: not the signal window, not the
holding window, not the threshold. That is the whole point -- see
PLAN.md section 0.4 -- and it is why adding a single knob would cost far
more than the extra critical value suggests.

Panel machinery is shared with the hour-of-day study and imported from
`strategy.intraday_bias.panel`, which is generic (bar loading, tradeable
days, session labelling, the audit) and parameterised by block size.
"""
