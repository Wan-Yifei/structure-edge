# Option GEX Analyzer

Measures **Gamma Exposure (GEX)** from near-expiry options to estimate
market-maker hedging pressure on the underlying.

## Quick start

```powershell
# Show interactive chart (OpenD must be running)
uv run strategy/option/gex.py --code US.SOXL --dte 7

# Save to auto-named file:  SOXL_20260705_gex_dte7.png
uv run strategy/option/gex.py --code US.SOXL --dte 7 --out strategy/option/output/

# Wider expiry window
uv run strategy/option/gex.py --code US.NVDA --dte 14 --out strategy/option/output/
```

`--dte N` filters for all expiry dates within N calendar days from today.
`--out <dir>` auto-generates the filename; `--out <file.png>` saves to that exact path.

---

## Background

### What is GEX?

Options market-makers (MMs) are typically **net short options** — they sell
contracts to retail and institutional buyers and delta-hedge their inventory
by trading the underlying.  The rate at which their hedge must change as the
underlying moves is determined by **gamma**:

```
GEX per contract  =  gamma × open_interest × 100 × spot_price   (shares)

Net GEX  =  Σ GEX_calls  −  Σ GEX_puts
```

| Net GEX | MM position | Market behaviour |
|---------|-------------|------------------|
| **Positive** | Net long gamma | MMs sell rallies, buy dips → **suppresses volatility**, mean-reversion |
| **Negative** | Net short gamma | MMs buy rallies, sell dips → **amplifies moves**, trending / volatile |

### What is Zero Gamma (the "gamma flip" price)?

Zero Gamma is the hypothetical **spot price** at which total dealer GEX would
flip sign. Gamma is not a fixed per-option number — it changes as the
underlying moves (it peaks ATM and decays away from the strike), so finding
this price requires **repricing every option's gamma at a range of
hypothetical spot levels**, not just reading gamma at today's price:

```
for each hypothetical spot S in a price grid:
    gamma_i(S)  =  Black-Scholes gamma of option i, repriced at S
                   (using that option's own strike, DTE, implied vol)
    GEX(S)      =  Σ_i  sign_i × gamma_i(S) × open_interest_i × 100 × S

Zero Gamma  =  the S where GEX(S) crosses zero (interpolated)
```

`gex.py` implements this in `_zero_gamma()`, using each option's own
`option_implied_volatility` (moomoo reports this in **percent**, e.g. `241.3`
→ divide by 100 for the BS formula) and `strike_time` to get time-to-expiry.
It was validated against moomoo's own in-app Gamma Exposure chart for
`US.SOXL` (2026-07-19, 2026-07-24 expiry): computed $194.70 vs. moomoo's
$194.15 — within 0.3%.

> **Common pitfall** — it's tempting to instead take each option's gamma
> *at today's spot price* (already provided by the broker), sort by strike,
> and find where the running cumulative sum crosses zero. This is cheap to
> compute (no repricing needed) and is shown in the chart as the blue
> "Cumulative GEX" line, but it answers a different question — "how is GEX
> distributed across strikes right now" — and its zero-crossing is **not**
> the same price as the true Zero Gamma above. On SOXL the two disagreed by
> over $100 (a naive cumulative-sum implementation gave $79 vs. the correct
> $194.70).

### What is ITM?

**In The Money (ITM)** means the option has intrinsic value right now:

| Type | ITM when | Example (spot $72.50) |
|------|----------|-----------------------|
| Call | strike < spot | $70 call: can buy at $70, worth $2.50 immediately |
| Put  | strike > spot | $75 put: can sell at $75 when market is $72.50 |

> Note: **gamma peaks at ATM** (strike ≈ spot), not at deep-ITM strikes.
> The chart shows all strikes; the ITM shading is for orientation only.

---

## Example output

### Terminal

```
Fetching data for US.SOXL  (DTE ≤ 7) ...
  Spot: $72.50   20d avg vol: 44,820,000 sh
  Expiries: ['2026-07-07', '2026-07-11']
  Options loaded: 84

  Net GEX : -2.3M sh  ▼ Destabilizing
  ITM Call OI : 4,982 contracts = 498.2K sh  (1.1% of avg vol)
  ITM Put  OI : 7,614 contracts = 761.4K sh  (1.7% of avg vol)
  Combined ITM: 1.3M sh = 2.8% of avg vol
Chart saved: strategy/option/output/SOXL_20260705_gex_dte7.png
```

### Chart anatomy

```
┌─────────────────────────────────────────────────────────────────┐
│  SOXL — Option GEX  │  Data: 2026-07-05  │  Spot: $72.50       │
│  Expiries: 2026-07-07 · 2026-07-11                              │
│                                                                  │
│  ▲ Net GEX                                                       │
│  |    ██                                                         │
│  |    ██  ██                  (blue bars: calls > puts)          │
│  0 ───────────────── $72.50 ──────────────────────────          │
│          ██  ██  ██  |  ██  ██  ██  ██                           │
│          ██  ██  ██  |  ██  ██  ██  ██  (red bars: puts > calls) │
│  ▼                   │                                           │
│                   spot line                                      │
│  ◀── ITM Call zone ──┼──── ITM Put zone ──────────────▶          │
│         (blue tint)  │          (red tint)                       │
├──────────────────────────────────────────────────────────────────┤
│ Net GEX Summary  │ ITM Call Options │ ITM Put Options │ Combined │
│ Data: 2026-07-05 │ OI: 4,982 cts   │ OI: 7,614 cts   │ Total OI │
│ Expiries: ...    │ Equiv: 498.2K   │ Equiv: 761.4K   │ 1.3M sh  │
│ Net GEX: -2.3M   │ % AvgVol: 1.1%  │ % AvgVol: 1.7%  │ 2.8%     │
│ ▼ Destabilizing  │                 │                 │          │
└──────────────────────────────────────────────────────────────────┘
```

**Title bar** — ticker, data fetch date, live spot price, and all expiry
dates covered by this run.  Use the date to know how stale a saved chart is.

**Bar chart** — each bar = one strike price, height = net GEX contribution
(calls minus puts) at that strike.

- **Blue bar** — calls dominate at this strike; MMs are net long gamma here
- **Red bar** — puts dominate; MMs net short gamma here
- **Yellow dashed line** — spot price at fetch time
- **Blue/red tint** — ITM zones (orientation only; gamma peaks at ATM, not deep ITM)

**Stats panel** — four columns showing ITM OI, equivalent share counts,
and their size relative to the 20-day average daily volume of the underlying.

---

## How to interpret results

### Reading Net GEX direction

| Signal | What it suggests |
|--------|-----------------|
| Large positive Net GEX, spot near ATM cluster | Strong MM pin force; price likely to oscillate around the GEX peak strike |
| Large negative Net GEX | MM amplification risk; intraday moves may be exaggerated |
| GEX near zero | Mixed positioning; no strong structural bias |
| GEX flips sign intraday | MM hedge rebalancing can cause sudden directional surges |

### Reading ITM % of avg vol

The "% of 20d Avg Vol" column measures how large the ITM options inventory
is relative to the underlying's daily liquidity.

| Range | Interpretation |
|-------|----------------|
| < 1%  | Modest options overlay; limited structural impact |
| 1–5%  | Meaningful; MM delta hedges are a notable flow component |
| > 5%  | Large; options-driven flows can dominate price action near expiry |

### Expiry pinning

Near expiry, MMs dynamically hedge their gamma.  If a large OI cluster sits
at a specific strike, MMs must sell the underlying as it rises above that
strike and buy as it falls below — creating a gravitational pull known as
**max pain / gamma pinning**.  The strike with the tallest blue bar is the
most likely pin candidate.

---

## Limitations

1. **Dealer direction assumed** — the formula assumes MMs are net short.
   If a large directional institution sold calls/puts to hedge their own
   book, GEX sign can be misleading.

2. **OI is end-of-day** — intraday OI changes are not captured; early-week
   readings for Friday expiry will be less accurate than Thursday readings.

3. **Single underlying** — SOXL as a 3× leveraged ETF has its own
   rebalancing flows that interact with options GEX in a non-linear way.

4. **Gamma is model-dependent** — reported gamma comes from moomoo's
   Black-Scholes pricing; actual MM gamma may differ if they use different
   models or carry residual vega exposure.

---

## File layout

```
strategy/option/
├── gex.py          — main script
├── README.md       — this file (English, committed)
├── README_zh.md    — Chinese notes (local only, not committed)
├── .gitignore      — excludes output/ and README_zh.md
└── output/         — saved charts (not committed)
    └── SOXL_20260705_gex_dte7.png
```
