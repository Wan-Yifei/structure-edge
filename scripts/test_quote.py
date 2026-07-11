"""Quick test: fetch real-time quote for a list of codes via moomoo API."""
import sys
from moomoo import OpenQuoteContext, RET_OK

CODES = ["US.SOXL", "US.SOXS", "US.SPY"]
HOST  = "127.0.0.1"
PORT  = 11111

from moomoo import SubType

ctx = OpenQuoteContext(host=HOST, port=PORT)
ctx.subscribe(CODES, [SubType.QUOTE])
ret, data = ctx.get_stock_quote(CODES)
ctx.close()

if ret != RET_OK:
    print(f"FAILED: {data}")
    sys.exit(1)

cols = ["code", "last_price", "open_price", "high_price", "low_price",
        "volume", "update_time", "market_status"]
print(data[[c for c in cols if c in data.columns]].to_string(index=False))
