# Fast smoke-test backtest (minimal params, single symbol)
#
# Usage (from project root):
#   .\scripts\backtest_fast.ps1
#   .\scripts\backtest_fast.ps1 -Codes US.NVDA

param(
    [string[]] $Codes = @("US.MU")
)

Write-Host "==> Fast smoke test  codes=$($Codes -join ',')" -ForegroundColor Cyan
uv run backtest/run.py --fast --codes @Codes
