# Exhaustive grid-search backtest + HTML report
#
# Usage (from project root):
#   .\scripts\backtest_grid.ps1
#   .\scripts\backtest_grid.ps1 -Codes US.NVDA,US.AAPL
#   .\scripts\backtest_grid.ps1 -Codes US.MU -Start 2025-01-01 -End 2025-12-31 -NoReuse
#
# The HTML report is written automatically to:
#   backtest/results/<timestamp>/report_<CODE>.html

param(
    [string[]] $Codes    = @("US.MU"),
    [string]   $Start    = "",
    [string]   $End      = "",
    [int]      $Workers  = 0,
    [switch]   $NoReuse,
    [switch]   $NoResume
)

$run_args = @("backtest/run.py", "--codes") + $Codes

if ($Start)    { $run_args += "--start",   $Start   }
if ($End)      { $run_args += "--end",     $End     }
if ($Workers -gt 0) { $run_args += "--workers", $Workers }
if ($NoReuse)  { $run_args += "--no-reuse"  }
if ($NoResume) { $run_args += "--no-resume" }

Write-Host "==> Grid backtest  codes=$($Codes -join ',')" -ForegroundColor Cyan
uv run @run_args
