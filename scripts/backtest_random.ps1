# Random-search backtest + HTML report
#
# Usage (from project root):
#   .\scripts\backtest_random.ps1
#   .\scripts\backtest_random.ps1 -Codes US.NVDA,US.AAPL -N 500
#   .\scripts\backtest_random.ps1 -Codes US.MU -Start 2025-01-01 -End 2025-12-31 -NoReuse
#
# The HTML report is written automatically to:
#   backtest/results/<timestamp>/report_<CODE>.html

param(
    [string[]] $Codes    = @("US.MU"),
    [string]   $Start    = "",
    [string]   $End      = "",
    [int]      $N        = 300,
    [int]      $Seed     = 42,
    [int]      $Workers  = 0,
    [switch]   $NoReuse,
    [switch]   $NoResume
)

$run_args = @("backtest/run.py", "--codes") + $Codes + @("--random", $N, "--seed", $Seed)

if ($Start)    { $run_args += "--start", $Start }
if ($End)      { $run_args += "--end",   $End   }
if ($Workers -gt 0) { $run_args += "--workers", $Workers }
if ($NoReuse)  { $run_args += "--no-reuse"  }
if ($NoResume) { $run_args += "--no-resume" }

Write-Host "==> Random backtest  codes=$($Codes -join ',')  N=$N  seed=$Seed" -ForegroundColor Cyan
uv run @run_args
