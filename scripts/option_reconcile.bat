@echo off
REM Reconcile the option tape against the broker's own daily volume.
REM
REM Run at about 16:45 ET, NOT at the close: 5.6%% of prints and 7.8%% of
REM contracts arrive after 16:00 as late reports, trailing to roughly 16:15,
REM and the broker's daily `volume` is not settled until after that. Run too
REM early the ratio reads low for an honest reason and says nothing.
REM
REM This is the only check that can tell you a session was under-collected,
REM and the failure it looks for -- a subscription that quietly stops
REM delivering -- leaves the process looking perfectly healthy.
cd /d "E:\project\moomoo"
echo ================================================================
echo  Option tape reconciliation  -  US.SOXL
echo.
echo  Reading:  coverage N%%  -- stored volume vs the broker's own
echo            "caught NOTHING" -- the shape of a dropped subscription
echo            "NOT COMPARABLE" -- wrong day; pass --session YYYY-MM-DD
echo ================================================================
echo.
"C:\Users\wanyi\.local\bin\uv.exe" run analysis/option_tick_collector.py --code US.SOXL --reconcile-only
echo.
pause
