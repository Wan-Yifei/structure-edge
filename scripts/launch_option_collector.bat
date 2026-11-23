@echo off
REM Option tape collector -- SOXL, first live session.
REM
REM Deliberately a VISIBLE console, unlike launch_viewer.vbs: day one is a
REM watched run. Ctrl-C in this window is also the only shutdown that reaches
REM the collector's cleanup -- on Windows a parent's terminate() calls
REM TerminateProcess, which runs no handler and no `finally`, so the
REM unsubscribe / reconcile / prune / checkpoint would all be skipped.
cd /d "E:\project\moomoo"
echo ================================================================
echo  Option tape collector  -  US.SOXL
echo.
echo  The option tape CANNOT be re-fetched. If this session is missed
echo  or mis-collected, that day is gone.
echo.
echo  Watch for:  "DATA RECEIVED"        within a minute of 09:30
echo              "telemetry: subscribed N of 200"   every ~30 min
echo              "subscribed N of 200" where N drops  -- eviction
echo.
echo  Stop with Ctrl-C (NOT by closing the window) at about 16:25.
echo  Then run scripts\option_reconcile.bat at about 16:45.
echo ================================================================
echo.
"C:\Users\wanyi\.local\bin\uv.exe" run analysis/option_tick_collector.py --code US.SOXL
echo.
echo Collector exited with errorlevel %errorlevel%.
pause
