@echo off
REM Launch the viewer with the axis repaint probe on. ONE SESSION, then go
REM back to launch_viewer.bat -- the probe is for answering a question, not
REM for every day.
REM
REM A separate file rather than a switch in launch_viewer.bat: an
REM instrument that is on by default is one nobody remembers is running,
REM and every later reading carries it.
REM
REM It reports every 300 s and once more when the window is closed, so
REM close the window at the end of the session rather than killing the
REM process -- a kill takes the closing reading with it. The 300 s
REM readings survive either way, which is why the interval is not 0.
REM
REM Output goes to scripts\viewer_probe.log, NOT viewer.log, so the normal
REM log is not overwritten and the session's readings stay in one file.
REM See analysis\qt_axis_probe.py for what the numbers mean -- and note it
REM prints no verdict, deliberately.

cd /d "E:\project\moomoo"
set "MOOMOO_AXIS_PROBE=1"
set "MOOMOO_AXIS_PROBE_SECS=300"
echo Axis probe ON. Readings go to scripts\viewer_probe.log every 300s.
echo Close the viewer window normally at the end to get the final reading.
"C:\Users\wanyi\.local\bin\uv.exe" run main.py trade_viewer_qt > "scripts\viewer_probe.log" 2>&1
if errorlevel 1 (
    echo.
    echo Trade Viewer exited with an error - see scripts\viewer_probe.log
    pause
)
