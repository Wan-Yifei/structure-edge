@echo off
cd /d "E:\project\moomoo"
"C:\Users\wanyi\.local\bin\uv.exe" run main.py trade_viewer_qt
if errorlevel 1 (
    echo.
    echo Trade Viewer exited with an error - see above.
    pause
)
