@echo off
cd /d "E:\project\moomoo"
"C:\Users\wanyi\.local\bin\uv.exe" run main.py scheduler
if errorlevel 1 (
    echo.
    echo Scheduler exited with an error - see above.
    pause
)
