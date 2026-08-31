@echo off
cd /d "E:\project\moomoo"
"C:\Users\wanyi\.local\bin\uv.exe" run main.py scanner
if errorlevel 1 (
    echo.
    echo Signal Scanner exited with an error - see above.
    pause
)
