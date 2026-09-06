@echo off
cd /d "E:\project\moomoo"
"C:\Users\wanyi\.local\bin\uv.exe" run main.py replay_trainer
if errorlevel 1 (
    echo.
    echo Replay Trainer exited with an error - see above.
    pause
)
