Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c cd /d E:\project\moomoo && C:\Users\wanyi\.local\bin\uv.exe run main.py trade_viewer_qt > scripts\viewer.log 2>&1", 0, False
