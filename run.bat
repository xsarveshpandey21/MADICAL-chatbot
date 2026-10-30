@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    cd RAGChatBot
    "..\.venv\Scripts\python.exe" app.py
) else (
    cd RAGChatBot
    python app.py
)
pause
