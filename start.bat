@echo off
rem Double-click to open the LiveSubtitle control window (no console)
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m voicetype %*
