@echo off
rem Double-click to refresh the data and publish the site (commits + pushes docs/index.html).
cd /d "%~dp0"
.venv\Scripts\python src\update.py --push
pause
