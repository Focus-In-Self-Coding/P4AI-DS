@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" check_datasets.py
set CHECK_EXIT=%ERRORLEVEL%
pause
exit /b %CHECK_EXIT%
