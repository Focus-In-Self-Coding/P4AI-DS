@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Set up the MLP House Price project (Windows)
echo ============================================================
where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python launcher 'py' was not found.
  echo Install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/
  echo During installation, enable the option to add Python to PATH.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 goto :failed
)
echo Installing or updating required packages. This may take a while...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
echo.
echo Setup completed successfully.
echo Next: place the two Kaggle CSV files in the data folder.
echo Run check_datasets.bat before running run_mlp.bat.
pause
exit /b 0
:failed
echo.
echo Setup failed. Read the error above, then check README.md.
pause
exit /b 1
