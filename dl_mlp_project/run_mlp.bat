@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Run the MLP experiments on both Kaggle datasets
echo ============================================================
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
if not exist "data\Housing.csv" (
  echo ERROR: missing data\Housing.csv
  echo Download the Housing Prices CSV from Kaggle and see data\README.md.
  pause
  exit /b 1
)
if not exist "data\kc_house_data.csv" (
  echo ERROR: missing data\kc_house_data.csv
  echo Download the King County CSV from Kaggle and see data\README.md.
  pause
  exit /b 1
)
echo Starting the full five-configuration, 5-fold CV experiment.
echo It may take a long time. Keep this window open until it finishes.
echo Output folder: results_mlp
".venv\Scripts\python.exe" dl_mlp_experiment.py --housing_csv "data\Housing.csv" --king_county_csv "data\kc_house_data.csv" --output_dir "results_mlp"
set RUN_EXIT=%ERRORLEVEL%
echo.
if %RUN_EXIT% EQU 0 (
  echo Experiment completed. Open results_mlp\combined_test_summary.csv first.
) else (
  echo The experiment failed. Check the error above and README.md.
)
pause
exit /b %RUN_EXIT%
