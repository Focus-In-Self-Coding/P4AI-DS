# MLP Deep Learning Branch — House-Price Regression

This is the complete **project-code and documentation package** for the MLP part of the group assignment. It implements a feed-forward Multilayer Perceptron (MLP) on both datasets:

1. [Housing Prices Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
2. [King County House Sales Prediction (Kaggle)](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

**Important:** the ZIP intentionally does not include the Kaggle CSVs, and it does not contain real result files because those must be produced by running the experiment on the exact datasets selected by your group. Therefore, the code package is complete, but the empirical project is not submission-ready until you download the CSVs, run the official experiment, inspect the outputs, and fill in the report's result placeholders.

## What is included

- `dl_mlp_experiment.py` — main training, CV tuning, evaluation and artifact-generation program.
- `check_datasets.py` — checks whether both CSV files exist, can be read, and contain a usable `price` target.
- `setup_windows.bat` — creates a virtual environment and installs dependencies.
- `check_datasets.bat` — runs the data checks on Windows.
- `run_mlp.bat` — runs the official experiment with the default folder/CSV names.
- `smoke_test.py` — short end-to-end test using synthetic data only; its metrics must never be reported as project results.
- `requirements.txt` — Python dependencies.
- `MLP_House_Prices_Experiments.ipynb` — optional Jupyter launcher for the same main experiment.
- `MLP_Report_Draft.docx` and `MLP_Report_Draft.md` — editable report drafts; results placeholders must be filled after the real run.
- `related_work_methodology_draft.md` — supplementary report text.
- `PRE_SUBMISSION_CHECKLIST.md` — final handoff checklist.
- `data/README.md` — tells you where to obtain and place the CSVs.

## Recommended environment

- Windows 10/11
- Python 3.10, 3.11, or 3.12
- VS Code (recommended) or a terminal
- Internet access to install dependencies and download the two Kaggle datasets
- Enough memory to one-hot encode the King County data

## Step 1 — Put the dataset files in the project

The folder should look like this after you download and extract the CSVs:

```text
dl_mlp_project/
├── data/
│   ├── Housing.csv
│   └── kc_house_data.csv
├── dl_mlp_experiment.py
├── check_datasets.py
├── setup_windows.bat
├── check_datasets.bat
├── run_mlp.bat
├── smoke_test.py
├── requirements.txt
├── README.md
├── MLP_House_Prices_Experiments.ipynb
├── MLP_Report_Draft.docx
├── MLP_Report_Draft.md
├── related_work_methodology_draft.md
└── PRE_SUBMISSION_CHECKLIST.md
```

Download the CSVs from the two Kaggle links above and extract them if they are inside ZIP files. Put them in `data/`. The default filenames are `Housing.csv` and `kc_house_data.csv`. If your files have different names, either rename the files (without changing their contents) or update the paths in the run command. See `data/README.md`.

The dataset CSVs are not redistributed in this code bundle; downloading them is your responsibility.

## Step 2 — Install dependencies (Windows)

In File Explorer, open the extracted project folder and double-click `setup_windows.bat`. Alternatively, use the VS Code terminal opened in the project root and run:

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py` is not recognized, install Python 3.10–3.12 and enable the Python PATH option in its installer. The package installation may take a while because PyTorch is included.

## Step 3 — Check the datasets before a long run

Double-click `check_datasets.bat`, or run:

```bat
.venv\Scripts\python.exe check_datasets.py
```

Confirm that both files load and that each has a `price` column with positive values. The script reports the row count, columns, target issues, missing cells, and a short preview. If the CSV names are different, pass explicit arguments:

```bat
.venv\Scripts\python.exe check_datasets.py --housing_csv "data\your_housing_file.csv" --king_county_csv "data\your_king_county_file.csv"
```

## Step 4 — Optional setup test (synthetic data only)

Before using the real data, you can test that PyTorch, preprocessing, training and output saving work on your computer:

```bat
.venv\Scripts\python.exe smoke_test.py
```

This deliberately runs a tiny, shortened version of the pipeline using generated data. It does not require your Kaggle CSVs. **All scores produced by the smoke test are artificial and must never be used in the report, compared with teammates, or submitted as results.** The script deletes its temporary files when finished.

## Step 5 — Run the official experiment on the Kaggle data

**Recommended:** double-click `run_mlp.bat` after setup and the data checks pass.

Or run this command from the project root:

```bat
.venv\Scripts\python.exe dl_mlp_experiment.py --housing_csv "data\Housing.csv" --king_county_csv "data\kc_house_data.csv" --output_dir "results_mlp"
```

To force CPU mode:

```bat
.venv\Scripts\python.exe dl_mlp_experiment.py --housing_csv "data\Housing.csv" --king_county_csv "data\kc_house_data.csv" --output_dir "results_mlp" --cpu
```

If the environment is already activated, `python dl_mlp_experiment.py ...` is also fine. The complete official experiment is compute-intensive: two datasets, five configurations per dataset, five CV folds per configuration, early stopping, final fitting and feature importance. Keep the terminal open until it finishes. The program can take a substantial amount of time on a CPU.

The script is set up with the default dataset filenames. If yours differ, update the command's two paths. Do not run the terminal and notebook versions at the same time.

## Experimental protocol

- Outer train/test split: 80% / 20%; `random_state=42`.
- Model and shuffled-CV seed: `36`.
- Hyperparameter tuning: five-fold cross-validation on the outer training partition only.
- Early stopping uses an inner validation subset from the fold's fitting data; the CV scoring fold is excluded from model fitting and epoch selection.
- Final model early stopping uses a validation subset taken only from the training partition. The held-out test set is not used to choose the configuration or epoch.
- Numeric preprocessing: median imputation and standardization.
- Categorical preprocessing: most-frequent imputation and one-hot encoding.
- `id` is dropped, King County `date` is transformed into `sale_year`, `sale_month`, and `sale_dayofweek`, and `zipcode` is treated as categorical.
- The target is transformed with `log1p(price)` and standardized using the current model-fit subset. Predictions are returned to the original price scale before calculating MAE, RMSE and R².
- The selected configuration is the one with the lowest mean five-fold CV RMSE in original price units.
- Final reporting metrics: MAE, RMSE, R², final model-fitting time and total CV-search wall time.
- Permutation feature importance is calculated on a validation subset of the outer training data, never on the held-out test set.

## MLP configurations

| Configuration | Hidden layers | Dropout | Learning rate | Weight decay | Batch size |
|---|---:|---:|---:|---:|---:|
| 1 | 64, 32 | 0.00 | 0.001 | 0.0001 | 32 |
| 2 | 128, 64 | 0.10 | 0.001 | 0.0001 | 64 |
| 3 | 128, 64, 32 | 0.20 | 0.0005 | 0.001 | 128 |
| 4 | 256, 128 | 0.10 | 0.0005 | 0.0001 | 128 |
| 5 | 64, 32 | 0.20 | 0.001 | 0.01 | 64 |

All configurations use ReLU hidden activations, a one-value regression output, AdamW, mean squared error, a ReduceLROnPlateau scheduler, gradient clipping and early stopping (up to 200 epochs, patience 22).

## Output files after the official run

The program creates `results_mlp/` and writes:

### Root files

- `combined_test_summary.csv` — one row per dataset with selected config, CV RMSE, test MAE/RMSE/R² and timing.
- `experiment_protocol.json` — split, seeds, hyperparameter configurations and software versions.

### Inside each dataset subfolder

- `cv_results.csv` — mean and standard deviation of metrics for each configuration across CV folds.
- `cv_fold_details.csv` — each configuration/fold score, epoch and fit time.
- `best_hyperparameters.json` — selected configuration and final best epoch.
- `best_config_and_metrics.json` — selected configuration, validation/test metrics and timing.
- `split_manifest.csv` — original row indices assigned to train or test.
- `test_predictions.csv` — test actual price, predicted price and residual.
- `final_training_history.csv` — training/validation loss by epoch.
- `validation_permutation_importance.csv` — validation-RMSE increase after feature shuffling.
- `best_mlp_model.pt` — saved PyTorch model weights and target-scaling information.
- `preprocessor.joblib` — fitted preprocessing pipeline.
- `figures/` — CV comparison, actual-vs-predicted, residual and permutation-importance plots.

Open `combined_test_summary.csv` first. Use `cv_results.csv` to explain model selection, the JSON files to record hyperparameters, and the figures and predictions to analyze the errors. Keep all raw result files intact.

## Notebook option

`MLP_House_Prices_Experiments.ipynb` is an optional launcher for the same Python script. Install the VS Code Jupyter extension, open the notebook, select the `.venv` kernel and run the cell. It expects the default data paths. Do not run the notebook and `run_mlp.bat` simultaneously.

## Report status

`MLP_Report_Draft.docx` and `MLP_Report_Draft.md` contain the introduction, Related Work, methodology, experimental design, and a Results/Discussion framework. The report is a **draft**, not a completed empirical report. Replace every placeholder with values from your official `results_mlp/` output, include the generated figures, and remove all bracketed placeholders before submitting. Do not copy values from the synthetic smoke test.

## Common problems

- **File not found:** check the folder is named `data` and that CSV paths/filenames match the run command.
- **`price` column missing:** inspect the header row. The original datasets should contain a `price` column; do not silently select a different target.
- **`ModuleNotFoundError`:** run setup again, or ensure the `.venv` Python interpreter is selected/used.
- **PowerShell activation blocked:** use the `.venv\Scripts\python.exe` commands shown here, which do not require activation.
- **Out of memory:** close memory-heavy applications and try `--cpu`; avoid launching multiple runs at once. If memory is still insufficient, discuss an approved change to the feature representation with the group and document it.
- **Long runtime:** expected for the full CV search. Do not reduce configurations or epoch limits for official results without recording the change and agreeing it with the group.

## Before submission

Use `PRE_SUBMISSION_CHECKLIST.md`. Verify that all report numbers can be traced to the real result files, confirm the same data split is used by teammates, and compare your MLP with other models only after confirming that metrics use the same test rows and original price units.
