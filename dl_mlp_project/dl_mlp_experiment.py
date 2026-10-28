"""Reproducible MLP experiments for two housing-price regression datasets.

Protocol:
- Outer train/test split: 80/20, random_state=42.
- Model randomness and shuffled CV: seed=36.
- Hyperparameter selection: 5-fold CV on the outer training split.
- Every CV fold uses an inner validation split for early stopping; the outer CV fold
  is used only to score the selected configuration for that fold.
- Final training uses a separate validation subset taken from outer training data.
- Test set is evaluated only after the configuration is chosen and final model trained.
- The target is log1p-transformed for fitting; metrics are reported in original price units.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 36
SPLIT_SEED = 42
TARGET = "price"
LOG_TARGET = True
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 22


@dataclass(frozen=True)
class MLPConfig:
    name: str
    hidden_dims: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int


CONFIGS = [
    MLPConfig("mlp_64_32_d0_lr1e-3_wd1e-4_b32", (64, 32), 0.00, 1e-3, 1e-4, 32),
    MLPConfig("mlp_128_64_d0.1_lr1e-3_wd1e-4_b64", (128, 64), 0.10, 1e-3, 1e-4, 64),
    MLPConfig("mlp_128_64_32_d0.2_lr5e-4_wd1e-3_b128", (128, 64, 32), 0.20, 5e-4, 1e-3, 128),
    MLPConfig("mlp_256_128_d0.1_lr5e-4_wd1e-4_b128", (256, 128), 0.10, 5e-4, 1e-4, 128),
    MLPConfig("mlp_64_32_d0.2_lr1e-3_wd1e-2_b64", (64, 32), 0.20, 1e-3, 1e-2, 64),
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_dataframe(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read a CSV and apply only deterministic, row-wise feature cleanup."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)

    if TARGET not in df.columns:
        raise ValueError(
            f"Couldn't find a '{TARGET}' target column in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # King County's date field is a timestamp-like string such as 20141013T000000.
    # Extract the calendar date into numeric features; never use the raw date string.
    if "date" in df.columns:
        raw_date = df["date"].astype(str)
        pieces = raw_date.str.extract(r"(\d{4})\D?(\d{2})\D?(\d{2})")
        normalized = pieces[0] + "-" + pieces[1] + "-" + pieces[2]
        parsed = pd.to_datetime(normalized, errors="coerce")
        if parsed.notna().any():
            df["sale_year"] = parsed.dt.year.astype("float64")
            df["sale_month"] = parsed.dt.month.astype("float64")
            df["sale_dayofweek"] = parsed.dt.dayofweek.astype("float64")
        df = df.drop(columns=["date"])

    # IDs do not generalize as house attributes and may cause spurious memorization.
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # ZIP code is a location category, not a continuous measurement.
    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype("string")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.loc[df[TARGET].notna()].copy()
    if df.empty:
        raise ValueError(f"No usable target values found in {path}.")
    if (df[TARGET] <= 0).any() and LOG_TARGET:
        raise ValueError("log1p target transformation is enabled, so all house prices must be > 0.")

    y = df.pop(TARGET).astype("float64")
    X = df.replace([np.inf, -np.inf], np.nan)
    if X.shape[1] == 0:
        raise ValueError("No input features remain after preprocessing.")
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [
        c for c in X.columns
        if (pd.api.types.is_object_dtype(X[c])
            or pd.api.types.is_string_dtype(X[c])
            or pd.api.types.is_categorical_dtype(X[c])
            or pd.api.types.is_bool_dtype(X[c]))
    ]
    numeric = [c for c in X.columns if c not in categorical]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # Compatibility with scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder),
    ])
    transformers = []
    if numeric:
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        transformers.append(("cat", cat_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def transform_target(y: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    return np.log1p(arr) if LOG_TARGET else arr


def invert_target(y_transformed: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    values = np.asarray(y_transformed, dtype=np.float64) * y_std + y_mean
    return np.expm1(values) if LOG_TARGET else values


class MLPRegressor(nn.Module):
    def __init__(self, n_features: int, config: MLPConfig):
        super().__init__()
        layers: list[nn.Module] = []
        current = n_features
        for width in config.hidden_dims:
            layers.append(nn.Linear(current, width))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def _data_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_early_stopping(
    X_fit: np.ndarray,
    y_fit_scaled: np.ndarray,
    X_val: np.ndarray,
    y_val_scaled: np.ndarray,
    config: MLPConfig,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> tuple[MLPRegressor, int, list[dict[str, float]]]:
    set_seed(SEED)
    model = MLPRegressor(X_fit.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )
    loss_fn = nn.MSELoss()
    loader = _data_loader(X_fit, y_fit_scaled, config.batch_size, shuffle=True)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val_scaled.astype(np.float32)).to(device)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), y_val_t).cpu())
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_mse_scaled": val_loss})
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, history


def predict_log_scaled(model: MLPRegressor, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X).to(device)).detach().cpu().numpy()
    return np.asarray(out, dtype=np.float64)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_cv_config(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: MLPConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_rows = []
    best_epochs = []
    for fold, (idx_fit, idx_score) in enumerate(cv.split(X_train), start=1):
        X_fold_fit_all = X_train.iloc[idx_fit]
        y_fold_fit_all = y_train.iloc[idx_fit]
        X_fold_score = X_train.iloc[idx_score]
        y_fold_score = y_train.iloc[idx_score]

        # Inner validation is used for early stopping; outer fold is scoring-only.
        X_fit, X_early, y_fit, y_early = train_test_split(
            X_fold_fit_all, y_fold_fit_all, test_size=0.15, random_state=SEED
        )
        prep = make_preprocessor(X_fit)
        A_fit = dense_float32(prep.fit_transform(X_fit))
        A_early = dense_float32(prep.transform(X_early))
        A_score = dense_float32(prep.transform(X_fold_score))

        yt_fit = transform_target(y_fit)
        target_mean = float(yt_fit.mean())
        target_std = float(yt_fit.std()) or 1.0
        y_fit_scaled = ((yt_fit - target_mean) / target_std).astype(np.float32)
        y_early_scaled = ((transform_target(y_early) - target_mean) / target_std).astype(np.float32)

        started = time.perf_counter()
        model, best_epoch, _ = train_early_stopping(
            A_fit, y_fit_scaled, A_early, y_early_scaled, config, device
        )
        elapsed = time.perf_counter() - started
        pred_scaled = predict_log_scaled(model, A_score, device)
        pred = invert_target(pred_scaled, target_mean, target_std)
        metrics = regression_metrics(y_fold_score.to_numpy(), pred)
        best_epochs.append(best_epoch)
        fold_rows.append({
            "config": config.name,
            "fold": fold,
            **metrics,
            "best_epoch": best_epoch,
            "fit_time_sec": elapsed,
            "n_inner_train": len(X_fit),
            "n_inner_validation": len(X_early),
            "n_cv_score": len(X_fold_score),
        })

    summary = {
        "config": config.name,
        "hidden_dims": list(config.hidden_dims),
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "cv_MAE_mean": float(np.mean([r["MAE"] for r in fold_rows])),
        "cv_MAE_std": float(np.std([r["MAE"] for r in fold_rows], ddof=1)),
        "cv_RMSE_mean": float(np.mean([r["RMSE"] for r in fold_rows])),
        "cv_RMSE_std": float(np.std([r["RMSE"] for r in fold_rows], ddof=1)),
        "cv_R2_mean": float(np.mean([r["R2"] for r in fold_rows])),
        "cv_R2_std": float(np.std([r["R2"] for r in fold_rows], ddof=1)),
        "median_best_epoch": int(np.median(best_epochs)),
        "total_cv_fit_time_sec": float(sum(r["fit_time_sec"] for r in fold_rows)),
    }
    return summary, fold_rows


def validation_permutation_importance(
    model: MLPRegressor,
    preprocessor: ColumnTransformer,
    X_val_raw: pd.DataFrame,
    y_val: pd.Series,
    target_mean: float,
    target_std: float,
    device: torch.device,
    repeats: int = 5,
) -> pd.DataFrame:
    """Permutation importance on a validation subset of TRAIN only (not test)."""
    baseline_X = dense_float32(preprocessor.transform(X_val_raw))
    base_pred = invert_target(predict_log_scaled(model, baseline_X, device), target_mean, target_std)
    base_rmse = float(np.sqrt(mean_squared_error(y_val.to_numpy(), base_pred)))
    rng = np.random.default_rng(SEED)
    rows = []
    for feature in X_val_raw.columns:
        changes = []
        for _ in range(repeats):
            permuted = X_val_raw.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            perm_X = dense_float32(preprocessor.transform(permuted))
            pred = invert_target(predict_log_scaled(model, perm_X, device), target_mean, target_std)
            perm_rmse = float(np.sqrt(mean_squared_error(y_val.to_numpy(), pred)))
            changes.append(perm_rmse - base_rmse)
        rows.append({
            "feature": feature,
            "importance_RMSE_increase_mean": float(np.mean(changes)),
            "importance_RMSE_increase_std": float(np.std(changes, ddof=1)) if repeats > 1 else 0.0,
            "validation_baseline_RMSE": base_rmse,
        })
    return pd.DataFrame(rows).sort_values("importance_RMSE_increase_mean", ascending=False)


def save_plots(
    dataset_name: str,
    out_dir: Path,
    cv_summary: pd.DataFrame,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    importance: pd.DataFrame,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ranked = cv_summary.sort_values("cv_RMSE_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(ranked["config"], ranked["cv_RMSE_mean"], xerr=ranked["cv_RMSE_std"], capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV RMSE (original price units; lower is better)")
    ax.set_title(f"{dataset_name}: MLP hyperparameter comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / "cv_rmse_by_config.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, pred_test, alpha=0.55, s=18)
    low = float(min(np.min(y_test), np.min(pred_test)))
    high = float(max(np.max(y_test), np.max(pred_test)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title(f"{dataset_name}: final test predictions")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    residuals = pred_test - y_test
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(pred_test, residuals, alpha=0.55, s=18)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Prediction error (predicted - actual)")
    ax.set_title(f"{dataset_name}: test residual plot")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_residuals.png", dpi=160)
    plt.close(fig)

    top = importance.head(15).sort_values("importance_RMSE_increase_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance_RMSE_increase_mean"],
            xerr=top["importance_RMSE_increase_std"], capsize=2)
    ax.set_xlabel("Increase in validation RMSE after permutation (higher = more important)")
    ax.set_title(f"{dataset_name}: permutation feature importance (validation set)")
    fig.tight_layout()
    fig.savefig(fig_dir / "validation_permutation_importance.png", dpi=160)
    plt.close(fig)


def run_experiment(dataset_name: str, csv_path: str | Path, root_out: Path, device: torch.device) -> dict[str, Any]:
    print(f"\n{'=' * 76}\nDataset: {dataset_name}\nCSV: {csv_path}\nDevice: {device}\n{'=' * 76}")
    X, y = prepare_dataframe(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SPLIT_SEED
    )
    print(f"Rows={len(X)}, features_before_encoding={X.shape[1]}, train={len(X_train)}, test={len(X_test)}")

    out_dir = root_out / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the original CSV row index so teammates can recreate the exact split.
    split_manifest = pd.DataFrame({
        "source_row_index": list(X_train.index) + list(X_test.index),
        "split": ["train"] * len(X_train) + ["test"] * len(X_test),
    }).sort_values("source_row_index")
    split_manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    cv_summaries = []
    all_fold_rows = []
    search_started = time.perf_counter()
    for i, config in enumerate(CONFIGS, start=1):
        print(f"[{i}/{len(CONFIGS)}] CV tuning: {config.name}")
        summary, fold_rows = evaluate_cv_config(X_train, y_train, config, device)
        cv_summaries.append(summary)
        all_fold_rows.extend(fold_rows)
        print(f"    CV RMSE={summary['cv_RMSE_mean']:.3f} +/- {summary['cv_RMSE_std']:.3f}; "
              f"CV MAE={summary['cv_MAE_mean']:.3f}; CV R2={summary['cv_R2_mean']:.4f}")
    cv_search_time = time.perf_counter() - search_started
    cv_df = pd.DataFrame(cv_summaries).sort_values("cv_RMSE_mean", ascending=True)
    folds_df = pd.DataFrame(all_fold_rows)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    folds_df.to_csv(out_dir / "cv_fold_details.csv", index=False)

    best_name = str(cv_df.iloc[0]["config"])
    best_config = next(c for c in CONFIGS if c.name == best_name)
    print(f"Selected by lowest mean CV RMSE: {best_name}")

    # Independent validation subset from the outer training split. Test remains untouched.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED
    )
    preprocessor = make_preprocessor(X_fit)
    A_fit = dense_float32(preprocessor.fit_transform(X_fit))
    A_val = dense_float32(preprocessor.transform(X_val))
    A_test = dense_float32(preprocessor.transform(X_test))

    yt_fit = transform_target(y_fit)
    target_mean = float(yt_fit.mean())
    target_std = float(yt_fit.std()) or 1.0
    y_fit_scaled = ((yt_fit - target_mean) / target_std).astype(np.float32)
    y_val_scaled = ((transform_target(y_val) - target_mean) / target_std).astype(np.float32)

    final_started = time.perf_counter()
    model, final_best_epoch, history = train_early_stopping(
        A_fit, y_fit_scaled, A_val, y_val_scaled, best_config, device
    )
    final_fit_time = time.perf_counter() - final_started
    pd.DataFrame(history).to_csv(out_dir / "final_training_history.csv", index=False)

    # Validation-based explanation is deliberately computed before using the test set.
    importance = validation_permutation_importance(
        model, preprocessor, X_val, y_val, target_mean, target_std, device, repeats=5
    )
    importance.to_csv(out_dir / "validation_permutation_importance.csv", index=False)

    # The held-out test set is used only for this final report and diagnostic figures.
    pred_test_scaled = predict_log_scaled(model, A_test, device)
    pred_test = invert_target(pred_test_scaled, target_mean, target_std)
    test_metrics = regression_metrics(y_test.to_numpy(), pred_test)
    y_val_pred = invert_target(predict_log_scaled(model, A_val, device), target_mean, target_std)
    val_metrics = regression_metrics(y_val.to_numpy(), y_val_pred)

    result = {
        "dataset": dataset_name,
        "csv_path": str(csv_path),
        "n_rows": int(len(X)),
        "n_features_before_encoding": int(X.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split": {"test_size": 0.2, "random_state": SPLIT_SEED},
        "cv": {"n_splits": 5, "shuffle": True, "random_state": SEED},
        "model_seed": SEED,
        "log1p_target": LOG_TARGET,
        "selected_hyperparameters": {
            "name": best_config.name,
            "hidden_dims": list(best_config.hidden_dims),
            "dropout": best_config.dropout,
            "learning_rate": best_config.learning_rate,
            "weight_decay": best_config.weight_decay,
            "batch_size": best_config.batch_size,
            "final_best_epoch": final_best_epoch,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "selected_cv_results": cv_df.iloc[0].to_dict(),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "final_fit_time_sec": final_fit_time,
        "cv_search_wall_time_sec": cv_search_time,
        "device": str(device),
    }
    with open(out_dir / "best_config_and_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(out_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(result["selected_hyperparameters"], f, indent=2, ensure_ascii=False)

    prediction_frame = pd.DataFrame({
        "source_row_index": X_test.index.to_numpy(),
        "actual_price": y_test.to_numpy(),
        "predicted_price": pred_test,
        "error_predicted_minus_actual": pred_test - y_test.to_numpy(),
    })
    prediction_frame.to_csv(out_dir / "test_predictions.csv", index=False)

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": int(A_fit.shape[1]),
        "config": asdict(best_config),
        "target_mean": target_mean,
        "target_std": target_std,
        "log_target": LOG_TARGET,
        "seed": SEED,
    }, out_dir / "best_mlp_model.pt")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    save_plots(dataset_name, out_dir, cv_df, y_test.to_numpy(), pred_test, importance)
    print(f"Final validation RMSE={val_metrics['RMSE']:.3f}; test RMSE={test_metrics['RMSE']:.3f}; "
          f"test MAE={test_metrics['MAE']:.3f}; test R2={test_metrics['R2']:.4f}")
    print(f"Saved artifacts to: {out_dir.resolve()}")
    return {
        "dataset": dataset_name,
        "best_config": best_config.name,
        "cv_RMSE_mean": result["selected_cv_results"]["cv_RMSE_mean"],
        "cv_RMSE_std": result["selected_cv_results"]["cv_RMSE_std"],
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_R2": test_metrics["R2"],
        "final_fit_time_sec": final_fit_time,
        "cv_search_wall_time_sec": cv_search_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible MLP house-price experiments.")
    parser.add_argument("--housing_csv", required=True, help="Path to Kaggle Housing.csv")
    parser.add_argument("--king_county_csv", required=True, help="Path to King County house-sales CSV")
    parser.add_argument("--output_dir", default="results_mlp", help="Directory where results will be saved")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    summaries.append(run_experiment("housing_prices", args.housing_csv, output_dir, device))
    summaries.append(run_experiment("king_county", args.king_county_csv, output_dir, device))
    combined = pd.DataFrame(summaries)
    combined.to_csv(output_dir / "combined_test_summary.csv", index=False)
    with open(output_dir / "experiment_protocol.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_test_split": "80/20, random_state=42",
            "cross_validation": "5-fold KFold on training split, shuffle=True, random_state=36",
            "model_random_seed": 36,
            "target_transform": "log1p(price); all reported metrics inverted to original price scale",
            "metric_selection": "lowest mean CV RMSE in original price units",
            "test_policy": "test set is not used for hyperparameter selection or early stopping",
            "models": [asdict(c) for c in CONFIGS],
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
        }, f, indent=2, ensure_ascii=False)
    print("\nCombined final test results:")
    print(combined.to_string(index=False))
    print(f"\nAll outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
