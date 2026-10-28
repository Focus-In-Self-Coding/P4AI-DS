"""Quick end-to-end smoke test using synthetic data only.

This checks the script/output pipeline without needing the Kaggle CSVs. It temporarily
reduces the search to one tiny model and a few epochs. The synthetic scores are NOT
project results and must never be used in the report or submitted as experiment results.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "dl_mlp_experiment.py"
SPEC = importlib.util.spec_from_file_location("mlp_smoke_module", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load experiment script: {SCRIPT}")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def make_data(path: Path, rows: int, include_date: bool) -> None:
    rng = np.random.default_rng(123 if include_date else 456)
    area = rng.integers(500, 4000, size=rows)
    bedrooms = rng.integers(1, 6, size=rows)
    condition = rng.choice(["average", "good", "fair"], size=rows)
    price = 50000 + area * 120 + bedrooms * 12000 + (condition == "good") * 18000
    price = price + rng.normal(0, 15000, size=rows)
    data = {
        "id": np.arange(1, rows + 1),
        "area": area,
        "bedrooms": bedrooms,
        "condition": condition,
        "price": np.maximum(price, 1000),
    }
    if include_date:
        dates = pd.date_range("2014-01-01", periods=rows, freq="D")
        data["date"] = dates.strftime("%Y%m%dT000000")
        data["zipcode"] = rng.choice([98001, 98002, 98003, 98004], size=rows)
    pd.DataFrame(data).to_csv(path, index=False)


def main() -> None:
    # Keep the smoke test fast; this is intentionally not the official experiment setup.
    module.CONFIGS = [module.CONFIGS[0]]
    module.MAX_EPOCHS = 3
    module.EARLY_STOPPING_PATIENCE = 2

    with tempfile.TemporaryDirectory(prefix="mlp_smoke_test_") as tmp:
        tmp_path = Path(tmp)
        housing_path = tmp_path / "Housing.csv"
        king_path = tmp_path / "kc_house_data.csv"
        output_path = tmp_path / "synthetic_smoke_outputs"
        make_data(housing_path, rows=80, include_date=False)
        make_data(king_path, rows=100, include_date=True)

        sys.argv = [
            str(SCRIPT),
            "--housing_csv", str(housing_path),
            "--king_county_csv", str(king_path),
            "--output_dir", str(output_path),
            "--cpu",
        ]
        module.main()

        expected = [
            output_path / "combined_test_summary.csv",
            output_path / "experiment_protocol.json",
            output_path / "housing_prices" / "best_hyperparameters.json",
            output_path / "king_county" / "best_hyperparameters.json",
            output_path / "housing_prices" / "figures" / "test_actual_vs_predicted.png",
            output_path / "king_county" / "figures" / "test_actual_vs_predicted.png",
        ]
        missing = [str(p) for p in expected if not p.exists()]
        if missing:
            raise RuntimeError("Smoke test did not produce expected files:\n" + "\n".join(missing))
        summary = pd.read_csv(output_path / "combined_test_summary.csv")
        if len(summary) != 2:
            raise RuntimeError(f"Expected 2 dataset result rows, got {len(summary)}")

    print("\nSMOKE TEST PASSED.")
    print("The pipeline ran with synthetic data and created expected outputs.")
    print("Do not use these synthetic metrics in your report. Run the official experiment with the Kaggle CSVs.")


if __name__ == "__main__":
    main()
