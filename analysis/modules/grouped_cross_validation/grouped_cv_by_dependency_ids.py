"""Grouped cross-validation for the production water-column random forest.

This analysis keeps all records from the same source study, sampling coordinate,
or mapped lake in the same fold. It uses the production feature set and random-
forest hyperparameters retained in this file from the production training setup.
"""

from __future__ import annotations

import os
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline


REPO_ROOT = Path(os.environ.get("LAKE_MP_REPO_ROOT", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.environ.get("LAKE_MP_DATA_ROOT", REPO_ROOT / "data"))
TRAIN_PATH = Path(os.environ.get("LAKE_MP_TRAIN_DATA", DATA_ROOT / "model_products" / "train_data.csv"))
WATER_PATH = Path(os.environ.get("LAKE_MP_WATER_METADATA", DATA_ROOT / "evidence" / "training_sensitivity" / "water_metadata_quality_frozen.csv"))
GLOBAL_LAKES_PATH = Path(os.environ.get("LAKE_MP_GLOBAL_PREDICTIONS", DATA_ROOT / "model_products" / "global_predictions.csv.gz"))
OUT_DIR = Path(os.environ.get("LAKE_MP_OUTPUT_DIR", REPO_ROOT / "outputs")) / "grouped_cross_validation"

FEATURES = [
    "Lake_area", "Shore_dev", "Vol_total", "Res_time",
    "Total_POP_SERVED", "Average_DF", "Primary_Waste_Discharge",
    "Secondary_Waste_Discharge", "Advanced_Waste_Discharge",
    "RSE_paved", "RSE_gravel", "RSE_other", "prec",
    "emis_tyre_TSP_HEG", "emis_brake_TSP_HEG", "PM2_5", "PM10",
    "Mismanaged", "Total_Plast", "fish_gdp_sqkm", "Cultivated_land",
    "Artificial_surface",
]
TARGET = "ln"
RANDOM_STATE = 42
N_SPLITS = 5
RF_PARAMS = {
    "bootstrap": True,
    "max_depth": 10,
    "max_features": 0.5,
    "min_samples_leaf": 1,
    "min_samples_split": 10,
    "n_estimators": 500,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "oob_score": True,
}


def sequential_ids(values: pd.Series, prefix: str) -> pd.Series:
    """Assign stable IDs in order of first appearance."""
    codes, _ = pd.factorize(values, sort=False)
    return pd.Series(
        [f"{prefix}_{code + 1:03d}" for code in codes], index=values.index
    )


def build_group_map(water: pd.DataFrame) -> pd.DataFrame:
    if pd.isna(water["Reference"].iloc[0]):
        raise ValueError("The first source-study title is missing; forward fill is unsafe.")

    mapping = water[["lon", "lat", "Reference"]].copy()
    mapping.insert(0, "row_id", np.arange(len(mapping), dtype=int))
    mapping["Source_Paper"] = mapping["Reference"].ffill().astype(str).str.strip()
    mapping["Study_ID"] = sequential_ids(mapping["Source_Paper"], "Study")

    coordinate_key = (
        mapping["lon"].round(6).map(lambda value: f"{value:.6f}")
        + "|"
        + mapping["lat"].round(6).map(lambda value: f"{value:.6f}")
    )
    mapping["Sampling_Site_ID"] = sequential_ids(coordinate_key, "Site")

    # The production workflow represents each of the 522,946 predicted lakes by
    # one centroid row in global_predictions.csv and uses great-circle nearest-
    # neighbour matching for observation-to-lake linkage (code/MPB/Test.py).
    # The zero-based row index is therefore the retained production Lake_ID.
    global_lakes = pd.read_csv(GLOBAL_LAKES_PATH, usecols=["lon", "lat"])
    tree = BallTree(
        np.radians(global_lakes[["lat", "lon"]].to_numpy()),
        metric="haversine",
    )
    distance_rad, lake_index = tree.query(
        np.radians(mapping[["lat", "lon"]].to_numpy()), k=1
    )
    lake_index = lake_index[:, 0].astype(int)
    mapping["Lake_ID"] = [f"Lake_{index:06d}" for index in lake_index]
    mapping["Lake_row_index"] = lake_index
    mapping["Lake_match_distance_km"] = distance_rad[:, 0] * 6371.0088
    mapping["Lake_ID_source"] = "nearest_production_lake_centroid"
    return mapping


def evaluate_grouping(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, group_name: str
) -> tuple[dict[str, object], pd.DataFrame]:
    unique_groups = groups.nunique(dropna=False)
    if unique_groups < N_SPLITS:
        raise ValueError(f"{group_name} has only {unique_groups} unique groups.")

    splitter = GroupKFold(n_splits=N_SPLITS)
    predictions = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(**RF_PARAMS)),
            ]
        )
        estimator.fit(X.iloc[train_index], y.iloc[train_index])
        predicted = estimator.predict(X.iloc[test_index])
        predictions[test_index] = predicted
        fold_rows.append(
            {
                "grouping": group_name,
                "fold": fold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "n_test_groups": groups.iloc[test_index].nunique(),
                "R2": r2_score(y.iloc[test_index], predicted),
                "RMSE_ln_items_m-3": mean_squared_error(
                    y.iloc[test_index], predicted
                ) ** 0.5,
                "Pearson_r": pearsonr(y.iloc[test_index], predicted).statistic,
            }
        )

    if np.isnan(predictions).any():
        raise RuntimeError(f"{group_name}: some observations lack out-of-fold predictions.")
    folds = pd.DataFrame(fold_rows)
    pooled = {
        "grouping": group_name,
        "n_samples": len(y),
        "n_groups": unique_groups,
        "n_splits": N_SPLITS,
        "R2_pooled": r2_score(y, predictions),
        "RMSE_pooled_ln_items_m-3": mean_squared_error(y, predictions) ** 0.5,
        "Pearson_r_pooled": pearsonr(y, predictions).statistic,
        "R2_fold_mean": folds["R2"].mean(),
        "R2_fold_sd": folds["R2"].std(ddof=1),
        "RMSE_fold_mean_ln_items_m-3": folds["RMSE_ln_items_m-3"].mean(),
        "RMSE_fold_sd_ln_items_m-3": folds["RMSE_ln_items_m-3"].std(ddof=1),
        "Pearson_r_fold_mean": folds["Pearson_r"].mean(),
        "Pearson_r_fold_sd": folds["Pearson_r"].std(ddof=1),
    }
    prediction_table = pd.DataFrame(
        {
            "row_id": np.arange(len(y), dtype=int),
            "grouping": group_name,
            "group_id": groups.astype(str),
            "observed_ln_items_m-3": y.to_numpy(),
            "predicted_ln_items_m-3": predictions,
        }
    )
    return pooled, pd.concat([folds, prediction_table], ignore_index=True, sort=False)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH).dropna(subset=[TARGET]).reset_index(drop=True)
    water = (pd.read_excel(WATER_PATH, sheet_name="Supplementary_Data_in_water")
             if WATER_PATH.suffix.lower() in {".xlsx", ".xls"}
             else pd.read_csv(WATER_PATH)).reset_index(drop=True)
    if len(train) != len(water):
        raise ValueError(
            f"Training matrix ({len(train)}) and water metadata ({len(water)}) differ."
        )

    mapping = build_group_map(water)
    X = train[FEATURES]
    y = train[TARGET]

    summaries: list[dict[str, object]] = []
    diagnostics: list[pd.DataFrame] = []
    for column in ("Study_ID", "Sampling_Site_ID", "Lake_ID"):
        summary, detail = evaluate_grouping(X, y, mapping[column], column)
        summaries.append(summary)
        diagnostics.append(detail)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUT_DIR / "water_dependency_group_map.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(
        OUT_DIR / "grouped_cv_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(diagnostics, ignore_index=True, sort=False).to_csv(
        OUT_DIR / "grouped_cv_fold_and_prediction_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = [
        f"Python={platform.python_version()}",
        f"scikit-learn={version('scikit-learn')}",
        f"pandas={version('pandas')}",
        f"scipy={version('scipy')}",
        f"n_samples={len(train)}",
        f"n_features={len(FEATURES)}",
        f"random_state={RANDOM_STATE}",
        f"n_splits={N_SPLITS}",
        "RF_PARAMS=" + repr(RF_PARAMS),
        f"global_lake_rows={pd.read_csv(GLOBAL_LAKES_PATH, usecols=['lon']).shape[0]}",
        f"lake_groups={mapping['Lake_ID'].nunique()}",
        f"lake_match_distance_km_median={mapping['Lake_match_distance_km'].median():.6f}",
        f"lake_match_distance_km_max={mapping['Lake_match_distance_km'].max():.6f}",
    ]
    (OUT_DIR / "grouped_cv_run_metadata.txt").write_text(
        "\n".join(metadata) + "\n", encoding="utf-8"
    )

    print(pd.DataFrame(summaries).to_string(index=False))
    print("\n" + "\n".join(metadata))


if __name__ == "__main__":
    main()
