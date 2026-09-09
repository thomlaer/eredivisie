from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "outputs" / "latest" / "historical_features.csv"
OUTPUT_PATH = ROOT / "outputs" / "experiments" / "model_search.csv"
META_COLUMNS = {
    "date",
    "season",
    "kickoff_utc",
    "venue",
    "match_number",
    "match_key",
    "home_goals",
    "away_goals",
    "actual_outcome",
    "target",
    "odds_source",
}
ODDS_COLUMNS = {
    "has_odds",
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_overround",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronological Eredivisie XGBoost model search.")
    parser.add_argument("--quick", action="store_true", help="Run only the compact search grid.")
    return parser.parse_args()


def feature_set(frame: pd.DataFrame, profile: str) -> tuple[list[str], list[str]]:
    candidates = [column for column in frame.columns if column not in META_COLUMNS | ODDS_COLUMNS]
    categorical = [column for column in ("home_team", "away_team") if column in candidates]
    numeric = [
        column
        for column in candidates
        if column not in categorical
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().sum() >= 100
    ]

    if profile == "all":
        return numeric, categorical

    def is_compact(column: str) -> bool:
        if column in {
            "round",
            "month_sin",
            "month_cos",
            "elo_expected_home",
            "elo_diff",
            "h2h_matches",
            "h2h_home_ppg",
            "h2h_home_gd",
        }:
            return True
        if column.startswith("diff_"):
            return not any(token in column for token in ("yellow", "red", "xg_"))
        return column.startswith(("home_form5_", "away_form5_", "home_prev_", "away_prev_")) and not any(
            token in column for token in ("yellow", "red", "xg_")
        )

    numeric = [column for column in numeric if is_compact(column)]
    return numeric, categorical if profile == "compact_teams" else []


def pipeline(numeric: list[str], categorical: list[str], params: dict[str, float | int]) -> Pipeline:
    transformers: list[tuple[str, object, list[str]]] = [
        ("numeric", SimpleImputer(strategy="median"), numeric)
    ]
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=5),
                categorical,
            )
        )
    preprocess = ColumnTransformer(transformers, remainder="drop")
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def adjusted_metrics(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    best: tuple[float, float, float] | None = None
    for multiplier in np.arange(0.80, 1.61, 0.05):
        adjusted = probabilities.copy()
        adjusted[:, 1] *= multiplier
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        accuracy = float(accuracy_score(y, adjusted.argmax(axis=1)))
        loss = float(log_loss(y, adjusted, labels=[0, 1, 2]))
        candidate = (accuracy - 0.015 * loss, accuracy, float(multiplier))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[1], best[2], -1.0


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
    validation = frame[frame["season_start"].isin([2023, 2024])].copy()
    test = frame[frame["season_start"].eq(2025)].copy()

    profiles = ("compact", "compact_teams", "all")
    starts = (2005, 2012, 2015, 2017, 2019)
    if args.quick:
        profiles = ("compact", "compact_teams")
        starts = (2012, 2015, 2017)
    grids = [
        {
            "label": "regularized_shallow",
            "n_estimators": 650,
            "max_depth": 2,
            "learning_rate": 0.03,
            "min_child_weight": 8,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.20,
            "reg_lambda": 5.0,
        },
        {
            "label": "current_no_weights",
            "n_estimators": 550,
            "max_depth": 3,
            "learning_rate": 0.035,
            "min_child_weight": 4,
            "subsample": 0.90,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.05,
            "reg_lambda": 2.5,
        },
        {
            "label": "compact_deeper",
            "n_estimators": 450,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 10,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.30,
            "reg_lambda": 6.0,
        },
    ]
    if args.quick:
        grids = grids[:2]

    rows: list[dict[str, object]] = []
    for profile, start, params in product(profiles, starts, grids):
        train = frame[frame["season_start"].between(start, 2022)].copy()
        numeric, categorical = feature_set(train, profile)
        columns = numeric + categorical
        estimator = pipeline(numeric, categorical, params)
        estimator.fit(train[columns], train["target"])

        val_base = estimator.predict_proba(validation[columns])
        val_accuracy, draw_multiplier, _ = adjusted_metrics(validation["target"].to_numpy(), val_base)
        val_adjusted = val_base.copy()
        val_adjusted[:, 1] *= draw_multiplier
        val_adjusted /= val_adjusted.sum(axis=1, keepdims=True)

        final_train = frame[frame["season_start"].between(start, 2024)].copy()
        final = pipeline(numeric, categorical, params)
        final.fit(final_train[columns], final_train["target"])
        test_probabilities = final.predict_proba(test[columns])
        test_probabilities[:, 1] *= draw_multiplier
        test_probabilities /= test_probabilities.sum(axis=1, keepdims=True)
        rows.append(
            {
                "profile": profile,
                "train_from": start,
                "config": params["label"],
                "features": len(columns),
                "draw_multiplier": draw_multiplier,
                "val_accuracy": val_accuracy,
                "val_log_loss": float(log_loss(validation["target"], val_adjusted, labels=[0, 1, 2])),
                "test_accuracy": float(accuracy_score(test["target"], test_probabilities.argmax(axis=1))),
                "test_log_loss": float(log_loss(test["target"], test_probabilities, labels=[0, 1, 2])),
                "test_draw_rate": float(np.mean(test_probabilities.argmax(axis=1) == 1)),
            }
        )
        print(json.dumps(rows[-1]))

    output = pd.DataFrame(rows).sort_values(
        ["val_accuracy", "val_log_loss"], ascending=[False, True]
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print("\nTop validation configurations")
    print(output.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
