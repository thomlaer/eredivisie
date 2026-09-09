from __future__ import annotations

import json
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
CONTEXT_PATH = ROOT / "data" / "external" / "transfermarkt_eredivisie_context.csv"
OUTPUT_PATH = ROOT / "outputs" / "experiments" / "transfermarkt_features.csv"
WALK_FORWARD_PATH = ROOT / "outputs" / "experiments" / "transfermarkt_walk_forward.csv"
META = {
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
ODDS = {
    "has_odds",
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_overround",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
}


def compact_base(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_candidates = [
        column
        for column in frame.columns
        if column not in META | ODDS | {"home_team", "away_team"}
        and not column.startswith("tm_")
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().sum() >= 100
    ]

    def keep(column: str) -> bool:
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

    return [column for column in numeric_candidates if keep(column)], []


def context_columns(frame: pd.DataFrame, profile: str) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column.startswith("tm_")
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().sum() >= 100
    ]
    if profile == "none":
        return []
    if profile == "manager":
        return [column for column in columns if "manager_" in column]
    if profile == "value_shape":
        return [
            column
            for column in columns
            if any(token in column for token in ("lineup_value", "lineup_age", "lineup_stability", "share"))
        ]
    if profile == "player_form":
        return [column for column in columns if "prior_players_" in column]
    if profile == "value_form":
        return [
            column
            for column in columns
            if any(token in column for token in ("lineup_value", "lineup_age", "lineup_stability", "share", "prior_players_"))
        ]
    return columns


def estimator(columns: list[str], params: str) -> Pipeline:
    preprocess = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median", add_indicator=True), columns)],
        remainder="drop",
    )
    if params == "shallow":
        model_params = dict(
            n_estimators=650,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=8,
            subsample=0.90,
            colsample_bytree=0.80,
            reg_alpha=0.20,
            reg_lambda=5.0,
        )
    else:
        model_params = dict(
            n_estimators=550,
            max_depth=3,
            learning_rate=0.035,
            min_child_weight=4,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.5,
        )
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        **model_params,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def market_probabilities(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    market = frame[["market_prob_away", "market_prob_draw", "market_prob_home"]].to_numpy(dtype=float)
    available = np.isfinite(market).all(axis=1)
    return market, available


def blend(base: np.ndarray, frame: pd.DataFrame, odds_weight: float, draw_multiplier: float) -> np.ndarray:
    output = base.copy()
    market, available = market_probabilities(frame)
    output[available] = (1.0 - odds_weight) * output[available] + odds_weight * market[available]
    output[:, 1] *= draw_multiplier
    output /= output.sum(axis=1, keepdims=True)
    return output


def tune(frame: pd.DataFrame, base: np.ndarray) -> tuple[float, float, float, float]:
    y = frame["target"].to_numpy()
    best: tuple[float, float, float, float, float] | None = None
    for odds_weight in np.arange(0.0, 0.81, 0.10):
        for draw_multiplier in np.arange(0.80, 1.41, 0.05):
            probabilities = blend(base, frame, float(odds_weight), float(draw_multiplier))
            accuracy = float(accuracy_score(y, probabilities.argmax(axis=1)))
            loss = float(log_loss(y, probabilities, labels=[0, 1, 2]))
            candidate = (accuracy - 0.015 * loss, accuracy, -loss, float(odds_weight), float(draw_multiplier))
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    return best[3], best[4], best[1], -best[2]


def main() -> None:
    features = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
    context = pd.read_csv(CONTEXT_PATH).drop(columns=["date", "home_team", "away_team"])
    frame = features.merge(context, on="match_key", how="left")
    train = frame[frame["season_start"].between(2012, 2022)].copy()
    validation = frame[frame["season_start"].isin([2023, 2024])].copy()
    test = frame[frame["season_start"].eq(2025)].copy()
    base, _ = compact_base(frame)

    rows: list[dict[str, object]] = []
    for profile in ("none", "manager", "value_shape", "player_form", "value_form", "all"):
        for model_profile in ("shallow", "current"):
            columns = base + context_columns(frame, profile)
            model = estimator(columns, model_profile)
            model.fit(train[columns], train["target"])
            validation_base = model.predict_proba(validation[columns])
            odds_weight, draw_multiplier, val_accuracy, val_loss = tune(validation, validation_base)

            final_train = frame[frame["season_start"].between(2012, 2024)].copy()
            final_model = estimator(columns, model_profile)
            final_model.fit(final_train[columns], final_train["target"])
            test_probabilities = blend(
                final_model.predict_proba(test[columns]), test, odds_weight, draw_multiplier
            )
            row = {
                "profile": profile,
                "model": model_profile,
                "features": len(columns),
                "odds_weight": odds_weight,
                "draw_multiplier": draw_multiplier,
                "val_accuracy": val_accuracy,
                "val_log_loss": val_loss,
                "test_accuracy": float(accuracy_score(test["target"], test_probabilities.argmax(axis=1))),
                "test_log_loss": float(log_loss(test["target"], test_probabilities, labels=[0, 1, 2])),
                "test_draw_rate": float(np.mean(test_probabilities.argmax(axis=1) == 1)),
            }
            rows.append(row)
            print(json.dumps(row))

    market, available = market_probabilities(test)
    rows.append(
        {
            "profile": "market_only",
            "model": "market",
            "features": 3,
            "odds_weight": 1.0,
            "draw_multiplier": 1.0,
            "val_accuracy": np.nan,
            "val_log_loss": np.nan,
            "test_accuracy": float(accuracy_score(test.loc[available, "target"], market[available].argmax(axis=1))),
            "test_log_loss": float(log_loss(test.loc[available, "target"], market[available], labels=[0, 1, 2])),
            "test_draw_rate": float(np.mean(market[available].argmax(axis=1) == 1)),
        }
    )
    output = pd.DataFrame(rows).sort_values(["val_accuracy", "val_log_loss"], ascending=[False, True])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print("\nResults sorted by validation")
    print(output.to_string(index=False))

    walk_rows: list[dict[str, object]] = []
    for test_season in range(2019, 2026):
        validation_seasons = [test_season - 2, test_season - 1]
        fold_train = frame[frame["season_start"].between(2012, test_season - 3)].copy()
        fold_validation = frame[frame["season_start"].isin(validation_seasons)].copy()
        fold_test = frame[frame["season_start"].eq(test_season)].copy()
        for profile in ("none", "value_shape", "value_form", "all"):
            columns = base + context_columns(frame, profile)
            tuning_model = estimator(columns, "shallow")
            tuning_model.fit(fold_train[columns], fold_train["target"])
            odds_weight, draw_multiplier, _, _ = tune(
                fold_validation, tuning_model.predict_proba(fold_validation[columns])
            )
            final_train = frame[frame["season_start"].between(2012, test_season - 1)].copy()
            final_model = estimator(columns, "shallow")
            final_model.fit(final_train[columns], final_train["target"])
            base_probabilities = final_model.predict_proba(fold_test[columns])
            probabilities = blend(base_probabilities, fold_test, odds_weight, draw_multiplier)
            walk_rows.append(
                {
                    "test_season": test_season,
                    "profile": profile,
                    "rows": len(fold_test),
                    "odds_weight": odds_weight,
                    "draw_multiplier": draw_multiplier,
                    "accuracy": float(accuracy_score(fold_test["target"], probabilities.argmax(axis=1))),
                    "log_loss": float(log_loss(fold_test["target"], probabilities, labels=[0, 1, 2])),
                    "no_odds_accuracy": float(
                        accuracy_score(fold_test["target"], base_probabilities.argmax(axis=1))
                    ),
                    "no_odds_log_loss": float(
                        log_loss(fold_test["target"], base_probabilities, labels=[0, 1, 2])
                    ),
                }
            )
            print(json.dumps(walk_rows[-1]))

    walk_output = pd.DataFrame(walk_rows)
    walk_output.to_csv(WALK_FORWARD_PATH, index=False)
    summary = (
        walk_output.groupby("profile", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "rows": int(group["rows"].sum()),
                    "mean_accuracy": np.average(group["accuracy"], weights=group["rows"]),
                    "mean_log_loss": np.average(group["log_loss"], weights=group["rows"]),
                    "mean_no_odds_accuracy": np.average(group["no_odds_accuracy"], weights=group["rows"]),
                    "mean_no_odds_log_loss": np.average(group["no_odds_log_loss"], weights=group["rows"]),
                    "seasons_won_vs_none": int(
                        sum(
                            row.accuracy
                            > walk_output[
                                (walk_output["profile"].eq("none"))
                                & (walk_output["test_season"].eq(row.test_season))
                            ]["accuracy"].iloc[0]
                            for row in group.itertuples(index=False)
                        )
                    ),
                }
            ),
            include_groups=False,
        )
        .sort_values(["mean_accuracy", "mean_log_loss"], ascending=[False, True])
    )
    print("\nWalk-forward summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
