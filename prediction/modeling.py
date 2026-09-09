from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

from .config import (
    OUTCOME_DRAW_MULTIPLIER_NO_ODDS,
    OUTCOME_DRAW_MULTIPLIER_WITH_ODDS,
    OUTCOME_ODDS_WEIGHT,
    RECENCY_HALF_LIFE_DAYS,
    RECENCY_MIN_WEIGHT,
    RANDOM_SEED,
)
from .features import ID_TO_OUTCOME, ODDS_FEATURES, OUTCOME_TO_ID, SCORE_MARKET_FEATURES, outcome_name


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
CATEGORICAL_COLUMNS = ["home_team", "away_team"]


@dataclass
class ModelBundle:
    classifier: Pipeline
    home_score_model: Pipeline
    away_score_model: Pipeline
    numeric_features: list[str]
    categorical_features: list[str]
    score_numeric_features: list[str]
    score_categorical_features: list[str]
    draw_multiplier: float
    draw_multiplier_no_odds: float
    odds_weight: float


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [column for column in frame.columns if column not in META_COLUMNS]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in candidates]
    numeric = [
        column
        for column in candidates
        if column not in categorical
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().sum() >= min(100, max(1, len(frame) // 50))
    ]
    return numeric, categorical


def base_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = feature_columns(frame)
    numeric = [column for column in numeric if column not in ODDS_FEATURES + SCORE_MARKET_FEATURES]

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
        if column.startswith(("home_form5_", "away_form5_", "home_prev_", "away_prev_")):
            return not any(token in column for token in ("yellow", "red", "xg_"))
        if column.startswith("tm_"):
            return any(
                token in column
                for token in (
                    "lineup_value",
                    "lineup_age",
                    "lineup_stability",
                    "_share",
                    "prior_players_",
                )
            )
        return False

    return [column for column in numeric if keep(column)], []


def score_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = base_feature_columns(frame)
    minimum_rows = min(100, max(1, len(frame) // 50))
    # Historical walk-forward testing found the no-vig 1X2 probabilities more
    # reliable for exact scores than the noisier over/under and AH columns.
    numeric.extend(
        column
        for column in ("market_prob_home", "market_prob_draw", "market_prob_away")
        if column in frame and frame[column].notna().sum() >= minimum_rows
    )
    return list(dict.fromkeys(numeric)), categorical


def _preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=5)
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            ("categorical", one_hot, categorical_features),
        ],
        remainder="drop",
    )


def classifier_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        n_estimators=650,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.80,
        min_child_weight=8,
        reg_alpha=0.20,
        reg_lambda=5.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline([("preprocess", _preprocessor(numeric_features, categorical_features)), ("model", model)])


def score_pipeline(numeric_features: list[str], categorical_features: list[str], seed: int) -> Pipeline:
    model = XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        n_estimators=550,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.80,
        min_child_weight=8,
        reg_alpha=0.15,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline([("preprocess", _preprocessor(numeric_features, categorical_features)), ("model", model)])


def align(frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in numeric_features:
        if column not in output:
            output[column] = np.nan
    for column in categorical_features:
        if column not in output:
            output[column] = "Unknown"
    return output[numeric_features + categorical_features]


def sample_weights(dates: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(dates, errors="coerce")
    max_date = parsed.max()
    ages = (max_date - parsed).dt.days.fillna(RECENCY_HALF_LIFE_DAYS).clip(lower=0).to_numpy(dtype=float)
    weights = np.exp(-math.log(2.0) * ages / RECENCY_HALF_LIFE_DAYS)
    weights = np.maximum(weights, RECENCY_MIN_WEIGHT)
    return weights / weights.mean()


def adjust_draw(probabilities: np.ndarray, multiplier: float) -> np.ndarray:
    adjusted = np.asarray(probabilities, dtype=float).copy()
    adjusted[:, OUTCOME_TO_ID["draw"]] *= multiplier
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def market_probabilities(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns = ["market_prob_away", "market_prob_draw", "market_prob_home"]
    market = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    available = np.isfinite(market).all(axis=1) & (market.sum(axis=1) > 0)
    if available.any():
        market[available] /= market[available].sum(axis=1, keepdims=True)
    return market, available


def blend_probabilities(
    base: np.ndarray,
    frame: pd.DataFrame,
    odds_weight: float,
    draw_multiplier: float,
    draw_multiplier_no_odds: float | None = None,
) -> np.ndarray:
    output = np.asarray(base, dtype=float).copy()
    market, available = market_probabilities(frame)
    if odds_weight > 0 and available.any():
        output[available] = (1.0 - odds_weight) * output[available] + odds_weight * market[available]
    no_odds_multiplier = draw_multiplier if draw_multiplier_no_odds is None else draw_multiplier_no_odds
    multipliers = np.where(available, draw_multiplier, no_odds_multiplier)
    output[:, OUTCOME_TO_ID["draw"]] *= multipliers
    output /= output.sum(axis=1, keepdims=True)
    return output


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = probabilities.argmax(axis=1)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1, 2]).astype(int).tolist(),
        "predicted_draw_rate": float(np.mean(predicted == OUTCOME_TO_ID["draw"])),
        "actual_draw_rate": float(np.mean(y_true == OUTCOME_TO_ID["draw"])),
    }


def tune_probability_layer(
    validation: pd.DataFrame,
    y_validation: np.ndarray,
    base_probabilities: np.ndarray,
) -> tuple[float, float, list[dict[str, float]]]:
    candidates: list[dict[str, float]] = []
    for odds_weight in np.arange(0.0, 1.01, 0.10):
        for draw_multiplier in (0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30):
            probabilities = blend_probabilities(
                base_probabilities,
                validation,
                float(odds_weight),
                float(draw_multiplier),
            )
            metrics = _classification_metrics(y_validation, probabilities)
            selection_score = metrics["accuracy"] - 0.015 * metrics["log_loss"]
            candidates.append(
                {
                    "odds_weight": float(odds_weight),
                    "draw_multiplier": float(draw_multiplier),
                    "accuracy": metrics["accuracy"],
                    "log_loss": metrics["log_loss"],
                    "selection_score": selection_score,
                }
            )
    candidates.sort(key=lambda item: (item["selection_score"], item["accuracy"], -item["log_loss"]), reverse=True)
    best = candidates[0]
    return best["draw_multiplier"], best["odds_weight"], candidates


def tune_no_odds_draw(
    y_validation: np.ndarray,
    base_probabilities: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    candidates: list[dict[str, float]] = []
    for draw_multiplier in np.arange(0.80, 1.61, 0.05):
        probabilities = adjust_draw(base_probabilities, float(draw_multiplier))
        metrics = _classification_metrics(y_validation, probabilities)
        selection_score = metrics["accuracy"] - 0.015 * metrics["log_loss"]
        candidates.append(
            {
                "draw_multiplier": float(draw_multiplier),
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "predicted_draw_rate": metrics["predicted_draw_rate"],
                "selection_score": selection_score,
            }
        )
    candidates.sort(key=lambda item: (item["selection_score"], item["accuracy"], -item["log_loss"]), reverse=True)
    return candidates[0]["draw_multiplier"], candidates


def fit_bundle(
    train: pd.DataFrame,
    *,
    draw_multiplier: float,
    draw_multiplier_no_odds: float | None = None,
    odds_weight: float,
) -> ModelBundle:
    numeric_features, categorical_features = base_feature_columns(train)
    score_numeric_features, score_categorical_features = score_feature_columns(train)
    x_train = align(train, numeric_features, categorical_features)
    classifier = classifier_pipeline(numeric_features, categorical_features)
    classifier.fit(x_train, train["target"].to_numpy())
    x_score_train = align(train, score_numeric_features, score_categorical_features)
    home_score_model = score_pipeline(score_numeric_features, score_categorical_features, RANDOM_SEED + 1)
    away_score_model = score_pipeline(score_numeric_features, score_categorical_features, RANDOM_SEED + 2)
    home_score_model.fit(x_score_train, train["home_goals"].to_numpy(dtype=float))
    away_score_model.fit(x_score_train, train["away_goals"].to_numpy(dtype=float))
    return ModelBundle(
        classifier=classifier,
        home_score_model=home_score_model,
        away_score_model=away_score_model,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        score_numeric_features=score_numeric_features,
        score_categorical_features=score_categorical_features,
        draw_multiplier=draw_multiplier,
        draw_multiplier_no_odds=(
            draw_multiplier if draw_multiplier_no_odds is None else draw_multiplier_no_odds
        ),
        odds_weight=odds_weight,
    )


def predict_probabilities(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    x = align(frame, bundle.numeric_features, bundle.categorical_features)
    base = bundle.classifier.predict_proba(x)
    return blend_probabilities(
        base,
        frame,
        bundle.odds_weight,
        bundle.draw_multiplier,
        bundle.draw_multiplier_no_odds,
    )


def predict_expected_goals(bundle: ModelBundle, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = align(frame, bundle.score_numeric_features, bundle.score_categorical_features)
    home = np.clip(bundle.home_score_model.predict(x), 0.05, 6.5)
    away = np.clip(bundle.away_score_model.predict(x), 0.05, 6.5)
    return home, away


def poisson_probabilities(expected_goals: float, max_goals: int = 8) -> np.ndarray:
    expected_goals = float(np.clip(expected_goals, 0.05, 8.0))
    probabilities = np.zeros(max_goals + 1, dtype=float)
    probabilities[0] = math.exp(-expected_goals)
    for goals in range(1, max_goals + 1):
        probabilities[goals] = probabilities[goals - 1] * expected_goals / goals
    probabilities /= probabilities.sum()
    return probabilities


def best_score_for_outcome(home_xg: float, away_xg: float, outcome: str) -> tuple[int, int, float]:
    matrix = np.outer(poisson_probabilities(home_xg), poisson_probabilities(away_xg))
    home_grid, away_grid = np.indices(matrix.shape)
    if outcome == "home_win":
        valid = home_grid > away_grid
    elif outcome == "away_win":
        valid = home_grid < away_grid
    else:
        valid = home_grid == away_grid
    constrained = np.where(valid, matrix, -1.0)
    home_goals, away_goals = np.unravel_index(int(np.argmax(constrained)), constrained.shape)
    return int(home_goals), int(away_goals), float(matrix[home_goals, away_goals])


def prediction_frame(bundle: ModelBundle, frame: pd.DataFrame, *, include_score: bool = False) -> pd.DataFrame:
    probabilities = predict_probabilities(bundle, frame)
    home_xg, away_xg = predict_expected_goals(bundle, frame)
    predicted_ids = probabilities.argmax(axis=1)
    predicted_outcomes = [ID_TO_OUTCOME[int(value)] for value in predicted_ids]
    columns = [
        column
        for column in (
            "match_key",
            "match_number",
            "round",
            "date",
            "kickoff_utc",
            "venue",
            "home_team",
            "away_team",
            "odds_source",
            "odds_home",
            "odds_draw",
            "odds_away",
        )
        if column in frame.columns
    ]
    output = frame[columns].reset_index(drop=True).copy()
    output["predicted_outcome"] = predicted_outcomes
    output["predicted_winner"] = [
        row.home_team if outcome == "home_win" else row.away_team if outcome == "away_win" else "Gelijk"
        for row, outcome in zip(output.itertuples(index=False), predicted_outcomes)
    ]
    output["prob_away_win"] = probabilities[:, OUTCOME_TO_ID["away_win"]]
    output["prob_draw"] = probabilities[:, OUTCOME_TO_ID["draw"]]
    output["prob_home_win"] = probabilities[:, OUTCOME_TO_ID["home_win"]]
    output["expected_home_goals"] = home_xg
    output["expected_away_goals"] = away_xg
    if include_score:
        scores = [
            best_score_for_outcome(float(hxg), float(axg), outcome)
            for hxg, axg, outcome in zip(home_xg, away_xg, predicted_outcomes)
        ]
        output["predicted_home_score"] = [score[0] for score in scores]
        output["predicted_away_score"] = [score[1] for score in scores]
        output["predicted_score"] = [f"{score[0]}-{score[1]}" for score in scores]
        output["predicted_score_probability"] = [score[2] for score in scores]
    output["probability_source"] = [
        (
            "market_calibrated" if bundle.odds_weight >= 0.999
            else "xgboost+odds"
        ) if int(getattr(row, "has_odds", 0) or 0) and bundle.odds_weight > 0 else "xgboost"
        for row in frame.itertuples(index=False)
    ]
    return output


def score_metrics(frame: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float]:
    actual_home = frame["home_goals"].to_numpy(dtype=int)
    actual_away = frame["away_goals"].to_numpy(dtype=int)
    predicted_home = predictions["predicted_home_score"].to_numpy(dtype=int)
    predicted_away = predictions["predicted_away_score"].to_numpy(dtype=int)
    actual_outcome = [outcome_name(home, away) for home, away in zip(actual_home, actual_away)]
    predicted_outcome = predictions["predicted_outcome"].tolist()
    return {
        "exact_score_accuracy": float(np.mean((actual_home == predicted_home) & (actual_away == predicted_away))),
        "score_outcome_accuracy": float(np.mean(np.asarray(actual_outcome) == np.asarray(predicted_outcome))),
        "home_goals_mae": float(mean_absolute_error(actual_home, predictions["expected_home_goals"])),
        "away_goals_mae": float(mean_absolute_error(actual_away, predictions["expected_away_goals"])),
        "total_goals_mae": float(
            mean_absolute_error(actual_home + actual_away, predictions["expected_home_goals"] + predictions["expected_away_goals"])
        ),
    }


def feature_importance(bundle: ModelBundle, limit: int = 25) -> list[dict[str, float | str]]:
    preprocessor = bundle.classifier.named_steps["preprocess"]
    model = bundle.classifier.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:limit]
    return [
        {"feature": str(names[index]).replace("numeric__", "").replace("categorical__", ""), "importance": float(importances[index])}
        for index in order
    ]


def evaluate_and_tune(
    features: pd.DataFrame,
    validation_seasons: set[int],
    test_season: int,
) -> tuple[dict[str, Any], float, float, float]:
    train = features[features["season_start"] < min(validation_seasons)].copy()
    validation = features[features["season_start"].isin(validation_seasons)].copy()
    test = features[features["season_start"] == test_season].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological train/validation/test split is empty.")

    tuning_bundle = fit_bundle(train, draw_multiplier=1.0, odds_weight=0.0)
    validation_x = align(validation, tuning_bundle.numeric_features, tuning_bundle.categorical_features)
    validation_base = tuning_bundle.classifier.predict_proba(validation_x)
    _, _, tuning_rows = tune_probability_layer(
        validation,
        validation["target"].to_numpy(),
        validation_base,
    )
    _, no_odds_tuning_rows = tune_no_odds_draw(
        validation["target"].to_numpy(),
        validation_base,
    )
    draw_multiplier = OUTCOME_DRAW_MULTIPLIER_WITH_ODDS
    draw_multiplier_no_odds = OUTCOME_DRAW_MULTIPLIER_NO_ODDS
    odds_weight = OUTCOME_ODDS_WEIGHT

    test_train = features[features["season_start"] < test_season].copy()
    test_bundle = fit_bundle(
        test_train,
        draw_multiplier=draw_multiplier,
        draw_multiplier_no_odds=draw_multiplier_no_odds,
        odds_weight=odds_weight,
    )
    test_probabilities = predict_probabilities(test_bundle, test)
    base_test = test_bundle.classifier.predict_proba(
        align(test, test_bundle.numeric_features, test_bundle.categorical_features)
    )
    market, market_available = market_probabilities(test)
    metrics: dict[str, Any] = {
        "split": {
            "train_seasons_through": int(min(validation_seasons) - 1),
            "validation_seasons": sorted(validation_seasons),
            "test_season": int(test_season),
            "rows_train": int(len(train)),
            "rows_validation": int(len(validation)),
            "rows_test": int(len(test)),
        },
        "selected": {
            "strategy": "outcome_only_walk_forward_fixed",
            "draw_multiplier": draw_multiplier,
            "draw_multiplier_no_odds": draw_multiplier_no_odds,
            "odds_weight": odds_weight,
            **_classification_metrics(test["target"].to_numpy(), test_probabilities),
        },
        "xgboost_without_odds": _classification_metrics(
            test["target"].to_numpy(), adjust_draw(base_test, draw_multiplier_no_odds)
        ),
        "market_only": _classification_metrics(
            test.loc[market_available, "target"].to_numpy(), market[market_available]
        ) if market_available.any() else {"rows": 0},
        "tuning_top": tuning_rows[:10],
        "no_odds_tuning_top": no_odds_tuning_rows[:10],
        "feature_importance": feature_importance(test_bundle),
    }
    return metrics, draw_multiplier, draw_multiplier_no_odds, odds_weight


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
