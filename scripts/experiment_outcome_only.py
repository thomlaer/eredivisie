from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction.modeling import (
    OUTCOME_TO_ID,
    align,
    fit_bundle,
    market_probabilities,
    poisson_probabilities,
    predict_expected_goals,
)


FEATURES = ROOT / "outputs" / "latest" / "historical_features.csv"
OUTPUT = ROOT / "outputs" / "experiments" / "outcome_only_strategy.csv"


def poisson_outcomes(home_xg: np.ndarray, away_xg: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for home, away in zip(home_xg, away_xg):
        matrix = np.outer(poisson_probabilities(float(home)), poisson_probabilities(float(away)))
        rows.append([float(np.triu(matrix, 1).sum()), float(np.trace(matrix)), float(np.tril(matrix, -1).sum())])
    output = np.asarray(rows, dtype=float)
    return output / output.sum(axis=1, keepdims=True)


def model_parts(bundle, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = align(frame, bundle.numeric_features, bundle.categorical_features)
    xgb = bundle.classifier.predict_proba(x)
    market, market_available = market_probabilities(frame)
    home_xg, away_xg = predict_expected_goals(bundle, frame)
    poisson = poisson_outcomes(home_xg, away_xg)
    return xgb, market, market_available, poisson


def combine(
    xgb: np.ndarray,
    market: np.ndarray,
    market_available: np.ndarray,
    poisson: np.ndarray,
    market_weight: float,
    poisson_weight: float,
    away_multiplier: float,
    draw_multiplier: float,
) -> np.ndarray:
    output = (1.0 - poisson_weight) * xgb + poisson_weight * poisson
    if market_weight > 0 and market_available.any():
        output[market_available] = (
            (1.0 - market_weight) * output[market_available]
            + market_weight * market[market_available]
        )
    output[:, OUTCOME_TO_ID["away_win"]] *= away_multiplier
    output[:, OUTCOME_TO_ID["draw"]] *= draw_multiplier
    return output / output.sum(axis=1, keepdims=True)


def metrics(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    predicted = probabilities.argmax(axis=1)
    return (
        float(accuracy_score(y, predicted)),
        float(log_loss(y, probabilities, labels=[0, 1, 2])),
        float(np.mean(predicted == OUTCOME_TO_ID["draw"])),
    )


def tune(
    y: np.ndarray,
    xgb: np.ndarray,
    market: np.ndarray,
    market_available: np.ndarray,
    poisson: np.ndarray,
    *,
    use_market: bool,
) -> dict[str, float]:
    market_weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if use_market else [0.0]
    candidates: list[dict[str, float]] = []
    for market_weight, poisson_weight, away_multiplier, draw_multiplier in itertools.product(
        market_weights,
        [0.0, 0.15, 0.30, 0.45],
        [0.85, 0.95, 1.00, 1.05, 1.15],
        [0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40, 1.60],
    ):
        probabilities = combine(
            xgb,
            market,
            market_available,
            poisson,
            market_weight,
            poisson_weight,
            away_multiplier,
            draw_multiplier,
        )
        accuracy, loss, draw_rate = metrics(y, probabilities)
        candidates.append(
            {
                "market_weight": market_weight,
                "poisson_weight": poisson_weight,
                "away_multiplier": away_multiplier,
                "draw_multiplier": draw_multiplier,
                "accuracy": accuracy,
                "log_loss": loss,
                "draw_rate": draw_rate,
            }
        )
    return max(candidates, key=lambda row: (row["accuracy"], -row["log_loss"]))


def close_second_predictions(probabilities: np.ndarray, margin: float) -> np.ndarray:
    order = np.argsort(probabilities, axis=1)
    top = order[:, -1]
    second = order[:, -2]
    top_probability = probabilities[np.arange(len(probabilities)), top]
    second_probability = probabilities[np.arange(len(probabilities)), second]
    use_second = (top_probability - second_probability) <= margin
    return np.where(use_second, second, top)


def run_season(features: pd.DataFrame, test_season: int) -> list[dict[str, float | int | str]]:
    validation_seasons = {test_season - 2, test_season - 1}
    train = features[features["season_start"] < min(validation_seasons)].copy()
    validation = features[features["season_start"].isin(validation_seasons)].copy()
    final_train = features[features["season_start"] < test_season].copy()
    test = features[features["season_start"].eq(test_season)].copy()
    if min(map(len, (train, validation, final_train, test))) == 0:
        return []

    tuning_bundle = fit_bundle(train, draw_multiplier=1.0, odds_weight=0.0)
    validation_parts = model_parts(tuning_bundle, validation)
    selected = tune(
        validation["target"].to_numpy(dtype=int),
        *validation_parts,
        use_market=True,
    )
    selected_no_odds = tune(
        validation["target"].to_numpy(dtype=int),
        *validation_parts,
        use_market=False,
    )

    final_bundle = fit_bundle(final_train, draw_multiplier=1.0, odds_weight=0.0)
    xgb, market, available, poisson = model_parts(final_bundle, test)
    y = test["target"].to_numpy(dtype=int)
    rows: list[dict[str, float | int | str]] = []

    strategies = {
        "xgboost_raw": combine(xgb, market, available, poisson, 0.0, 0.0, 1.0, 1.0),
        "market_only": market,
        "selected_outcome": combine(
            xgb,
            market,
            available,
            poisson,
            selected["market_weight"],
            selected["poisson_weight"],
            selected["away_multiplier"],
            selected["draw_multiplier"],
        ),
        "selected_no_odds": combine(
            xgb,
            market,
            available,
            poisson,
            0.0,
            selected_no_odds["poisson_weight"],
            selected_no_odds["away_multiplier"],
            selected_no_odds["draw_multiplier"],
        ),
    }
    for draw_multiplier in (0.75, 0.85, 0.95, 1.00, 1.10, 1.20, 1.30):
        strategies[f"no_odds_draw_{draw_multiplier:.2f}"] = combine(
            xgb, market, available, poisson, 0.0, 0.0, 1.0, draw_multiplier
        )
    for strategy, probabilities in strategies.items():
        mask = available if strategy == "market_only" else np.ones(len(test), dtype=bool)
        accuracy, loss, draw_rate = metrics(y[mask], probabilities[mask])
        rows.append(
            {
                "test_season": test_season,
                "strategy": strategy,
                "rows": int(mask.sum()),
                "accuracy": accuracy,
                "log_loss": loss,
                "draw_rate": draw_rate,
                "market_weight": selected["market_weight"] if strategy == "selected_outcome" else 0.0,
                "poisson_weight": (
                    selected["poisson_weight"] if strategy == "selected_outcome" else
                    selected_no_odds["poisson_weight"] if strategy == "selected_no_odds" else 0.0
                ),
                "away_multiplier": (
                    selected["away_multiplier"] if strategy == "selected_outcome" else
                    selected_no_odds["away_multiplier"] if strategy == "selected_no_odds" else 1.0
                ),
                "draw_multiplier": (
                    selected["draw_multiplier"] if strategy == "selected_outcome" else
                    selected_no_odds["draw_multiplier"] if strategy == "selected_no_odds" else 1.0
                ),
            }
        )

    selected_probabilities = strategies["selected_outcome"]
    for margin in (0.03, 0.05, 0.08, 0.10):
        predicted = close_second_predictions(selected_probabilities, margin)
        rows.append(
            {
                "test_season": test_season,
                "strategy": f"wk_close_second_{margin:.2f}",
                "rows": int(len(test)),
                "accuracy": float(accuracy_score(y, predicted)),
                "log_loss": float(log_loss(y, selected_probabilities, labels=[0, 1, 2])),
                "draw_rate": float(np.mean(predicted == OUTCOME_TO_ID["draw"])),
                "market_weight": selected["market_weight"],
                "poisson_weight": selected["poisson_weight"],
                "away_multiplier": selected["away_multiplier"],
                "draw_multiplier": selected["draw_multiplier"],
            }
        )
    return rows


def main() -> None:
    features = pd.read_csv(FEATURES, low_memory=False)
    rows: list[dict[str, float | int | str]] = []
    for season in range(2019, 2026):
        print(f"Testing season {season}/{str(season + 1)[-2:]}")
        rows.extend(run_season(features, season))
    output = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    summary = (
        output.groupby("strategy", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "rows": int(group["rows"].sum()),
                    "accuracy": float(np.average(group["accuracy"], weights=group["rows"])),
                    "log_loss": float(np.average(group["log_loss"], weights=group["rows"])),
                    "draw_rate": float(np.average(group["draw_rate"], weights=group["rows"])),
                }
            ),
            include_groups=False,
        )
        .sort_values(["accuracy", "log_loss"], ascending=[False, True])
    )
    print(summary.to_string(index=False))
    print(json.dumps({"output": str(OUTPUT), "rows": len(output)}, indent=2))


if __name__ == "__main__":
    main()
