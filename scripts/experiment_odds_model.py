from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from experiment_transfermarkt_features import compact_base, context_columns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "experiments" / "direct_odds_model.csv"
ODDS_COLUMNS = [
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_overround",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
]
SCORE_MARKET_COLUMNS = [
    "score_market_prob_over25",
    "score_market_total_overround",
    "score_market_handicap",
    "score_market_handicap_home_prob",
    "score_market_handicap_away_prob",
    "score_market_total_lambda",
    "score_market_goal_diff_proxy",
    "score_market_home_xg_proxy",
    "score_market_away_xg_proxy",
    "score_market_has_totals",
    "score_market_has_handicap",
]


def estimator(columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("numeric", SimpleImputer(strategy="median", add_indicator=True), columns)],
                    remainder="drop",
                ),
            ),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=3,
                    n_estimators=500,
                    max_depth=2,
                    learning_rate=0.025,
                    min_child_weight=10,
                    subsample=0.90,
                    colsample_bytree=0.80,
                    reg_alpha=0.30,
                    reg_lambda=6.0,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def adjust(probabilities: np.ndarray, multiplier: float) -> np.ndarray:
    output = probabilities.copy()
    output[:, 1] *= multiplier
    output /= output.sum(axis=1, keepdims=True)
    return output


def tune_draw(frame: pd.DataFrame, probabilities: np.ndarray) -> tuple[float, float, float]:
    y = frame["target"].to_numpy()
    best: tuple[float, float, float, float] | None = None
    for multiplier in np.arange(0.70, 1.61, 0.05):
        candidate_probabilities = adjust(probabilities, float(multiplier))
        accuracy = float(accuracy_score(y, candidate_probabilities.argmax(axis=1)))
        loss = float(log_loss(y, candidate_probabilities, labels=[0, 1, 2]))
        candidate = (accuracy - 0.015 * loss, accuracy, -loss, float(multiplier))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[3], best[1], -best[2]


def main() -> None:
    features = pd.read_csv(ROOT / "outputs" / "latest" / "historical_features.csv")
    context = pd.read_csv(ROOT / "data" / "external" / "transfermarkt_eredivisie_context.csv").drop(
        columns=["date", "home_team", "away_team"]
    )
    frame = features.merge(context, on="match_key", how="left")
    compact, _ = compact_base(frame)
    profiles = {
        "odds_only": ODDS_COLUMNS,
        "odds_plus_score_markets": ODDS_COLUMNS + SCORE_MARKET_COLUMNS,
        "compact_plus_odds": compact + ODDS_COLUMNS,
        "compact_plus_all_markets": compact + ODDS_COLUMNS + SCORE_MARKET_COLUMNS,
        "value_form_plus_odds": compact + context_columns(frame, "value_form") + ODDS_COLUMNS,
    }
    rows: list[dict[str, object]] = []
    for test_season in range(2019, 2026):
        validation_seasons = [test_season - 2, test_season - 1]
        for profile, columns in profiles.items():
            train = frame[
                frame["season_start"].between(2012, test_season - 3) & frame["has_odds"].eq(1)
            ]
            validation = frame[
                frame["season_start"].isin(validation_seasons) & frame["has_odds"].eq(1)
            ]
            test = frame[frame["season_start"].eq(test_season) & frame["has_odds"].eq(1)]
            tuning_model = estimator(columns)
            tuning_model.fit(train[columns], train["target"])
            multiplier, _, _ = tune_draw(validation, tuning_model.predict_proba(validation[columns]))

            final_train = frame[
                frame["season_start"].between(2012, test_season - 1) & frame["has_odds"].eq(1)
            ]
            final_model = estimator(columns)
            final_model.fit(final_train[columns], final_train["target"])
            probabilities = adjust(final_model.predict_proba(test[columns]), multiplier)
            market = test[["market_prob_away", "market_prob_draw", "market_prob_home"]].to_numpy(dtype=float)
            row = {
                "test_season": test_season,
                "profile": profile,
                "rows": len(test),
                "draw_multiplier": multiplier,
                "accuracy": float(accuracy_score(test["target"], probabilities.argmax(axis=1))),
                "log_loss": float(log_loss(test["target"], probabilities, labels=[0, 1, 2])),
                "market_accuracy": float(accuracy_score(test["target"], market.argmax(axis=1))),
                "market_log_loss": float(log_loss(test["target"], market, labels=[0, 1, 2])),
            }
            rows.append(row)
            print(json.dumps(row))

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT, index=False)
    summary = (
        output.groupby("profile", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "rows": int(group["rows"].sum()),
                    "accuracy": np.average(group["accuracy"], weights=group["rows"]),
                    "log_loss": np.average(group["log_loss"], weights=group["rows"]),
                    "market_accuracy": np.average(group["market_accuracy"], weights=group["rows"]),
                    "market_log_loss": np.average(group["market_log_loss"], weights=group["rows"]),
                    "seasons_beating_market": int(sum(group["accuracy"] > group["market_accuracy"])),
                }
            ),
            include_groups=False,
        )
        .sort_values(["accuracy", "log_loss"], ascending=[False, True])
    )
    print("\nDirect odds model walk-forward summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
