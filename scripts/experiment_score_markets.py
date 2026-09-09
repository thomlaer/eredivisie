from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from experiment_transfermarkt_features import blend, compact_base, estimator, tune

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prediction.modeling import best_score_for_outcome  # noqa: E402
from prediction.names import canonical_team, name_key  # noqa: E402


OUTPUT = ROOT / "outputs" / "experiments" / "score_market_features.csv"


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def first_valid(frame: pd.DataFrame, choices: list[str]) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in choices:
        values = numeric(frame, column)
        output = output.where(output.notna(), values)
    return output


def poisson_total_lambda(over_probability: float) -> float:
    if not np.isfinite(over_probability):
        return np.nan
    target = float(np.clip(over_probability, 0.01, 0.99))
    low, high = 0.05, 8.0
    for _ in range(35):
        middle = (low + high) / 2.0
        over = 1.0 - math.exp(-middle) * (1.0 + middle + middle * middle / 2.0)
        if over < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def load_score_markets() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted((ROOT / "data" / "raw" / "football_data").glob("N1_*.csv")):
        raw = pd.read_csv(path, encoding="cp1252", on_bad_lines="skip")
        if not {"Date", "HomeTeam", "AwayTeam"}.issubset(raw.columns):
            continue
        date = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed")
        over_odds = first_valid(raw, ["AvgC>2.5", "PC>2.5", "B365C>2.5", "Avg>2.5", "P>2.5", "B365>2.5", "BbAv>2.5"])
        under_odds = first_valid(raw, ["AvgC<2.5", "PC<2.5", "B365C<2.5", "Avg<2.5", "P<2.5", "B365<2.5", "BbAv<2.5"])
        over_implied = 1.0 / over_odds
        under_implied = 1.0 / under_odds
        total_overround = over_implied + under_implied
        probability_over = over_implied / total_overround

        handicap = first_valid(raw, ["AHCh", "AHh", "BbAHh"])
        handicap_home_odds = first_valid(raw, ["AvgCAHH", "PCAHH", "B365CAHH", "AvgAHH", "PAHH", "B365AHH", "BbAvAHH"])
        handicap_away_odds = first_valid(raw, ["AvgCAHA", "PCAHA", "B365CAHA", "AvgAHA", "PAHA", "B365AHA", "BbAvAHA"])
        handicap_home_implied = 1.0 / handicap_home_odds
        handicap_away_implied = 1.0 / handicap_away_odds
        handicap_overround = handicap_home_implied + handicap_away_implied

        output = pd.DataFrame(
            {
                "date": date,
                "home_team": raw["HomeTeam"].map(canonical_team),
                "away_team": raw["AwayTeam"].map(canonical_team),
                "score_odds_over25": over_odds,
                "score_odds_under25": under_odds,
                "score_market_prob_over25": probability_over,
                "score_market_total_overround": total_overround,
                "score_market_handicap": handicap,
                "score_market_handicap_home_prob": handicap_home_implied / handicap_overround,
                "score_market_handicap_away_prob": handicap_away_implied / handicap_overround,
            }
        )
        output["score_market_total_lambda"] = output["score_market_prob_over25"].map(poisson_total_lambda)
        output["score_market_goal_diff_proxy"] = -output["score_market_handicap"]
        output["score_market_home_xg_proxy"] = (
            output["score_market_total_lambda"] + output["score_market_goal_diff_proxy"]
        ) / 2.0
        output["score_market_away_xg_proxy"] = (
            output["score_market_total_lambda"] - output["score_market_goal_diff_proxy"]
        ) / 2.0
        output["match_key"] = (
            output["date"].dt.strftime("%Y-%m-%d")
            + "|"
            + output["home_team"].map(name_key)
            + "|"
            + output["away_team"].map(name_key)
        )
        pieces.append(output.drop(columns=["date", "home_team", "away_team"]))
    return pd.concat(pieces, ignore_index=True).drop_duplicates("match_key", keep="last")


def score_model(columns: list[str], seed: int) -> Pipeline:
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
                XGBRegressor(
                    objective="count:poisson",
                    eval_metric="poisson-nloglik",
                    n_estimators=550,
                    max_depth=2,
                    learning_rate=0.03,
                    min_child_weight=8,
                    subsample=0.90,
                    colsample_bytree=0.80,
                    reg_alpha=0.15,
                    reg_lambda=5.0,
                    random_state=seed,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def score_metrics(test: pd.DataFrame, home_xg: np.ndarray, away_xg: np.ndarray, outcome_probabilities: np.ndarray) -> dict[str, float]:
    outcome_names = np.asarray(["away_win", "draw", "home_win"])[outcome_probabilities.argmax(axis=1)]
    predicted = [
        best_score_for_outcome(float(home), float(away), str(outcome))[:2]
        for home, away, outcome in zip(home_xg, away_xg, outcome_names)
    ]
    predicted_home = np.asarray([value[0] for value in predicted])
    predicted_away = np.asarray([value[1] for value in predicted])
    actual_home = test["home_goals"].to_numpy(dtype=int)
    actual_away = test["away_goals"].to_numpy(dtype=int)
    return {
        "exact_accuracy": float(np.mean((actual_home == predicted_home) & (actual_away == predicted_away))),
        "home_mae": float(mean_absolute_error(actual_home, home_xg)),
        "away_mae": float(mean_absolute_error(actual_away, away_xg)),
        "total_mae": float(mean_absolute_error(actual_home + actual_away, home_xg + away_xg)),
    }


def main() -> None:
    frame = pd.read_csv(ROOT / "outputs" / "latest" / "historical_features.csv")
    frame = frame.merge(load_score_markets(), on="match_key", how="left")
    compact, _ = compact_base(frame)
    total_columns = [column for column in frame.columns if column.startswith("score_market_") and "handicap" not in column and "goal_diff" not in column and "_xg_" not in column]
    handicap_columns = [column for column in frame.columns if column.startswith("score_market_handicap") or "goal_diff_proxy" in column]
    proxy_columns = [column for column in frame.columns if column.endswith("_xg_proxy")]
    profiles = {
        "base": compact,
        "base_1x2": compact + ["market_prob_home", "market_prob_draw", "market_prob_away"],
        "base_totals": compact + total_columns,
        "base_totals_handicap": compact + total_columns + handicap_columns,
        "base_all_score_markets": compact + total_columns + handicap_columns + proxy_columns,
    }
    rows: list[dict[str, object]] = []
    for test_season in range(2021, 2026):
        train = frame[frame["season_start"].between(2012, test_season - 3)]
        validation = frame[frame["season_start"].isin([test_season - 2, test_season - 1])]
        final_train = frame[frame["season_start"].between(2012, test_season - 1)]
        test = frame[frame["season_start"].eq(test_season)]

        classifier = estimator(compact, "shallow")
        classifier.fit(train[compact], train["target"])
        odds_weight, draw_multiplier, _, _ = tune(
            validation, classifier.predict_proba(validation[compact])
        )
        final_classifier = estimator(compact, "shallow")
        final_classifier.fit(final_train[compact], final_train["target"])
        outcome_probabilities = blend(
            final_classifier.predict_proba(test[compact]), test, odds_weight, draw_multiplier
        )

        for profile, columns in profiles.items():
            home_model = score_model(columns, 43)
            away_model = score_model(columns, 44)
            home_model.fit(final_train[columns], final_train["home_goals"])
            away_model.fit(final_train[columns], final_train["away_goals"])
            home_xg = np.clip(home_model.predict(test[columns]), 0.05, 6.5)
            away_xg = np.clip(away_model.predict(test[columns]), 0.05, 6.5)
            row = {
                "test_season": test_season,
                "profile": profile,
                "rows": len(test),
                "score_market_coverage": float(test["score_market_prob_over25"].notna().mean()),
                **score_metrics(test, home_xg, away_xg, outcome_probabilities),
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
                    "exact_accuracy": np.average(group["exact_accuracy"], weights=group["rows"]),
                    "home_mae": np.average(group["home_mae"], weights=group["rows"]),
                    "away_mae": np.average(group["away_mae"], weights=group["rows"]),
                    "total_mae": np.average(group["total_mae"], weights=group["rows"]),
                    "seasons_best_exact": int(
                        sum(
                            row.exact_accuracy
                            >= output[output["test_season"].eq(row.test_season)]["exact_accuracy"].max()
                            for row in group.itertuples(index=False)
                        )
                    ),
                }
            ),
            include_groups=False,
        )
        .sort_values(["exact_accuracy", "total_mae"], ascending=[False, True])
    )
    print("\nScore-market walk-forward summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
