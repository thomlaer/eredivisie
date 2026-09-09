from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from experiment_transfermarkt_features import compact_base, context_columns, estimator, tune, blend

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prediction.names import canonical_team  # noqa: E402


OUTPUT = ROOT / "outputs" / "experiments" / "clubelo_walk_forward.csv"


def add_clubelo(frame: pd.DataFrame) -> pd.DataFrame:
    ratings = pd.read_csv(ROOT / "data" / "external" / "clubelo_snapshots.csv", parse_dates=["date"])
    ratings = ratings[ratings["country"].eq("NED")].copy()
    ratings["team"] = ratings["club"].map(canonical_team)
    history: dict[str, tuple[list[int], list[float]]] = {}
    for team, group in ratings.groupby("team"):
        ordered = group.sort_values("date")
        history[str(team)] = (
            ordered["date"].values.astype("datetime64[D]").astype(np.int64).tolist(),
            ordered["elo"].astype(float).tolist(),
        )

    def snapshot(team: str, date_number: int) -> tuple[float, float, float]:
        if team not in history:
            return np.nan, np.nan, np.nan
        dates, values = history[team]
        index = bisect.bisect_left(dates, date_number) - 1
        if index < 0:
            return np.nan, np.nan, np.nan
        old_index = bisect.bisect_left(dates, date_number - 30) - 1
        momentum = values[index] - values[old_index] if old_index >= 0 else np.nan
        return values[index], momentum, float(date_number - dates[index])

    rows: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        date_number = int(pd.Timestamp(row.date).to_datetime64().astype("datetime64[D]").astype(np.int64))
        home_elo, home_momentum, home_age = snapshot(str(row.home_team), date_number)
        away_elo, away_momentum, away_age = snapshot(str(row.away_team), date_number)
        expected = np.nan
        if np.isfinite(home_elo) and np.isfinite(away_elo):
            expected = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + 65.0)) / 400.0))
        rows.append(
            {
                "clubelo_home": home_elo,
                "clubelo_away": away_elo,
                "clubelo_diff": home_elo - away_elo,
                "clubelo_expected_home": expected,
                "clubelo_home_momentum30": home_momentum,
                "clubelo_away_momentum30": away_momentum,
                "clubelo_momentum30_diff": home_momentum - away_momentum,
                "clubelo_home_age_days": home_age,
                "clubelo_away_age_days": away_age,
            }
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def main() -> None:
    features = pd.read_csv(ROOT / "outputs" / "latest" / "historical_features.csv", parse_dates=["date"])
    context = pd.read_csv(ROOT / "data" / "external" / "transfermarkt_eredivisie_context.csv").drop(
        columns=["date", "home_team", "away_team"]
    )
    frame = add_clubelo(features.merge(context, on="match_key", how="left"))
    compact, _ = compact_base(frame)
    clubelo = [column for column in frame.columns if column.startswith("clubelo_")]
    profiles = {
        "base": compact,
        "clubelo": compact + clubelo,
        "clubelo_value_form": compact + clubelo + context_columns(frame, "value_form"),
    }
    rows: list[dict[str, object]] = []
    for test_season in range(2019, 2026):
        validation_seasons = [test_season - 2, test_season - 1]
        for profile, columns in profiles.items():
            train = frame[frame["season_start"].between(2012, test_season - 3)]
            validation = frame[frame["season_start"].isin(validation_seasons)]
            test = frame[frame["season_start"].eq(test_season)]
            tuning_model = estimator(columns, "shallow")
            tuning_model.fit(train[columns], train["target"])
            odds_weight, draw_multiplier, _, _ = tune(
                validation, tuning_model.predict_proba(validation[columns])
            )
            final_train = frame[frame["season_start"].between(2012, test_season - 1)]
            final_model = estimator(columns, "shallow")
            final_model.fit(final_train[columns], final_train["target"])
            base_probabilities = final_model.predict_proba(test[columns])
            probabilities = blend(base_probabilities, test, odds_weight, draw_multiplier)
            row = {
                "test_season": test_season,
                "profile": profile,
                "rows": len(test),
                "clubelo_coverage": float(test["clubelo_diff"].notna().mean()),
                "odds_weight": odds_weight,
                "draw_multiplier": draw_multiplier,
                "accuracy": float(accuracy_score(test["target"], probabilities.argmax(axis=1))),
                "log_loss": float(log_loss(test["target"], probabilities, labels=[0, 1, 2])),
                "no_odds_accuracy": float(accuracy_score(test["target"], base_probabilities.argmax(axis=1))),
                "no_odds_log_loss": float(log_loss(test["target"], base_probabilities, labels=[0, 1, 2])),
            }
            rows.append(row)
            print(json.dumps(row))
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT, index=False)
    complete = output[output["test_season"].le(2024)]
    summary = (
        complete.groupby("profile", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "rows": int(group["rows"].sum()),
                    "accuracy": np.average(group["accuracy"], weights=group["rows"]),
                    "log_loss": np.average(group["log_loss"], weights=group["rows"]),
                    "no_odds_accuracy": np.average(group["no_odds_accuracy"], weights=group["rows"]),
                    "no_odds_log_loss": np.average(group["no_odds_log_loss"], weights=group["rows"]),
                }
            ),
            include_groups=False,
        )
        .sort_values(["accuracy", "log_loss"], ascending=[False, True])
    )
    print("\nClubElo walk-forward summary (complete ratings through 2024/25)")
    print(summary.to_string(index=False))
    print("\n2025/26 is reported separately because the snapshot file stops at 2025-06-01.")
    print(output[output["test_season"].eq(2025)].to_string(index=False))


if __name__ == "__main__":
    main()
