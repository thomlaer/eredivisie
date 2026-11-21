from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from prediction.config import (
    CURRENT_SEASON,
    MODEL_DIR,
    MODEL_TRAIN_FROM,
    OUTPUT_DIR,
    PREDICTION_HISTORY,
    TEST_SEASON,
    VALIDATION_SEASONS,
    ensure_directories,
)
from prediction.dashboard import publish_dashboard
from prediction.data import build_processed_data, update_sources, validate_espn_results
from prediction.external import attach_transfermarkt_context
from prediction.features import build_future_features, build_historical_features
from prediction.injuries import apply_current_availability, refresh_injuries, save_injury_report
from prediction.modeling import (
    evaluate_and_tune,
    fit_bundle,
    prediction_frame,
    predict_probabilities,
    save_bundle,
    save_metrics,
)
from prediction.simulation import current_table, simulate_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the Eredivisie prediction model and local dashboard data.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--force-live",
        action="store_true",
        help="Refresh current squads/player performance even when the local cache is fresh.",
    )
    parser.add_argument("--simulations", type=int, default=10_000)
    return parser.parse_args()


def attach_actuals(predictions: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    columns = ["match_key", "home_goals", "away_goals", "actual_outcome"]
    output = predictions.merge(actual[columns], on="match_key", how="left")
    output["actual_score"] = output["home_goals"].astype("Int64").astype(str) + "-" + output["away_goals"].astype("Int64").astype(str)
    output["winner_correct"] = output["predicted_outcome"] == output["actual_outcome"]
    return output


def attach_fixture_metadata(predictions: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = ["match_number", "round", "kickoff_utc", "venue"]
    available = [column for column in metadata_columns if column in fixtures]
    if not available:
        return predictions.copy()

    fixture_metadata = fixtures[["match_key", *available]].drop_duplicates("match_key", keep="last")
    output = predictions.merge(fixture_metadata, on="match_key", how="left", suffixes=("", "_fixture"))
    for column in available:
        fixture_column = f"{column}_fixture"
        if fixture_column not in output:
            continue
        if column in output:
            output[column] = output[column].where(output[column].notna(), output[fixture_column])
        else:
            output[column] = output[fixture_column]
        output = output.drop(columns=fixture_column)
    return output


def current_season_metrics(actual_features: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float | int]:
    probabilities = predictions[["prob_away_win", "prob_draw", "prob_home_win"]].to_numpy(dtype=float)
    y_true = actual_features["target"].to_numpy(dtype=int)
    return {
        "rows": int(len(actual_features)),
        "accuracy": float(accuracy_score(y_true, probabilities.argmax(axis=1))),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
    }


def played_prediction_metrics(played: pd.DataFrame) -> dict[str, float | int | None]:
    if played.empty:
        return {"rows": 0, "accuracy": None, "log_loss": None}
    outcome_ids = {"away_win": 0, "draw": 1, "home_win": 2}
    probabilities = played[["prob_away_win", "prob_draw", "prob_home_win"]].apply(
        pd.to_numeric, errors="coerce"
    )
    targets = played["actual_outcome"].map(outcome_ids)
    valid = probabilities.notna().all(axis=1) & targets.notna()
    if not valid.any():
        return {"rows": 0, "accuracy": None, "log_loss": None}
    values = probabilities.loc[valid].to_numpy(dtype=float)
    y_true = targets.loc[valid].to_numpy(dtype=int)
    return {
        "rows": int(valid.sum()),
        "accuracy": float(accuracy_score(y_true, values.argmax(axis=1))),
        "log_loss": float(log_loss(y_true, values, labels=[0, 1, 2])),
    }


def snapshot_predictions(
    upcoming: pd.DataFrame,
    path=PREDICTION_HISTORY,
    generated_at_utc: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    generated = pd.Timestamp.now(tz="UTC") if generated_at_utc is None else pd.Timestamp(generated_at_utc)
    generated = generated.tz_localize("UTC") if generated.tzinfo is None else generated.tz_convert("UTC")
    if path.exists():
        try:
            history = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            history = pd.DataFrame()
    else:
        history = pd.DataFrame()

    candidates = upcoming.copy()
    kickoff = pd.to_datetime(candidates.get("kickoff_utc"), utc=True, errors="coerce")
    fallback_date = pd.to_datetime(candidates.get("date"), utc=True, errors="coerce") + pd.Timedelta(days=1)
    candidates = candidates[(kickoff.fillna(fallback_date) > generated)].copy()
    if candidates.empty:
        return history
    candidates["snapshot_at_utc"] = generated.isoformat()
    if not history.empty and "match_key" in history:
        history = history[~history["match_key"].isin(candidates["match_key"])]
    frames = [frame.dropna(axis=1, how="all") for frame in (history, candidates) if not frame.empty]
    output = pd.concat(frames, ignore_index=True, sort=False)
    output = output.drop_duplicates("match_key", keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)
    return output


def outcome_export(frame: pd.DataFrame) -> pd.DataFrame:
    internal_columns = [
        "expected_home_goals",
        "expected_away_goals",
        "predicted_home_score",
        "predicted_away_score",
        "predicted_score",
        "predicted_score_probability",
    ]
    return frame.drop(columns=[column for column in internal_columns if column in frame], errors="ignore")


def main() -> None:
    args = parse_args()
    ensure_directories()
    if args.skip_download:
        report_path = OUTPUT_DIR.parent.parent / "data" / "raw" / "source_update_report.json"
        source_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    else:
        source_report = update_sources(force=args.force_download)
        source_report.update(
            refresh_injuries(force_squads=args.force_download or args.force_live)
        )
        report_path = OUTPUT_DIR.parent.parent / "data" / "raw" / "source_update_report.json"
        report_path.write_text(json.dumps(source_report, indent=2, ensure_ascii=True), encoding="utf-8")

    matches, fixtures = build_processed_data()
    source_report.update(validate_espn_results(matches))
    report_path.write_text(json.dumps(source_report, indent=2, ensure_ascii=True), encoding="utf-8")
    features, state_builder = build_historical_features(matches)
    features = attach_transfermarkt_context(features)
    features = features[features["season_start"] >= MODEL_TRAIN_FROM].reset_index(drop=True)

    metrics, draw_multiplier, draw_multiplier_no_odds, odds_weight = evaluate_and_tune(
        features, VALIDATION_SEASONS, TEST_SEASON
    )
    metrics["data"] = {
        "historical_matches": int(len(matches)),
        "model_rows": int(len(features)),
        "latest_result": str(matches["date"].max().date()),
        "historical_odds_rows": int(matches["has_odds"].sum()),
        "future_fixtures": int((~fixtures["is_played"]).sum()),
        "transfermarkt_context_rows": int(features.get("tm_match_available", pd.Series(dtype=float)).notna().sum()),
    }

    preseason_train = features[features["season_start"] < CURRENT_SEASON].copy()
    current_features = features[features["season_start"] == CURRENT_SEASON].copy()
    preseason_bundle = fit_bundle(
        preseason_train,
        draw_multiplier=draw_multiplier,
        draw_multiplier_no_odds=draw_multiplier_no_odds,
        odds_weight=odds_weight,
    )
    current_predictions = prediction_frame(preseason_bundle, current_features)
    played = attach_actuals(current_predictions, current_features)
    played = attach_fixture_metadata(played, fixtures)
    metrics["current_season_reconstructed"] = current_season_metrics(
        current_features, current_predictions
    )

    final_bundle = fit_bundle(
        features,
        draw_multiplier=draw_multiplier,
        draw_multiplier_no_odds=draw_multiplier_no_odds,
        odds_weight=odds_weight,
    )
    upcoming_fixtures = fixtures[~fixtures["is_played"]].copy()
    future_features = build_future_features(upcoming_fixtures, state_builder)
    future_features = attach_transfermarkt_context(future_features)
    future_features, injuries = apply_current_availability(future_features)
    save_injury_report(injuries)
    metrics["data"]["current_absences"] = int(len(injuries))
    metrics["data"]["current_injuries_matched"] = int(
        injuries.get("matched_roster", pd.Series(dtype=float)).fillna(0).sum()
    )
    metrics["data"]["current_injuries_unmatched"] = int(
        len(injuries) - metrics["data"]["current_injuries_matched"]
    )
    metrics["data"]["likely_starters_unavailable"] = int(
        injuries.get("selected_proxy", pd.Series(dtype=float)).fillna(0).sum()
    )
    upcoming = prediction_frame(final_bundle, future_features)

    standings = current_table(matches, CURRENT_SEASON)
    projected, champions = simulate_season(matches, upcoming, CURRENT_SEASON, simulations=args.simulations)
    public_upcoming = outcome_export(upcoming)
    history = snapshot_predictions(public_upcoming)
    archived = (
        history[history["match_key"].isin(current_features["match_key"])].copy()
        if "match_key" in history
        else pd.DataFrame()
    )
    archived_played = attach_actuals(archived, current_features) if not archived.empty else archived
    archived_played = attach_fixture_metadata(archived_played, fixtures)
    if not archived_played.empty:
        archived_played["forecast_status"] = "archived"

    reconstructed_played = outcome_export(played)
    if not reconstructed_played.empty:
        reconstructed_played = reconstructed_played[
            ~reconstructed_played["match_key"].isin(archived_played.get("match_key", []))
        ].copy()
        reconstructed_played["forecast_status"] = "reconstructed"
    public_played = pd.concat(
        [archived_played, reconstructed_played], ignore_index=True, sort=False
    )
    if "date" in public_played:
        public_played["date"] = pd.to_datetime(public_played["date"], errors="coerce")
    metrics["current_season"] = played_prediction_metrics(archived_played)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    public_upcoming.to_csv(OUTPUT_DIR / "upcoming_predictions.csv", index=False)
    public_played.to_csv(OUTPUT_DIR / "played_predictions.csv", index=False)
    standings.to_csv(OUTPUT_DIR / "current_table.csv", index=False)
    projected.to_csv(OUTPUT_DIR / "projected_table.csv", index=False)
    champions.to_csv(OUTPUT_DIR / "champion_probabilities.csv", index=False)
    features.to_csv(OUTPUT_DIR / "historical_features.csv", index=False)
    save_bundle(final_bundle, MODEL_DIR / "eredivisie_xgboost.joblib")
    save_metrics(metrics, OUTPUT_DIR / "model_metrics.json")
    publish_dashboard(public_upcoming, public_played, standings, projected, champions, injuries, metrics, source_report)

    print(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "matches": len(matches),
                "upcoming": len(upcoming),
                "test_accuracy": metrics["selected"]["accuracy"],
                "current_accuracy": metrics["current_season"]["accuracy"],
                "odds_weight": odds_weight,
                "draw_multiplier": draw_multiplier,
                "draw_multiplier_no_odds": draw_multiplier_no_odds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
