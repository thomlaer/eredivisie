from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DASHBOARD_PUBLIC, OUTPUT_DIR, RAW_DIR


def _json_default(value: Any) -> Any:
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].astype(str)
    clean = clean.replace({np.nan: None, pd.NaT: None})
    return clean.to_dict(orient="records")


def publish_dashboard(
    upcoming: pd.DataFrame,
    played: pd.DataFrame,
    current_standings: pd.DataFrame,
    projected: pd.DataFrame,
    champions: pd.DataFrame,
    injuries: pd.DataFrame,
    metrics: dict[str, Any],
    source_report: dict[str, Any],
) -> Path:
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    dashboard = {
        "metadata": {
            "generated_at_utc": generated,
            "season": "2026/27",
            "historical_matches": int(metrics.get("data", {}).get("historical_matches", 0)),
            "latest_result": metrics.get("data", {}).get("latest_result"),
            "model_accuracy": metrics.get("selected", {}).get("accuracy"),
            "model_log_loss": metrics.get("selected", {}).get("log_loss"),
            "current_season_accuracy": metrics.get("current_season", {}).get("accuracy"),
            "odds_weight": metrics.get("selected", {}).get("odds_weight"),
            "draw_multiplier": metrics.get("selected", {}).get("draw_multiplier"),
            "espn_status": source_report.get("espn_status"),
            "injury_status": source_report.get("injury_status"),
            "suspension_status": source_report.get("suspension_status"),
            "current_squad_status": source_report.get("current_squad_status"),
            "absences_total": int(len(injuries)),
            "injuries_matched": int(
                injuries.get("matched_roster", pd.Series(dtype=float)).fillna(0).sum()
            ),
            "injuries_unmatched": int(
                len(injuries) - injuries.get("matched_roster", pd.Series(dtype=float)).fillna(0).sum()
            ),
            "likely_starters_unavailable": int(
                injuries.get("selected_proxy", pd.Series(dtype=float)).fillna(0).sum()
            ),
        },
        "upcoming": _records(upcoming),
        "played": _records(played.sort_values("date", ascending=False)),
        "current_table": _records(current_standings),
        "projected_table": _records(projected),
        "champions": _records(champions),
        "injuries": _records(
            injuries[
                [
                    column
                    for column in (
                        "team",
                        "player",
                        "injury",
                        "absence_type",
                        "since",
                        "expected_return",
                        "availability",
                        "selected_proxy",
                        "matched_roster",
                        "market_value_m",
                        "estimated_value_impact_m",
                        "source",
                    )
                    if column in injuries
                ]
            ]
        ) if not injuries.empty else [],
        "feature_importance": metrics.get("feature_importance", []),
        "benchmarks": {
            "selected": metrics.get("selected", {}),
            "xgboost_without_odds": metrics.get("xgboost_without_odds", {}),
            "market_only": metrics.get("market_only", {}),
            "current_season": metrics.get("current_season", {}),
        },
        "downloads": {
            "upcoming_csv": "/files/upcoming_predictions.csv",
            "played_csv": "/files/played_predictions.csv",
            "standings_csv": "/files/projected_table.csv",
        },
    }
    destination = DASHBOARD_PUBLIC / "data" / "dashboard.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=True, default=_json_default),
        encoding="utf-8",
    )
    files = DASHBOARD_PUBLIC / "files"
    files.mkdir(parents=True, exist_ok=True)
    for name in ("upcoming_predictions.csv", "played_predictions.csv", "projected_table.csv"):
        source = OUTPUT_DIR / name
        if source.exists():
            shutil.copy2(source, files / name)
    return destination
