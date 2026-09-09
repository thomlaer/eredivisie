from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


OUTCOME_TO_ID = {"away_win": 0, "draw": 1, "home_win": 2}
ID_TO_OUTCOME = {value: key for key, value in OUTCOME_TO_ID.items()}
ODDS_FEATURES = [
    "has_odds",
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_overround",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
]
SCORE_MARKET_FEATURES = [
    "score_odds_over25",
    "score_odds_under25",
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


def outcome_name(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


@dataclass
class TeamState:
    elo: float = 1500.0
    last_date: pd.Timestamp | None = None
    last_season: int | None = None
    matches_total: int = 0
    recent: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=10))
    home_recent: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=10))
    away_recent: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=10))
    season_events: list[dict[str, float]] = field(default_factory=list)


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if not len(array) or np.isnan(array).all():
        return np.nan
    return float(np.nanmean(array))


def _summary(events: Iterable[dict[str, float]], limit: int | None = None) -> dict[str, float]:
    rows = list(events)
    if limit is not None:
        rows = rows[-limit:]
    if not rows:
        return {
            "matches": 0.0,
            "ppg": np.nan,
            "win_rate": np.nan,
            "draw_rate": np.nan,
            "loss_rate": np.nan,
            "goals_for": np.nan,
            "goals_against": np.nan,
            "goal_diff": np.nan,
            "xg_for": np.nan,
            "xg_against": np.nan,
            "shots_for": np.nan,
            "shots_against": np.nan,
            "shots_target_for": np.nan,
            "shots_target_against": np.nan,
            "corners_for": np.nan,
            "corners_against": np.nan,
            "yellow": np.nan,
            "red": np.nan,
        }
    keys = (
        "points",
        "win",
        "draw",
        "loss",
        "goals_for",
        "goals_against",
        "goal_diff",
        "xg_for",
        "xg_against",
        "shots_for",
        "shots_against",
        "shots_target_for",
        "shots_target_against",
        "corners_for",
        "corners_against",
        "yellow",
        "red",
    )
    means = {key: _safe_mean(row.get(key, np.nan) for row in rows) for key in keys}
    return {
        "matches": float(len(rows)),
        "ppg": means["points"],
        "win_rate": means["win"],
        "draw_rate": means["draw"],
        "loss_rate": means["loss"],
        "goals_for": means["goals_for"],
        "goals_against": means["goals_against"],
        "goal_diff": means["goal_diff"],
        "xg_for": means["xg_for"],
        "xg_against": means["xg_against"],
        "shots_for": means["shots_for"],
        "shots_against": means["shots_against"],
        "shots_target_for": means["shots_target_for"],
        "shots_target_against": means["shots_target_against"],
        "corners_for": means["corners_for"],
        "corners_against": means["corners_against"],
        "yellow": means["yellow"],
        "red": means["red"],
    }


def previous_season_lookup(matches: pd.DataFrame) -> dict[tuple[int, str], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for match in matches.itertuples(index=False):
        home_goals = int(match.home_goals)
        away_goals = int(match.away_goals)
        home_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
        away_points = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0
        rows.extend(
            [
                {
                    "season_start": int(match.season_start),
                    "team": match.home_team,
                    "points": home_points,
                    "gf": home_goals,
                    "ga": away_goals,
                },
                {
                    "season_start": int(match.season_start),
                    "team": match.away_team,
                    "points": away_points,
                    "gf": away_goals,
                    "ga": home_goals,
                },
            ]
        )
    team_matches = pd.DataFrame(rows)
    grouped = (
        team_matches.groupby(["season_start", "team"], as_index=False)
        .agg(matches=("points", "size"), points=("points", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    )
    grouped["ppg"] = grouped["points"] / grouped["matches"]
    grouped["gd_per_match"] = (grouped["gf"] - grouped["ga"]) / grouped["matches"]
    grouped["rank"] = grouped.groupby("season_start")["points"].rank(method="first", ascending=False)
    team_count = grouped.groupby("season_start")["team"].transform("size")
    grouped["rank_pct"] = 1.0 - (grouped["rank"] - 1.0) / (team_count - 1.0)
    return {
        (int(row.season_start) + 1, str(row.team)): {
            "prev_ppg": float(row.ppg),
            "prev_gd_per_match": float(row.gd_per_match),
            "prev_rank_pct": float(row.rank_pct),
            "prev_matches": float(row.matches),
        }
        for row in grouped.itertuples(index=False)
    }


class FeatureBuilder:
    def __init__(self, previous_lookup: dict[tuple[int, str], dict[str, float]]) -> None:
        self.states: dict[str, TeamState] = defaultdict(TeamState)
        self.h2h: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=10))
        self.previous_lookup = previous_lookup

    @staticmethod
    def _prepare_for_season(state: TeamState, season_start: int) -> None:
        if state.last_season is None:
            state.last_season = season_start
            return
        if state.last_season != season_start:
            state.elo = 1500.0 + 0.85 * (state.elo - 1500.0)
            state.season_events = []
            state.last_season = season_start

    @staticmethod
    def _prefix(target: dict[str, Any], prefix: str, values: dict[str, float]) -> None:
        for key, value in values.items():
            target[f"{prefix}_{key}"] = value

    def _team_features(
        self,
        team: str,
        date: pd.Timestamp,
        season_start: int,
        venue: str,
    ) -> dict[str, float]:
        state = self.states[team]
        self._prepare_for_season(state, season_start)
        rest_days = 21.0
        if state.last_date is not None:
            rest_days = float(np.clip((date - state.last_date).days, 1, 60))
        venue_events = state.home_recent if venue == "home" else state.away_recent
        previous = self.previous_lookup.get((season_start, team), {})
        output: dict[str, float] = {
            "elo": float(state.elo),
            "rest_days": rest_days,
            "experience_matches": float(state.matches_total),
            "prev_ppg": float(previous.get("prev_ppg", np.nan)),
            "prev_gd_per_match": float(previous.get("prev_gd_per_match", np.nan)),
            "prev_rank_pct": float(previous.get("prev_rank_pct", np.nan)),
            "promoted_or_new": float(not bool(previous)),
        }
        self._prefix(output, "form5", _summary(state.recent, 5))
        self._prefix(output, "form10", _summary(state.recent, 10))
        self._prefix(output, "venue5", _summary(venue_events, 5))
        self._prefix(output, "season", _summary(state.season_events))
        return output

    def _h2h_features(self, home_team: str, away_team: str) -> dict[str, float]:
        pair = tuple(sorted((home_team, away_team)))
        events = list(self.h2h[pair])[-5:]
        if not events:
            return {"h2h_matches": 0.0, "h2h_home_ppg": np.nan, "h2h_home_gd": np.nan}
        points: list[float] = []
        goal_diffs: list[float] = []
        for event in events:
            if event["home_team"] == home_team:
                gf, ga = event["home_goals"], event["away_goals"]
            else:
                gf, ga = event["away_goals"], event["home_goals"]
            points.append(3.0 if gf > ga else 1.0 if gf == ga else 0.0)
            goal_diffs.append(float(gf - ga))
        return {
            "h2h_matches": float(len(events)),
            "h2h_home_ppg": float(np.mean(points)),
            "h2h_home_gd": float(np.mean(goal_diffs)),
        }

    def feature_row(self, row: Any) -> dict[str, Any]:
        date = pd.Timestamp(row.date)
        season_start = int(row.season_start)
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home = self._team_features(home_team, date, season_start, "home")
        away = self._team_features(away_team, date, season_start, "away")
        home_elo = home["elo"]
        away_elo = away["elo"]
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + 65.0)) / 400.0))
        output: dict[str, Any] = {
            "date": date,
            "season_start": season_start,
            "season": getattr(row, "season", f"{season_start}/{str(season_start + 1)[-2:]}"),
            "home_team": home_team,
            "away_team": away_team,
            "month_sin": math.sin(2.0 * math.pi * date.month / 12.0),
            "month_cos": math.cos(2.0 * math.pi * date.month / 12.0),
            "elo_expected_home": expected_home,
            "elo_diff": home_elo - away_elo,
        }
        self._prefix(output, "home", home)
        self._prefix(output, "away", away)
        for key in sorted(set(home) & set(away)):
            if key == "elo":
                continue
            output[f"diff_{key}"] = home[key] - away[key]
        output.update(self._h2h_features(home_team, away_team))
        for column in ODDS_FEATURES + SCORE_MARKET_FEATURES:
            output[column] = getattr(row, column, np.nan)
        output["odds_source"] = getattr(row, "odds_source", "")
        output["round"] = getattr(row, "round", np.nan)
        output["match_number"] = getattr(row, "match_number", np.nan)
        output["kickoff_utc"] = getattr(row, "kickoff_utc", pd.NaT)
        output["venue"] = getattr(row, "venue", "")
        return output

    def update(self, row: Any) -> None:
        date = pd.Timestamp(row.date)
        season_start = int(row.season_start)
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home_goals = int(row.home_goals)
        away_goals = int(row.away_goals)
        home_state = self.states[home_team]
        away_state = self.states[away_team]
        self._prepare_for_season(home_state, season_start)
        self._prepare_for_season(away_state, season_start)

        home_points = 3.0 if home_goals > away_goals else 1.0 if home_goals == away_goals else 0.0
        away_points = 3.0 if away_goals > home_goals else 1.0 if home_goals == away_goals else 0.0
        home_event = {
            "points": home_points,
            "win": float(home_points == 3.0),
            "draw": float(home_points == 1.0),
            "loss": float(home_points == 0.0),
            "goals_for": float(home_goals),
            "goals_against": float(away_goals),
            "goal_diff": float(home_goals - away_goals),
            "xg_for": float(getattr(row, "home_xg", np.nan)),
            "xg_against": float(getattr(row, "away_xg", np.nan)),
            "shots_for": float(getattr(row, "home_shots", np.nan)),
            "shots_against": float(getattr(row, "away_shots", np.nan)),
            "shots_target_for": float(getattr(row, "home_shots_target", np.nan)),
            "shots_target_against": float(getattr(row, "away_shots_target", np.nan)),
            "corners_for": float(getattr(row, "home_corners", np.nan)),
            "corners_against": float(getattr(row, "away_corners", np.nan)),
            "yellow": float(getattr(row, "home_yellow", np.nan)),
            "red": float(getattr(row, "home_red", np.nan)),
        }
        away_event = {
            "points": away_points,
            "win": float(away_points == 3.0),
            "draw": float(away_points == 1.0),
            "loss": float(away_points == 0.0),
            "goals_for": float(away_goals),
            "goals_against": float(home_goals),
            "goal_diff": float(away_goals - home_goals),
            "xg_for": float(getattr(row, "away_xg", np.nan)),
            "xg_against": float(getattr(row, "home_xg", np.nan)),
            "shots_for": float(getattr(row, "away_shots", np.nan)),
            "shots_against": float(getattr(row, "home_shots", np.nan)),
            "shots_target_for": float(getattr(row, "away_shots_target", np.nan)),
            "shots_target_against": float(getattr(row, "home_shots_target", np.nan)),
            "corners_for": float(getattr(row, "away_corners", np.nan)),
            "corners_against": float(getattr(row, "home_corners", np.nan)),
            "yellow": float(getattr(row, "away_yellow", np.nan)),
            "red": float(getattr(row, "away_red", np.nan)),
        }
        for state, event, venue in (
            (home_state, home_event, "home"),
            (away_state, away_event, "away"),
        ):
            state.recent.append(event)
            (state.home_recent if venue == "home" else state.away_recent).append(event)
            state.season_events.append(event)
            state.last_date = date
            state.last_season = season_start
            state.matches_total += 1

        expected_home = 1.0 / (1.0 + 10.0 ** ((away_state.elo - (home_state.elo + 65.0)) / 400.0))
        actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        goal_margin = abs(home_goals - away_goals)
        margin_factor = 1.0 if goal_margin <= 1 else min(2.0, 1.0 + 0.25 * (goal_margin - 1))
        change = 24.0 * margin_factor * (actual_home - expected_home)
        home_state.elo += change
        away_state.elo -= change

        pair = tuple(sorted((home_team, away_team)))
        self.h2h[pair].append(
            {
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )


def build_historical_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, FeatureBuilder]:
    ordered = matches.sort_values(["date", "season_start", "home_team", "away_team"]).reset_index(drop=True)
    builder = FeatureBuilder(previous_season_lookup(ordered))
    rows: list[dict[str, Any]] = []
    for match in ordered.itertuples(index=False):
        features = builder.feature_row(match)
        home_goals = int(match.home_goals)
        away_goals = int(match.away_goals)
        outcome = outcome_name(home_goals, away_goals)
        features.update(
            {
                "home_goals": home_goals,
                "away_goals": away_goals,
                "actual_outcome": outcome,
                "target": OUTCOME_TO_ID[outcome],
                "match_key": match.match_key,
            }
        )
        rows.append(features)
        builder.update(match)
    return pd.DataFrame(rows), builder


def build_future_features(fixtures: pd.DataFrame, builder: FeatureBuilder) -> pd.DataFrame:
    rows = [builder.feature_row(row) | {"match_key": row.match_key} for row in fixtures.itertuples(index=False)]
    return pd.DataFrame(rows)
