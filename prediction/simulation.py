from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .config import RANDOM_SEED, SIMULATIONS


def current_table(matches: pd.DataFrame, season_start: int) -> pd.DataFrame:
    standings: dict[str, dict[str, int]] = defaultdict(
        lambda: {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "points": 0}
    )
    season = matches[matches["season_start"] == season_start]
    for row in season.itertuples(index=False):
        home = standings[row.home_team]
        away = standings[row.away_team]
        home_goals = int(row.home_goals)
        away_goals = int(row.away_goals)
        home["played"] += 1
        away["played"] += 1
        home["gf"] += home_goals
        home["ga"] += away_goals
        away["gf"] += away_goals
        away["ga"] += home_goals
        if home_goals > away_goals:
            home["wins"] += 1
            home["points"] += 3
            away["losses"] += 1
        elif home_goals < away_goals:
            away["wins"] += 1
            away["points"] += 3
            home["losses"] += 1
        else:
            home["draws"] += 1
            away["draws"] += 1
            home["points"] += 1
            away["points"] += 1
    rows = []
    for team, values in standings.items():
        rows.append({"team": team, **values, "gd": values["gf"] - values["ga"]})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(["points", "gd", "gf", "team"], ascending=[False, False, False, True]).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def _forced_scores(
    rng: np.random.Generator,
    outcomes: np.ndarray,
    home_xg: float,
    away_xg: float,
) -> tuple[np.ndarray, np.ndarray]:
    home = rng.poisson(max(0.05, float(home_xg)), size=len(outcomes)).astype(int)
    away = rng.poisson(max(0.05, float(away_xg)), size=len(outcomes)).astype(int)
    home_win = outcomes == 2
    draw = outcomes == 1
    away_win = outcomes == 0
    home[home_win & (home <= away)] = away[home_win & (home <= away)] + 1
    away[away_win & (away <= home)] = home[away_win & (away <= home)] + 1
    draw_goals = np.rint((home[draw] + away[draw]) / 2.0).astype(int)
    home[draw] = draw_goals
    away[draw] = draw_goals
    return home, away


def simulate_season(
    matches: pd.DataFrame,
    future_predictions: pd.DataFrame,
    season_start: int,
    simulations: int = SIMULATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = current_table(matches, season_start)
    teams = sorted(set(actual.get("team", pd.Series(dtype=str))) | set(future_predictions["home_team"]) | set(future_predictions["away_team"]))
    team_index = {team: index for index, team in enumerate(teams)}
    team_count = len(teams)
    points = np.zeros((simulations, team_count), dtype=np.int16)
    goals_for = np.zeros((simulations, team_count), dtype=np.int16)
    goals_against = np.zeros((simulations, team_count), dtype=np.int16)

    for row in actual.itertuples(index=False):
        index = team_index[row.team]
        points[:, index] = int(row.points)
        goals_for[:, index] = int(row.gf)
        goals_against[:, index] = int(row.ga)

    rng = np.random.default_rng(RANDOM_SEED)
    for row in future_predictions.sort_values(["date", "match_number"]).itertuples(index=False):
        probabilities = np.asarray([row.prob_away_win, row.prob_draw, row.prob_home_win], dtype=float)
        probabilities = np.clip(probabilities, 0.0, None)
        probabilities /= probabilities.sum()
        outcomes = rng.choice(3, size=simulations, p=probabilities)
        home_goals, away_goals = _forced_scores(
            rng,
            outcomes,
            float(row.expected_home_goals),
            float(row.expected_away_goals),
        )
        home_index = team_index[row.home_team]
        away_index = team_index[row.away_team]
        goals_for[:, home_index] += home_goals
        goals_against[:, home_index] += away_goals
        goals_for[:, away_index] += away_goals
        goals_against[:, away_index] += home_goals
        points[:, home_index] += np.where(outcomes == 2, 3, np.where(outcomes == 1, 1, 0)).astype(np.int16)
        points[:, away_index] += np.where(outcomes == 0, 3, np.where(outcomes == 1, 1, 0)).astype(np.int16)

    goal_difference = goals_for - goals_against
    position_counts = np.zeros((team_count, team_count), dtype=np.int32)
    for simulation in range(simulations):
        jitter = rng.uniform(0.0, 0.001, size=team_count)
        order = np.lexsort((jitter, -goals_for[simulation], -goal_difference[simulation], -points[simulation]))
        for position, index in enumerate(order):
            position_counts[index, position] += 1

    rows: list[dict[str, Any]] = []
    for team, index in team_index.items():
        probabilities = position_counts[index] / simulations
        rows.append(
            {
                "team": team,
                "expected_points": float(points[:, index].mean()),
                "expected_gd": float(goal_difference[:, index].mean()),
                "expected_rank": float(np.dot(probabilities, np.arange(1, team_count + 1))),
                "champion_prob": float(probabilities[0]),
                "top2_prob": float(probabilities[:2].sum()),
                "top3_prob": float(probabilities[:3].sum()),
                "top5_prob": float(probabilities[:5].sum()),
                "bottom3_prob": float(probabilities[-3:].sum()),
                "relegation_prob": float(probabilities[-2:].sum()),
            }
        )
    projected = pd.DataFrame(rows).sort_values(
        ["expected_rank", "expected_points", "expected_gd"], ascending=[True, False, False]
    ).reset_index(drop=True)
    projected.insert(0, "projected_rank", np.arange(1, len(projected) + 1))
    champions = projected.sort_values("champion_prob", ascending=False).reset_index(drop=True)
    champions.insert(0, "champion_rank", np.arange(1, len(champions) + 1))
    return projected, champions
