from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    r"C:\Users\thoml\OneDrive\Documenten\voetbal_prediction\data\kagglehub\datasets"
    r"\davidcariboo\player-scores\versions\671"
)
OUTPUT = ROOT / "data" / "external" / "transfermarkt_eredivisie_context.csv"
REPORT = ROOT / "data" / "external" / "transfermarkt_eredivisie_coverage.json"
CURRENT_SQUADS = ROOT / "data" / "external" / "transfermarkt_current_squads.csv"

CLUB_NAMES = {
    132: "NAC Breda",
    133: "SC Cambuur",
    192: "Roda JC",
    200: "FC Utrecht",
    202: "FC Groningen",
    234: "Feyenoord",
    235: "Waalwijk",
    306: "Heerenveen",
    317: "FC Twente",
    383: "PSV",
    385: "Fortuna Sittard",
    403: "Willem II",
    467: "NEC",
    468: "Sparta Rotterdam",
    499: "Vitesse",
    610: "Ajax",
    642: "Graafschap",
    723: "Almere City",
    724: "Volendam",
    798: "Excelsior",
    1090: "AZ",
    1268: "ADO Den Haag",
    1269: "PEC Zwolle",
    1283: "FC Emmen",
    1304: "Heracles",
    1426: "VVV Venlo",
    1434: "Telstar",
    1435: "Go Ahead Eagles",
    1455: "Dordrecht",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract leakage-safe Eredivisie Transfermarkt features.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args()


def key_text(value: str) -> str:
    return " ".join("".join(char.lower() if char.isalnum() else " " for char in value).split())


def match_key(date: pd.Timestamp, home: str, away: str) -> str:
    return f"{date:%Y-%m-%d}|{key_text(home)}|{key_text(away)}"


def load_starters(source: Path, game_ids: set[int]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    columns = ["date", "game_id", "player_id", "club_id", "type", "position"]
    for chunk in pd.read_csv(source / "game_lineups.csv", usecols=columns, chunksize=500_000):
        selected = chunk[chunk["game_id"].isin(game_ids) & chunk["type"].eq("starting_lineup")].copy()
        if not selected.empty:
            pieces.append(selected.drop(columns="type"))
    return pd.concat(pieces, ignore_index=True)


def load_player_rows(source: Path, player_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = pd.read_csv(
        source / "players.csv",
        usecols=[
            "player_id",
            "name",
            "date_of_birth",
            "last_season",
            "current_club_id",
            "position",
            "sub_position",
            "market_value_in_eur",
        ],
        parse_dates=["date_of_birth"],
    )
    current_ids = set(
        players.loc[
            players["last_season"].ge(2025) & players["current_club_id"].isin(CLUB_NAMES),
            "player_id",
        ].astype(int)
    )
    relevant_ids = player_ids | current_ids

    valuations = pd.read_csv(
        source / "player_valuations.csv",
        usecols=["player_id", "date", "market_value_in_eur"],
        parse_dates=["date"],
    )
    valuations = valuations[valuations["player_id"].isin(relevant_ids)].sort_values(["player_id", "date"])

    pieces: list[pd.DataFrame] = []
    columns = [
        "player_id",
        "date",
        "minutes_played",
        "goals",
        "assists",
        "yellow_cards",
        "red_cards",
    ]
    for chunk in pd.read_csv(source / "appearances.csv", usecols=columns, parse_dates=["date"], chunksize=500_000):
        selected = chunk[chunk["player_id"].isin(relevant_ids)]
        if not selected.empty:
            pieces.append(selected)
    appearances = pd.concat(pieces, ignore_index=True).sort_values(["player_id", "date"])
    return players, valuations, appearances


def current_squad_proxies(players: pd.DataFrame) -> dict[int, dict[int, str]]:
    active = players[
        players["last_season"].ge(2025)
        & players["current_club_id"].isin(CLUB_NAMES)
        & players["market_value_in_eur"].notna()
    ].copy()
    active["position_group"] = active["position"].map(position_group)
    output: dict[int, dict[int, str]] = {}
    quotas = {"gk": 1, "def": 4, "mid": 3, "att": 3}
    for club_id, group in active.groupby("current_club_id"):
        ordered = group.sort_values("market_value_in_eur", ascending=False)
        selected: list[int] = []
        for group_name, count in quotas.items():
            selected.extend(ordered[ordered["position_group"].eq(group_name)].head(count).index.tolist())
        if len(selected) < 11:
            selected.extend([index for index in ordered.index if index not in selected][: 11 - len(selected)])
        lineup: dict[int, str] = {}
        for index, row in ordered.loc[selected[:11]].iterrows():
            position = row["sub_position"] if pd.notna(row["sub_position"]) else row["position"]
            lineup[int(row["player_id"])] = str(position)
        if lineup:
            output[int(club_id)] = lineup
    return output


def timelines(frame: pd.DataFrame, value_columns: list[str]) -> dict[int, tuple[list[int], np.ndarray]]:
    output: dict[int, tuple[list[int], np.ndarray]] = {}
    for player_id, group in frame.groupby("player_id", sort=False):
        ordered = group.sort_values("date")
        dates = ordered["date"].values.astype("datetime64[D]").astype(np.int64).tolist()
        output[int(player_id)] = (dates, ordered[value_columns].to_numpy(dtype=float))
    return output


def position_group(position: str) -> str:
    text = str(position or "").lower()
    if "goalkeeper" in text:
        return "gk"
    if any(token in text for token in ("back", "sweeper")):
        return "def"
    if any(token in text for token in ("midfield",)):
        return "mid"
    return "att"


def player_snapshot(
    player_id: int,
    date_number: int,
    valuation_history: dict[int, tuple[list[int], np.ndarray]],
    appearance_history: dict[int, tuple[list[int], np.ndarray]],
) -> dict[str, float]:
    valuation = np.nan
    if player_id in valuation_history:
        dates, values = valuation_history[player_id]
        index = bisect.bisect_left(dates, date_number) - 1
        if index >= 0:
            valuation = float(values[index, 0])

    minutes = goals = assists = yellow = red = 0.0
    appearances = 0
    days_since = np.nan
    if player_id in appearance_history:
        dates, values = appearance_history[player_id]
        end = bisect.bisect_left(dates, date_number)
        start = max(0, end - 10)
        if end > start:
            recent = values[start:end]
            minutes, goals, assists, yellow, red = recent.sum(axis=0)
            appearances = end - start
            days_since = float(max(0, date_number - dates[end - 1]))
    per90 = 90.0 / max(minutes, 90.0)
    return {
        "value": valuation,
        "minutes_per_app": minutes / max(appearances, 1),
        "goals_per90": goals * per90,
        "assists_per90": assists * per90,
        "cards_per90": (yellow + 2.0 * red) * per90,
        "days_since": days_since,
    }


def lineup_features(
    lineup: dict[int, str] | None,
    previous_lineup: dict[int, str] | None,
    date: pd.Timestamp,
    birth_dates: dict[int, pd.Timestamp],
    valuation_history: dict[int, tuple[list[int], np.ndarray]],
    appearance_history: dict[int, tuple[list[int], np.ndarray]],
) -> dict[str, float]:
    empty = {
        "prior_lineup_available": 0.0,
        "prior_starter_count": 0.0,
        "prior_lineup_value_sum_m": np.nan,
        "prior_lineup_value_avg_m": np.nan,
        "prior_lineup_value_top3_m": np.nan,
        "prior_lineup_value_coverage": np.nan,
        "prior_lineup_age": np.nan,
        "prior_lineup_gk_share": np.nan,
        "prior_lineup_def_share": np.nan,
        "prior_lineup_mid_share": np.nan,
        "prior_lineup_att_share": np.nan,
        "prior_lineup_stability": np.nan,
        "prior_players_minutes_per_app": np.nan,
        "prior_players_goals_per90": np.nan,
        "prior_players_assists_per90": np.nan,
        "prior_players_cards_per90": np.nan,
        "prior_players_days_since": np.nan,
    }
    if not lineup:
        return empty

    date_number = int(date.to_datetime64().astype("datetime64[D]").astype(np.int64))
    snapshots = [
        player_snapshot(player_id, date_number, valuation_history, appearance_history)
        for player_id in lineup
    ]
    values = np.asarray([row["value"] for row in snapshots], dtype=float)
    finite_values = values[np.isfinite(values)]
    ages = [
        (date - birth_dates[player_id]).days / 365.25
        for player_id in lineup
        if player_id in birth_dates and pd.notna(birth_dates[player_id])
    ]
    positions = [position_group(position) for position in lineup.values()]
    count = float(len(lineup))
    stability = np.nan
    if previous_lineup:
        stability = len(set(lineup) & set(previous_lineup)) / max(len(lineup), 1)

    def mean_snapshot(key: str) -> float:
        values_for_key = np.asarray([row[key] for row in snapshots], dtype=float)
        return float(np.nanmean(values_for_key)) if np.isfinite(values_for_key).any() else np.nan

    return {
        "prior_lineup_available": 1.0,
        "prior_starter_count": count,
        "prior_lineup_value_sum_m": float(finite_values.sum() / 1_000_000.0) if finite_values.size else np.nan,
        "prior_lineup_value_avg_m": float(finite_values.mean() / 1_000_000.0) if finite_values.size else np.nan,
        "prior_lineup_value_top3_m": float(np.sort(finite_values)[-3:].sum() / 1_000_000.0) if finite_values.size else np.nan,
        "prior_lineup_value_coverage": float(finite_values.size / count),
        "prior_lineup_age": float(np.mean(ages)) if ages else np.nan,
        "prior_lineup_gk_share": positions.count("gk") / count,
        "prior_lineup_def_share": positions.count("def") / count,
        "prior_lineup_mid_share": positions.count("mid") / count,
        "prior_lineup_att_share": positions.count("att") / count,
        "prior_lineup_stability": stability,
        "prior_players_minutes_per_app": mean_snapshot("minutes_per_app"),
        "prior_players_goals_per90": mean_snapshot("goals_per90"),
        "prior_players_assists_per90": mean_snapshot("assists_per90"),
        "prior_players_cards_per90": mean_snapshot("cards_per90"),
        "prior_players_days_since": mean_snapshot("days_since"),
    }


def manager_features(
    club_id: int,
    manager: str,
    last_manager: dict[int, str],
    manager_history: dict[tuple[int, str], list[float]],
) -> dict[str, float]:
    manager = str(manager or "").strip()
    history = manager_history[(club_id, manager)] if manager else []
    return {
        "manager_available": float(bool(manager)),
        "manager_changed": float(bool(manager) and bool(last_manager.get(club_id)) and manager != last_manager[club_id]),
        "manager_matches": float(len(history)),
        "manager_ppg": float(np.mean([event[0] for event in history])) if history else np.nan,
        "manager_gd_per_match": float(np.mean([event[1] for event in history])) if history else np.nan,
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    games = pd.read_csv(
        source / "games.csv",
        usecols=[
            "game_id",
            "competition_id",
            "season",
            "date",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
            "home_club_manager_name",
            "away_club_manager_name",
        ],
        parse_dates=["date"],
    )
    games = games[games["competition_id"].eq("NL1")].sort_values(["date", "game_id"]).reset_index(drop=True)
    game_ids = set(games["game_id"].astype(int))
    starters = load_starters(source, game_ids)
    player_ids = set(starters["player_id"].astype(int))
    players, valuations, appearances = load_player_rows(source, player_ids)

    birth_dates = dict(zip(players["player_id"].astype(int), players["date_of_birth"]))
    squad_proxies = current_squad_proxies(players)
    valuation_history = timelines(valuations, ["market_value_in_eur"])
    appearance_history = timelines(
        appearances,
        ["minutes_played", "goals", "assists", "yellow_cards", "red_cards"],
    )
    lineup_by_game_club: dict[tuple[int, int], dict[int, str]] = {}
    for (game_id, club_id), group in starters.groupby(["game_id", "club_id"]):
        lineup_by_game_club[(int(game_id), int(club_id))] = {
            int(row.player_id): str(row.position) for row in group.itertuples(index=False)
        }

    last_lineups: dict[int, deque[dict[int, str]]] = defaultdict(lambda: deque(maxlen=2))
    last_manager: dict[int, str] = {}
    manager_history: dict[tuple[int, str], list[list[float]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for game in games.itertuples(index=False):
        home_id = int(game.home_club_id)
        away_id = int(game.away_club_id)
        home_name = CLUB_NAMES.get(home_id, str(home_id))
        away_name = CLUB_NAMES.get(away_id, str(away_id))
        date = pd.Timestamp(game.date)
        row: dict[str, Any] = {
            "match_key": match_key(date, home_name, away_name),
            "date": date,
            "home_team": home_name,
            "away_team": away_name,
            "tm_match_available": 1.0,
        }
        numeric_pairs: list[str] = []
        for side, club_id, manager in (
            ("home", home_id, game.home_club_manager_name),
            ("away", away_id, game.away_club_manager_name),
        ):
            previous = last_lineups[club_id][-1] if last_lineups[club_id] else None
            second_previous = last_lineups[club_id][-2] if len(last_lineups[club_id]) > 1 else None
            features = lineup_features(
                previous,
                second_previous,
                date,
                birth_dates,
                valuation_history,
                appearance_history,
            )
            features.update(manager_features(club_id, manager, last_manager, manager_history))
            for name, value in features.items():
                row[f"tm_{side}_{name}"] = value
                numeric_pairs.append(name)

        for name in sorted(set(numeric_pairs)):
            row[f"tm_diff_{name}"] = row[f"tm_home_{name}"] - row[f"tm_away_{name}"]
        rows.append(row)

        home_goals = float(game.home_club_goals)
        away_goals = float(game.away_club_goals)
        home_points = 3.0 if home_goals > away_goals else 1.0 if home_goals == away_goals else 0.0
        away_points = 3.0 if away_goals > home_goals else 1.0 if home_goals == away_goals else 0.0
        for club_id, manager, points, goal_diff in (
            (home_id, game.home_club_manager_name, home_points, home_goals - away_goals),
            (away_id, game.away_club_manager_name, away_points, away_goals - home_goals),
        ):
            manager = str(manager or "").strip()
            if manager:
                manager_history[(club_id, manager)].append([points, goal_diff])
                last_manager[club_id] = manager
        for club_id in (home_id, away_id):
            current = lineup_by_game_club.get((int(game.game_id), club_id))
            if current:
                last_lineups[club_id].append(current)

    historical_keys = {row["match_key"] for row in rows}
    fixtures_path = ROOT / "data" / "processed" / "fixtures.csv"
    fixture_rows = 0
    if fixtures_path.exists():
        fixtures = pd.read_csv(fixtures_path, parse_dates=["date"])
        club_ids = {name: club_id for club_id, name in CLUB_NAMES.items()}
        for fixture in fixtures.sort_values("date").itertuples(index=False):
            if fixture.match_key in historical_keys:
                continue
            home_name = str(fixture.home_team)
            away_name = str(fixture.away_team)
            if home_name not in club_ids or away_name not in club_ids:
                continue
            date = pd.Timestamp(fixture.date)
            row = {
                "match_key": fixture.match_key,
                "date": date,
                "home_team": home_name,
                "away_team": away_name,
                "tm_match_available": 0.0,
            }
            names: list[str] = []
            for side, team_name in (("home", home_name), ("away", away_name)):
                club_id = club_ids[team_name]
                lineup = squad_proxies.get(club_id)
                if not lineup and last_lineups[club_id]:
                    lineup = last_lineups[club_id][-1]
                previous = last_lineups[club_id][-1] if last_lineups[club_id] else None
                features = lineup_features(
                    lineup,
                    previous,
                    date,
                    birth_dates,
                    valuation_history,
                    appearance_history,
                )
                manager = last_manager.get(club_id, "")
                features.update(manager_features(club_id, manager, last_manager, manager_history))
                for name, value in features.items():
                    row[f"tm_{side}_{name}"] = value
                    names.append(name)
            for name in sorted(set(names)):
                row[f"tm_diff_{name}"] = row[f"tm_home_{name}"] - row[f"tm_away_{name}"]
            rows.append(row)
            fixture_rows += 1

    output = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)

    snapshot_date = pd.Timestamp.today().normalize()
    snapshot_number = int(snapshot_date.to_datetime64().astype("datetime64[D]").astype(np.int64))
    active = players[
        players["last_season"].ge(2025)
        & players["current_club_id"].isin(CLUB_NAMES)
    ].copy()
    squad_rows: list[dict[str, Any]] = []
    for player in active.itertuples(index=False):
        club_id = int(player.current_club_id)
        player_id = int(player.player_id)
        snapshot = player_snapshot(player_id, snapshot_number, valuation_history, appearance_history)
        value = snapshot["value"]
        if not np.isfinite(value):
            value = float(player.market_value_in_eur) if pd.notna(player.market_value_in_eur) else np.nan
        birth_date = pd.Timestamp(player.date_of_birth) if pd.notna(player.date_of_birth) else pd.NaT
        age = (snapshot_date - birth_date).days / 365.25 if pd.notna(birth_date) else np.nan
        position = player.sub_position if pd.notna(player.sub_position) else player.position
        squad_rows.append(
            {
                "player_id": player_id,
                "player": player.name,
                "team": CLUB_NAMES[club_id],
                "club_id": club_id,
                "position": position,
                "position_group": position_group(str(position)),
                "market_value_m": value / 1_000_000.0 if np.isfinite(value) else np.nan,
                "age": age,
                "minutes_per_app": snapshot["minutes_per_app"],
                "goals_per90": snapshot["goals_per90"],
                "assists_per90": snapshot["assists_per90"],
                "cards_per90": snapshot["cards_per90"],
                "days_since": snapshot["days_since"],
                "selected_proxy": int(player_id in squad_proxies.get(club_id, {})),
            }
        )
    current_squads = pd.DataFrame(squad_rows).sort_values(
        ["team", "selected_proxy", "market_value_m"], ascending=[True, False, False]
    )
    current_squads.to_csv(CURRENT_SQUADS, index=False)
    report = {
        "source": str(source),
        "games": int(len(games)),
        "games_with_22_starters": int(starters.groupby("game_id").size().eq(22).sum()),
        "players": int(len(player_ids)),
        "first_date": str(games["date"].min().date()),
        "last_date": str(games["date"].max().date()),
        "rows_with_both_prior_lineups": int(
            output[["tm_home_prior_lineup_available", "tm_away_prior_lineup_available"]].eq(1.0).all(axis=1).sum()
        ),
        "fixture_proxy_rows": fixture_rows,
        "current_squad_players": int(len(current_squads)),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
