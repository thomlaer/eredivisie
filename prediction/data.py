from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import (
    CURRENT_SEASON_START,
    ESPN_SCOREBOARD_URL,
    FIRST_SEASON_START,
    FIXTURE_FEED_URL,
    FOOTBALL_DATA_DIR,
    FOOTBALL_DATA_FIXTURES_URL,
    FOOTBALL_DATA_URL,
    MANUAL_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_directories,
)
from .names import canonical_team, name_key


USER_AGENT = "eredivisie-prediction/1.0"
ODDS_PRIORITY = (
    ("AvgCH", "AvgCD", "AvgCA", "closing_average"),
    ("B365CH", "B365CD", "B365CA", "closing_bet365"),
    ("PSCH", "PSCD", "PSCA", "closing_pinnacle"),
    ("AvgH", "AvgD", "AvgA", "opening_average"),
    ("PSH", "PSD", "PSA", "opening_pinnacle"),
    ("B365H", "B365D", "B365A", "opening_bet365"),
)


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def _download(url: str, destination: Path, *, timeout: int = 60) -> None:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 2 or getattr(exc.response, "status_code", 0) not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(1.5 * (attempt + 1))
    else:
        raise last_error or RuntimeError(f"Download failed: {url}")
    content = response.content
    if len(content) < 100:
        raise ValueError(f"Downloaded file is unexpectedly small: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)


def update_sources(force: bool = False) -> dict[str, Any]:
    ensure_directories()
    downloaded = 0
    reused = 0
    failures: list[str] = []
    for start_year in range(FIRST_SEASON_START, CURRENT_SEASON_START + 1):
        destination = FOOTBALL_DATA_DIR / f"N1_{season_code(start_year)}.csv"
        should_refresh = force or start_year >= CURRENT_SEASON_START or not destination.exists()
        if not should_refresh:
            reused += 1
            continue
        try:
            _download(FOOTBALL_DATA_URL.format(code=season_code(start_year)), destination)
            downloaded += 1
        except Exception as exc:  # source availability should not destroy an existing local rebuild
            failures.append(f"football_data_{start_year}: {exc}")
            if not destination.exists():
                continue

    fixture_path = RAW_DIR / "eredivisie_2026_fixtures.json"
    try:
        _download(FIXTURE_FEED_URL, fixture_path)
    except Exception as exc:
        failures.append(f"fixture_feed: {exc}")

    weekly_odds_path = RAW_DIR / "football_data_weekly_fixtures.csv"
    weekly_odds_status = "not_requested"
    try:
        _download(FOOTBALL_DATA_FIXTURES_URL, weekly_odds_path)
        weekly_odds_status = "ok"
    except Exception as exc:
        weekly_odds_status = f"unavailable:{type(exc).__name__}"
        failures.append(f"football_data_weekly_fixtures: {exc}")

    espn_path = RAW_DIR / "espn_ned1_scoreboard.json"
    espn_status = "not_requested"
    try:
        response = requests.get(
            ESPN_SCOREBOARD_URL,
            params={"limit": "1000", "dates": "20260801-20270601"},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("ESPN response is not a JSON object")
        espn_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        espn_status = f"ok:{len(payload.get('events', []))}"
    except Exception as exc:
        espn_status = f"unavailable:{type(exc).__name__}"
        failures.append(f"espn: {exc}")

    report = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "football_data_downloaded": downloaded,
        "football_data_reused": reused,
        "fixture_feed_available": fixture_path.exists(),
        "weekly_odds_status": weekly_odds_status,
        "espn_status": espn_status,
        "failures": failures,
    }
    (RAW_DIR / "source_update_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return report


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", on_bad_lines="skip")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _pick_odds(frame: pd.DataFrame) -> pd.DataFrame:
    home = pd.Series(np.nan, index=frame.index, dtype=float)
    draw = pd.Series(np.nan, index=frame.index, dtype=float)
    away = pd.Series(np.nan, index=frame.index, dtype=float)
    source = pd.Series("", index=frame.index, dtype=object)
    for home_col, draw_col, away_col, label in ODDS_PRIORITY:
        if not {home_col, draw_col, away_col}.issubset(frame.columns):
            continue
        h = _numeric(frame, home_col)
        d = _numeric(frame, draw_col)
        a = _numeric(frame, away_col)
        valid = home.isna() & h.gt(1.0) & d.gt(1.0) & a.gt(1.0)
        home.loc[valid] = h.loc[valid]
        draw.loc[valid] = d.loc[valid]
        away.loc[valid] = a.loc[valid]
        source.loc[valid] = label
    output = pd.DataFrame(
        {"odds_home": home, "odds_draw": draw, "odds_away": away, "odds_source": source},
        index=frame.index,
    )
    implied = 1.0 / output[["odds_home", "odds_draw", "odds_away"]]
    overround = implied.sum(axis=1, min_count=3)
    output["odds_overround"] = overround
    output["market_prob_home"] = implied["odds_home"] / overround
    output["market_prob_draw"] = implied["odds_draw"] / overround
    output["market_prob_away"] = implied["odds_away"] / overround
    output["has_odds"] = output[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1).astype(int)
    return output


def _first_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        values = _numeric(frame, column)
        output = output.where(output.notna(), values)
    return output


def _poisson_total_lambda(over_probability: float) -> float:
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


def _pick_score_markets(frame: pd.DataFrame) -> pd.DataFrame:
    over_odds = _first_numeric(
        frame,
        ["AvgC>2.5", "PC>2.5", "B365C>2.5", "Avg>2.5", "P>2.5", "B365>2.5", "BbAv>2.5"],
    )
    under_odds = _first_numeric(
        frame,
        ["AvgC<2.5", "PC<2.5", "B365C<2.5", "Avg<2.5", "P<2.5", "B365<2.5", "BbAv<2.5"],
    )
    over_implied = 1.0 / over_odds
    under_implied = 1.0 / under_odds
    total_overround = over_implied + under_implied

    handicap = _first_numeric(frame, ["AHCh", "AHh", "BbAHh"])
    handicap_home_odds = _first_numeric(
        frame,
        ["AvgCAHH", "PCAHH", "B365CAHH", "AvgAHH", "PAHH", "B365AHH", "BbAvAHH"],
    )
    handicap_away_odds = _first_numeric(
        frame,
        ["AvgCAHA", "PCAHA", "B365CAHA", "AvgAHA", "PAHA", "B365AHA", "BbAvAHA"],
    )
    handicap_home_implied = 1.0 / handicap_home_odds
    handicap_away_implied = 1.0 / handicap_away_odds
    handicap_overround = handicap_home_implied + handicap_away_implied

    output = pd.DataFrame(index=frame.index)
    output["score_odds_over25"] = over_odds
    output["score_odds_under25"] = under_odds
    output["score_market_prob_over25"] = over_implied / total_overround
    output["score_market_total_overround"] = total_overround
    output["score_market_handicap"] = handicap
    output["score_market_handicap_home_prob"] = handicap_home_implied / handicap_overround
    output["score_market_handicap_away_prob"] = handicap_away_implied / handicap_overround
    output["score_market_total_lambda"] = output["score_market_prob_over25"].map(_poisson_total_lambda)
    output["score_market_goal_diff_proxy"] = -output["score_market_handicap"]
    output["score_market_home_xg_proxy"] = (
        output["score_market_total_lambda"] + output["score_market_goal_diff_proxy"]
    ) / 2.0
    output["score_market_away_xg_proxy"] = (
        output["score_market_total_lambda"] - output["score_market_goal_diff_proxy"]
    ) / 2.0
    output["score_market_has_totals"] = output[["score_odds_over25", "score_odds_under25"]].notna().all(axis=1).astype(int)
    output["score_market_has_handicap"] = output[
        ["score_market_handicap", "score_market_handicap_home_prob", "score_market_handicap_away_prob"]
    ].notna().all(axis=1).astype(int)
    return output


def load_historical_matches() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted(FOOTBALL_DATA_DIR.glob("N1_*.csv")):
        code = path.stem.split("_")[-1]
        start_suffix = int(code[:2])
        start_year = 1900 + start_suffix if start_suffix >= 90 else 2000 + start_suffix
        raw = _read_csv_flexible(path)
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not required.issubset(raw.columns):
            continue
        match_date = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed")
        output = pd.DataFrame(
            {
                "date": match_date,
                "season_start": start_year,
                "season": season_label(start_year),
                "home_team": raw["HomeTeam"].map(canonical_team),
                "away_team": raw["AwayTeam"].map(canonical_team),
                "home_goals": _numeric(raw, "FTHG"),
                "away_goals": _numeric(raw, "FTAG"),
                "half_home_goals": _numeric(raw, "HTHG"),
                "half_away_goals": _numeric(raw, "HTAG"),
                "home_xg": _numeric(raw, "HxG"),
                "away_xg": _numeric(raw, "AxG"),
                "home_shots": _numeric(raw, "HS"),
                "away_shots": _numeric(raw, "AS"),
                "home_shots_target": _numeric(raw, "HST"),
                "away_shots_target": _numeric(raw, "AST"),
                "home_corners": _numeric(raw, "HC"),
                "away_corners": _numeric(raw, "AC"),
                "home_yellow": _numeric(raw, "HY"),
                "away_yellow": _numeric(raw, "AY"),
                "home_red": _numeric(raw, "HR"),
                "away_red": _numeric(raw, "AR"),
                "source_file": path.name,
            }
        )
        output = pd.concat(
            [
                output,
                _pick_odds(raw).reset_index(drop=True),
                _pick_score_markets(raw).reset_index(drop=True),
            ],
            axis=1,
        )
        pieces.append(output)

    if not pieces:
        raise FileNotFoundError("No Football-Data Eredivisie CSV files found.")
    matches = pd.concat(pieces, ignore_index=True)
    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches["match_key"] = (
        matches["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + matches["home_team"].map(name_key)
        + "|"
        + matches["away_team"].map(name_key)
    )
    matches = matches.sort_values(["date", "season_start", "home_team", "away_team"])
    matches = matches.drop_duplicates("match_key", keep="last").reset_index(drop=True)
    return matches


def load_fixtures() -> pd.DataFrame:
    path = RAW_DIR / "eredivisie_2026_fixtures.json"
    if not path.exists():
        raise FileNotFoundError(f"Fixture feed is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for item in payload:
        kickoff = pd.to_datetime(item.get("DateUtc"), utc=True, errors="coerce")
        rows.append(
            {
                "match_number": item.get("MatchNumber"),
                "round": item.get("RoundNumber"),
                "kickoff_utc": kickoff,
                "date": kickoff.tz_convert(None).normalize() if pd.notna(kickoff) else pd.NaT,
                "venue": item.get("Location") or "",
                "home_team": canonical_team(item.get("HomeTeam")),
                "away_team": canonical_team(item.get("AwayTeam")),
                "feed_home_goals": item.get("HomeTeamScore"),
                "feed_away_goals": item.get("AwayTeamScore"),
            }
        )
    fixtures = pd.DataFrame(rows).dropna(subset=["date", "home_team", "away_team"])
    fixtures["season_start"] = CURRENT_SEASON_START
    fixtures["season"] = season_label(CURRENT_SEASON_START)
    fixtures["match_key"] = (
        fixtures["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + fixtures["home_team"].map(name_key)
        + "|"
        + fixtures["away_team"].map(name_key)
    )
    return fixtures.sort_values(["kickoff_utc", "match_number"]).reset_index(drop=True)


def load_manual_upcoming_odds() -> pd.DataFrame:
    path = MANUAL_DIR / "upcoming_odds.csv"
    if not path.exists():
        return pd.DataFrame()
    odds = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"}
    if not required.issubset(odds.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce")
    odds["home_team"] = odds["home_team"].map(canonical_team)
    odds["away_team"] = odds["away_team"].map(canonical_team)
    odds["match_key"] = (
        odds["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + odds["home_team"].map(name_key)
        + "|"
        + odds["away_team"].map(name_key)
    )
    for column in ("odds_home", "odds_draw", "odds_away"):
        odds[column] = pd.to_numeric(odds[column], errors="coerce")
    implied = 1.0 / odds[["odds_home", "odds_draw", "odds_away"]]
    odds["odds_overround"] = implied.sum(axis=1, min_count=3)
    odds["market_prob_home"] = implied["odds_home"] / odds["odds_overround"]
    odds["market_prob_draw"] = implied["odds_draw"] / odds["odds_overround"]
    odds["market_prob_away"] = implied["odds_away"] / odds["odds_overround"]
    odds["odds_source"] = odds.get("odds_source", "manual_upcoming")
    odds["has_odds"] = 1
    return odds.dropna(subset=["date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"])


def load_weekly_upcoming_odds() -> pd.DataFrame:
    path = RAW_DIR / "football_data_weekly_fixtures.csv"
    if not path.exists():
        return pd.DataFrame()
    raw = _read_csv_flexible(path)
    if "Div" not in raw or not {"Date", "HomeTeam", "AwayTeam"}.issubset(raw.columns):
        return pd.DataFrame()
    raw = raw[raw["Div"].eq("N1")].reset_index(drop=True)
    if raw.empty:
        return pd.DataFrame()
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed"),
            "home_team": raw["HomeTeam"].map(canonical_team),
            "away_team": raw["AwayTeam"].map(canonical_team),
        }
    )
    output = pd.concat(
        [output, _pick_odds(raw).reset_index(drop=True), _pick_score_markets(raw).reset_index(drop=True)],
        axis=1,
    )
    output["match_key"] = (
        output["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + output["home_team"].map(name_key)
        + "|"
        + output["away_team"].map(name_key)
    )
    return output.dropna(subset=["date", "home_team", "away_team"])


def append_fixture_results(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    feed_results = fixtures[
        fixtures[["feed_home_goals", "feed_away_goals"]].notna().all(axis=1)
        & ~fixtures["match_key"].isin(matches["match_key"])
    ].copy()
    if not feed_results.empty:
        feed_results["home_goals"] = pd.to_numeric(
            feed_results["feed_home_goals"], errors="coerce"
        ).astype(int)
        feed_results["away_goals"] = pd.to_numeric(
            feed_results["feed_away_goals"], errors="coerce"
        ).astype(int)
        feed_results["source_file"] = "fixture_feed_current"
        feed_results["odds_source"] = ""
        feed_results["has_odds"] = 0
        feed_results["score_market_has_totals"] = 0
        feed_results["score_market_has_handicap"] = 0
        feed_results = feed_results.reindex(columns=matches.columns)
        matches = pd.concat([matches, feed_results], ignore_index=True)
        matches = matches.sort_values(["date", "season_start", "home_team", "away_team"])
        matches = matches.drop_duplicates("match_key", keep="first").reset_index(drop=True)
    return matches


def build_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = load_historical_matches()
    fixtures = load_fixtures()
    matches = append_fixture_results(matches, fixtures)
    current_results = matches[matches["season_start"] == CURRENT_SEASON_START][
        ["match_key", "home_goals", "away_goals"]
    ]
    fixtures = fixtures.merge(current_results, on="match_key", how="left")
    fixtures["is_played"] = fixtures[["home_goals", "away_goals"]].notna().all(axis=1)

    odds_columns = [
        "odds_home",
        "odds_draw",
        "odds_away",
        "odds_source",
        "odds_overround",
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
        "has_odds",
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
    weekly_odds = load_weekly_upcoming_odds()
    if not weekly_odds.empty:
        fixtures = fixtures.merge(weekly_odds[["match_key", *odds_columns]], on="match_key", how="left")
    else:
        for column in odds_columns:
            fixtures[column] = "" if column == "odds_source" else np.nan

    manual_odds = load_manual_upcoming_odds()
    if not manual_odds.empty:
        manual_columns = [column for column in odds_columns if column in manual_odds.columns]
        fixtures = fixtures.merge(
            manual_odds[["match_key", *manual_columns]], on="match_key", how="left", suffixes=("", "_manual")
        )
        for column in manual_columns:
            manual_column = f"{column}_manual"
            fixtures[column] = fixtures[manual_column].where(fixtures[manual_column].notna(), fixtures[column])
            fixtures = fixtures.drop(columns=manual_column)
    fixtures["has_odds"] = fixtures["has_odds"].fillna(0).astype(int)
    fixtures["score_market_has_totals"] = fixtures["score_market_has_totals"].fillna(0).astype(int)
    fixtures["score_market_has_handicap"] = fixtures["score_market_has_handicap"].fillna(0).astype(int)

    matches.to_csv(PROCESSED_DIR / "matches.csv", index=False)
    fixtures.to_csv(PROCESSED_DIR / "fixtures.csv", index=False)
    return matches, fixtures
