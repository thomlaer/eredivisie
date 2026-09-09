from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import MANUAL_DIR, RAW_DIR
from .names import canonical_team, name_key


ROOT = Path(__file__).resolve().parents[1]
TRANSFERMARKT_INJURY_URL = (
    "https://www.transfermarkt.nl/eredivisie/verletztespieler/wettbewerb/NL1/plus/1"
)
TRANSFERMARKT_SUSPENSION_URL = (
    "https://www.transfermarkt.nl/eredivisie/sperrenausfaelle/wettbewerb/NL1/plus/1"
)
TRANSFERMARKT_LEAGUE_URL = "https://www.transfermarkt.nl/eredivisie/startseite/wettbewerb/NL1"
TRANSFERMARKT_MINUTES_URL = (
    "https://www.transfermarkt.nl/eredivisie/dauerbrenner/wettbewerb/NL1/pos//detailpos/0/"
    "saison_id/2026/plus/1"
)
TRANSFERMARKT_SCORERS_URL = (
    "https://www.transfermarkt.nl/eredivisie/scorerliste/wettbewerb/NL1/saison_id/2026/plus/1"
)
INJURY_HTML = RAW_DIR / "transfermarkt_eredivisie_injuries.html"
INJURY_CSV = RAW_DIR / "transfermarkt_eredivisie_injuries.csv"
SUSPENSION_HTML = RAW_DIR / "transfermarkt_eredivisie_suspensions.html"
SUSPENSION_CSV = RAW_DIR / "transfermarkt_eredivisie_suspensions.csv"
LEAGUE_HTML = RAW_DIR / "transfermarkt_eredivisie_clubs.html"
LIVE_SQUADS = RAW_DIR / "transfermarkt_eredivisie_current_squads.csv"
CURRENT_PERFORMANCE = RAW_DIR / "transfermarkt_eredivisie_current_performance.csv"
CURRENT_SQUADS = ROOT / "data" / "external" / "transfermarkt_current_squads.csv"
MANUAL_AVAILABILITY = MANUAL_DIR / "player_availability.csv"
USER_AGENT = "Mozilla/5.0 (compatible; EredivisiePrediction/1.0; local analytics)"
SQUAD_CACHE_HOURS = 20.0

CLUB_IDS = {
    132: "NAC Breda",
    133: "SC Cambuur",
    200: "FC Utrecht",
    202: "FC Groningen",
    234: "Feyenoord",
    306: "Heerenveen",
    317: "FC Twente",
    383: "PSV",
    385: "Fortuna Sittard",
    403: "Willem II",
    467: "NEC",
    468: "Sparta Rotterdam",
    610: "Ajax",
    798: "Excelsior",
    1090: "AZ",
    1268: "ADO Den Haag",
    1269: "PEC Zwolle",
    1434: "Telstar",
    1435: "Go Ahead Eagles",
}

MANUAL_COLUMNS = [
    "team",
    "player",
    "player_id",
    "status",
    "availability",
    "available_from",
    "note",
]


def _id_from_href(href: str | None, entity: str) -> int | None:
    match = re.search(rf"/{entity}/(\d+)", str(href or ""))
    return int(match.group(1)) if match else None


def _market_value_m(value: object) -> float:
    text = str(value or "").replace("\xa0", " ").lower().strip()
    match = re.search(r"([\d.,]+)\s*(mln|dzd|mio|ths)", text)
    if not match:
        return np.nan
    number = float(match.group(1).replace(".", "").replace(",", "."))
    return number if match.group(2) in {"mln", "mio"} else number / 1000.0


def _dutch_date(value: object) -> str:
    text = str(value or "").strip().lower().replace(".", "")
    match = re.fullmatch(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", text)
    months = {
        "jan": 1, "feb": 2, "mrt": 3, "apr": 4, "mei": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dec": 12,
    }
    if not match or match.group(2) not in months:
        return ""
    return f"{int(match.group(3)):04d}-{months[match.group(2)]:02d}-{int(match.group(1)):02d}"


def parse_transfermarkt_injuries(content: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    table = soup.select_one("table.items")
    if table is None or table.tbody is None:
        raise ValueError("Transfermarkt injury table was not found")
    rows: list[dict[str, Any]] = []
    for table_row in table.tbody.find_all("tr", recursive=False):
        cells = table_row.find_all("td", recursive=False)
        if len(cells) < 8:
            continue
        player_link = cells[0].select_one('a[href*="/spieler/"]')
        team_link = cells[1].select_one('a[href*="/verein/"]')
        if player_link is None or team_link is None:
            continue
        player_id = _id_from_href(player_link.get("href"), "spieler")
        club_id = _id_from_href(team_link.get("href"), "verein")
        player = player_link.get("title") or player_link.get_text(" ", strip=True)
        cell_text = list(cells[0].stripped_strings)
        position = next((text for text in cell_text if name_key(text) != name_key(player)), "")
        since = cells[5].get_text(" ", strip=True)
        expected_return = cells[6].get_text(" ", strip=True)
        rows.append(
            {
                "player_id": player_id,
                "player": player,
                "team": CLUB_IDS.get(club_id, canonical_team(team_link.get("title") or "")),
                "club_id": club_id,
                "position": position,
                "injury": cells[4].get_text(" ", strip=True),
                "since": since,
                "since_date": _dutch_date(since),
                "expected_return": expected_return,
                "expected_return_date": _dutch_date(expected_return),
                "market_value_m": _market_value_m(cells[7].get_text(" ", strip=True)),
                "availability": 0.0,
                "absence_type": "injury",
                "source": "Transfermarkt",
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("Transfermarkt returned an empty injury table")
    return output.drop_duplicates(["team", "player"], keep="first").reset_index(drop=True)


def parse_transfermarkt_suspensions(content: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    table = soup.select_one("table.items")
    if table is None or table.tbody is None:
        raise ValueError("Transfermarkt suspension table was not found")
    rows: list[dict[str, Any]] = []
    for table_row in table.tbody.find_all("tr", recursive=False):
        cells = table_row.find_all("td", recursive=False)
        if len(cells) < 7:
            continue
        player_link = cells[0].select_one('a[href*="/spieler/"]')
        team_link = cells[1].select_one('a[href*="/verein/"]')
        if player_link is None or team_link is None:
            continue
        player = player_link.get("title") or player_link.get_text(" ", strip=True)
        cell_text = list(cells[0].stripped_strings)
        position = next((text for text in cell_text if name_key(text) != name_key(player)), "")
        club_id = _id_from_href(team_link.get("href"), "verein")
        since = cells[4].get_text(" ", strip=True)
        expected_return = cells[5].get_text(" ", strip=True)
        rows.append(
            {
                "player_id": _id_from_href(player_link.get("href"), "spieler"),
                "player": player,
                "team": CLUB_IDS.get(club_id, canonical_team(team_link.get("title") or "")),
                "club_id": club_id,
                "position": position,
                "injury": cells[3].get_text(" ", strip=True),
                "since": since,
                "since_date": _dutch_date(since),
                "expected_return": expected_return,
                "expected_return_date": _dutch_date(expected_return),
                "market_value_m": np.nan,
                "availability": 0.0,
                "absence_type": "suspension",
                "source": "Transfermarkt",
            }
        )
    return pd.DataFrame(rows)


def parse_transfermarkt_clubs(content: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    candidate_tables = [
        table for table in soup.select("table.items")
        if table.select_one('a[href*="/kader/verein/"]') is not None
    ]
    if not candidate_tables:
        raise ValueError("Transfermarkt Eredivisie club table was not found")
    rows: list[dict[str, Any]] = []
    for table_row in candidate_tables[0].select("tbody > tr"):
        squad_link = table_row.select_one('a[href*="/kader/verein/"]')
        club_link = table_row.select_one('a[href*="/startseite/verein/"]')
        if squad_link is None or club_link is None:
            continue
        club_id = _id_from_href(squad_link.get("href"), "verein")
        if club_id is None:
            continue
        title = club_link.get("title") or club_link.get_text(" ", strip=True)
        href = str(squad_link.get("href") or "")
        if "/plus/1" not in href:
            href = href.rstrip("/") + "/plus/1"
        rows.append(
            {
                "club_id": club_id,
                "team": CLUB_IDS.get(club_id, canonical_team(title)),
                "squad_url": urljoin("https://www.transfermarkt.nl", href),
                "performance_url": urljoin(
                    "https://www.transfermarkt.nl",
                    re.sub(
                        r"/kader/verein/(\d+)/saison_id/(\d+)",
                        r"/leistungsdaten/verein/\1/saison/\2",
                        href,
                    ),
                ),
            }
        )
    output = pd.DataFrame(rows).drop_duplicates("club_id", keep="first")
    if len(output) < 18:
        raise ValueError(f"Transfermarkt returned only {len(output)} Eredivisie clubs")
    return output.reset_index(drop=True)


def _position_group(position: object) -> str:
    text = str(position or "").lower()
    if "keeper" in text or "goalkeeper" in text:
        return "gk"
    if any(token in text for token in ("verdedig", "back", "defender", "sweeper")):
        return "def"
    if any(token in text for token in ("middenveld", "midfield")):
        return "mid"
    return "att"


def parse_transfermarkt_squad(content: bytes, team: str, club_id: int) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    table = soup.select_one("table.items")
    if table is None or table.tbody is None:
        raise ValueError(f"Transfermarkt squad table was not found for {team}")
    rows: list[dict[str, Any]] = []
    for table_row in table.tbody.find_all("tr", recursive=False):
        cells = table_row.find_all("td", recursive=False)
        if len(cells) < 9:
            continue
        player_link = table_row.select_one('a[href*="/profil/spieler/"]')
        if player_link is None:
            continue
        player_id = _id_from_href(player_link.get("href"), "spieler")
        player = player_link.get("title") or player_link.get_text(" ", strip=True)
        player_cell = player_link.find_parent("td")
        player_container = player_cell.find_parent("td") if player_cell is not None else None
        texts = list((player_container or cells[1]).stripped_strings)
        position = next((text for text in texts if name_key(text) != name_key(player)), "")
        age_match = re.search(r"\((\d{1,2})\)", cells[2].get_text(" ", strip=True))
        value_cell = next(
            (cell for cell in reversed(cells) if cell.select_one('a[href*="/marktwertverlauf/spieler/"]')),
            cells[-1],
        )
        rows.append(
            {
                "player_id": player_id,
                "player": player,
                "team": canonical_team(team),
                "club_id": int(club_id),
                "position": position,
                "position_group": _position_group(position),
                "market_value_m": _market_value_m(value_cell.get_text(" ", strip=True)),
                "age": float(age_match.group(1)) if age_match else np.nan,
                "snapshot_date": datetime.now(timezone.utc).date().isoformat(),
                "roster_source": "Transfermarkt live",
            }
        )
    output = pd.DataFrame(rows).drop_duplicates("player_id", keep="first")
    if len(output) < 11:
        raise ValueError(f"Transfermarkt returned only {len(output)} squad players for {team}")
    return output.reset_index(drop=True)


def _integer(value: object) -> int:
    match = re.search(r"\d[\d.]*", str(value or ""))
    return int(match.group(0).replace(".", "")) if match else 0


def _player_and_club(table_row: Any) -> tuple[int | None, str, int | None, str, str]:
    player_link = table_row.select_one('a[href*="/profil/spieler/"]')
    club_link = table_row.select_one('a[href*="/startseite/verein/"]')
    if player_link is None or club_link is None:
        return None, "", None, "", ""
    player = player_link.get("title") or player_link.get_text(" ", strip=True)
    player_cell = player_link.find_parent("td")
    player_container = player_cell.find_parent("td") if player_cell is not None else None
    texts = list((player_container or player_cell).stripped_strings) if player_cell is not None else []
    position = next((text for text in texts if name_key(text) != name_key(player)), "")
    club_id = _id_from_href(club_link.get("href"), "verein")
    team = CLUB_IDS.get(club_id, canonical_team(club_link.get("title") or ""))
    return _id_from_href(player_link.get("href"), "spieler"), player, club_id, team, position


def parse_transfermarkt_minutes(content: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    table = soup.select_one("table.items")
    if table is None or table.tbody is None:
        raise ValueError("Transfermarkt minutes table was not found")
    rows: list[dict[str, Any]] = []
    for table_row in table.tbody.find_all("tr", recursive=False):
        cells = table_row.find_all("td", recursive=False)
        if len(cells) < 10:
            continue
        player_id, player, club_id, team, position = _player_and_club(table_row)
        if player_id is None or club_id is None:
            continue
        rows.append(
            {
                "player_id": player_id,
                "player": player,
                "club_id": club_id,
                "team": team,
                "position": position,
                "season_appearances": _integer(cells[5].get_text(" ", strip=True)),
                "season_minutes": _integer(cells[8].get_text(" ", strip=True)),
                "season_goals": _integer(cells[9].get_text(" ", strip=True)),
            }
        )
    return pd.DataFrame(rows)


def parse_transfermarkt_scorers(content: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    table = soup.select_one("table.items")
    if table is None or table.tbody is None:
        raise ValueError("Transfermarkt scorer table was not found")
    rows: list[dict[str, Any]] = []
    for table_row in table.tbody.find_all("tr", recursive=False):
        cells = table_row.find_all("td", recursive=False)
        if len(cells) < 11:
            continue
        player_id, player, club_id, team, position = _player_and_club(table_row)
        if player_id is None or club_id is None:
            continue
        rows.append(
            {
                "player_id": player_id,
                "player": player,
                "club_id": club_id,
                "team": team,
                "position": position,
                "season_appearances": _integer(cells[5].get_text(" ", strip=True)),
                "season_goals": _integer(cells[8].get_text(" ", strip=True)),
                "season_assists": _integer(cells[9].get_text(" ", strip=True)),
            }
        )
    return pd.DataFrame(rows)


def parse_transfermarkt_club_performance(
    content: bytes,
    team: str,
    club_id: int,
) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    table = soup.select_one("table.items")
    if table is None or table.tbody is None:
        raise ValueError(f"Transfermarkt performance table was not found for {team}")
    rows: list[dict[str, Any]] = []
    for table_row in table.tbody.find_all("tr", recursive=False):
        cells = table_row.find_all("td", recursive=False)
        if len(cells) < 15:
            continue
        player_link = table_row.select_one('a[href*="/profil/spieler/"]')
        if player_link is None:
            continue
        appearances_text = cells[5].get_text(" ", strip=True)
        rows.append(
            {
                "player_id": _id_from_href(player_link.get("href"), "spieler"),
                "player": player_link.get("title") or player_link.get_text(" ", strip=True),
                "club_id": club_id,
                "team": canonical_team(team),
                "season_appearances": 0 if "niet ingezet" in appearances_text.lower() else _integer(appearances_text),
                "season_minutes": _integer(cells[14].get_text(" ", strip=True)),
                "season_goals": _integer(cells[6].get_text(" ", strip=True)),
                "season_assists": _integer(cells[7].get_text(" ", strip=True)),
                "season_yellow_cards": _integer(cells[8].get_text(" ", strip=True)),
                "season_second_yellow": _integer(cells[9].get_text(" ", strip=True)),
                "season_red_cards": _integer(cells[10].get_text(" ", strip=True)),
            }
        )
    output = pd.DataFrame(rows).drop_duplicates("player_id", keep="first")
    if output.empty:
        raise ValueError(f"Transfermarkt returned no performance rows for {team}")
    return output.reset_index(drop=True)


def _pagination_pages(content: bytes) -> int:
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    pages = [_integer(link.get_text(" ", strip=True)) for link in soup.select(".tm-pagination a")]
    return max([page for page in pages if page > 0], default=1)


def _paged_performance(
    session: requests.Session,
    base_url: str,
    parser: Any,
) -> pd.DataFrame:
    first = _get(session, base_url)
    frames = [parser(first.content)]
    for page in range(2, _pagination_pages(first.content) + 1):
        response = _get(session, f"{base_url}/page/{page}")
        frames.append(parser(response.content))
        time.sleep(0.12)
    return pd.concat(frames, ignore_index=True).drop_duplicates("player_id", keep="first")


def refresh_current_performance(session: requests.Session) -> str:
    minutes = _paged_performance(session, TRANSFERMARKT_MINUTES_URL, parse_transfermarkt_minutes)
    scorers = _paged_performance(session, TRANSFERMARKT_SCORERS_URL, parse_transfermarkt_scorers)
    performance = minutes.merge(
        scorers[["player_id", "season_goals", "season_assists"]],
        on="player_id",
        how="outer",
        suffixes=("", "_scorer"),
    )
    performance["season_goals"] = pd.to_numeric(
        performance["season_goals_scorer"], errors="coerce"
    ).fillna(pd.to_numeric(performance["season_goals"], errors="coerce")).fillna(0)
    performance["season_assists"] = pd.to_numeric(
        performance["season_assists"], errors="coerce"
    ).fillna(0)
    performance = performance.drop(columns="season_goals_scorer", errors="ignore")
    temporary = CURRENT_PERFORMANCE.with_suffix(".csv.tmp")
    performance.to_csv(temporary, index=False)
    temporary.replace(CURRENT_PERFORMANCE)
    return f"ok:{len(performance)}"


def _read_row_count(path: Path) -> int:
    try:
        return int(len(pd.read_csv(path))) if path.exists() else 0
    except (OSError, pd.errors.EmptyDataError):
        return 0


def _cache_is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists() or _read_row_count(path) == 0:
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours <= max_age_hours


def _get(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response


def _enrich_live_squads(live: pd.DataFrame) -> pd.DataFrame:
    form_columns = ["minutes_per_app", "goals_per90", "assists_per90", "cards_per90", "days_since"]
    if CURRENT_SQUADS.exists():
        historical = pd.read_csv(CURRENT_SQUADS)
        available = ["player_id", *[column for column in form_columns if column in historical]]
        historical = historical[available].drop_duplicates("player_id", keep="first")
        live = live.merge(historical, on="player_id", how="left")
    if CURRENT_PERFORMANCE.exists():
        performance = pd.read_csv(CURRENT_PERFORMANCE)
        performance_columns = [
            "player_id",
            "season_appearances",
            "season_minutes",
            "season_goals",
            "season_assists",
            "season_yellow_cards",
            "season_second_yellow",
            "season_red_cards",
        ]
        available_performance_columns = [
            column for column in performance_columns if column in performance.columns
        ]
        live = live.merge(performance[available_performance_columns], on="player_id", how="left")
    count_columns = (
        "season_appearances",
        "season_minutes",
        "season_goals",
        "season_assists",
        "season_yellow_cards",
        "season_second_yellow",
        "season_red_cards",
    )
    for column in count_columns:
        if column not in live:
            live[column] = 0.0
        live[column] = pd.to_numeric(live[column], errors="coerce").fillna(0.0)
    for column in form_columns:
        if column not in live:
            live[column] = np.nan
    current_minutes = live["season_minutes"]
    prior_minutes = 900.0
    for output_column, count_column in (("goals_per90", "season_goals"), ("assists_per90", "season_assists")):
        prior_rate = pd.to_numeric(live[output_column], errors="coerce").fillna(0.0)
        live[output_column] = (
            prior_rate * prior_minutes + live[count_column] * 90.0
        ) / (prior_minutes + current_minutes)
    current_card_equivalents = (
        live["season_yellow_cards"]
        + 2.0 * live["season_second_yellow"]
        + 3.0 * live["season_red_cards"]
    )
    prior_cards_per90 = pd.to_numeric(live["cards_per90"], errors="coerce").fillna(0.0)
    live["cards_per90"] = (
        prior_cards_per90 * prior_minutes + current_card_equivalents * 90.0
    ) / (prior_minutes + current_minutes)
    live["minutes_per_app"] = np.where(
        live["season_appearances"].gt(0),
        live["season_minutes"] / live["season_appearances"],
        live["minutes_per_app"],
    )
    live["selected_proxy"] = 0
    for _, indices in live.groupby("team").groups.items():
        group = live.loc[indices].copy()
        group["availability"] = 1.0
        selected = _select_lineup(group)
        live.loc[selected.index, "selected_proxy"] = 1
    return live.sort_values(
        ["team", "selected_proxy", "market_value_m"], ascending=[True, False, False]
    ).reset_index(drop=True)


def refresh_current_squads(session: requests.Session, force: bool = False) -> str:
    if not force and _cache_is_fresh(LIVE_SQUADS, SQUAD_CACHE_HOURS):
        return f"cached:{_read_row_count(LIVE_SQUADS)}"
    league = _get(session, TRANSFERMARKT_LEAGUE_URL)
    clubs = parse_transfermarkt_clubs(league.content)
    squad_frames: list[pd.DataFrame] = []
    performance_frames: list[pd.DataFrame] = []
    performance_failures = 0
    for club in clubs.itertuples(index=False):
        response = _get(session, club.squad_url)
        squad_frames.append(parse_transfermarkt_squad(response.content, club.team, int(club.club_id)))
        try:
            performance_response = _get(session, club.performance_url)
            performance_frames.append(
                parse_transfermarkt_club_performance(
                    performance_response.content,
                    club.team,
                    int(club.club_id),
                )
            )
        except Exception:
            performance_failures += 1
        time.sleep(0.12)
    if performance_frames:
        performance = pd.concat(performance_frames, ignore_index=True).drop_duplicates("player_id", keep="first")
        temporary = CURRENT_PERFORMANCE.with_suffix(".csv.tmp")
        performance.to_csv(temporary, index=False)
        temporary.replace(CURRENT_PERFORMANCE)
        performance_status = f"ok:{len(performance)}:{len(performance_frames)}clubs"
        if performance_failures:
            performance_status += f":{performance_failures}failed"
    else:
        try:
            performance_status = refresh_current_performance(session)
        except Exception as exc:
            performance_status = f"unavailable:{type(exc).__name__}"
    live = _enrich_live_squads(pd.concat(squad_frames, ignore_index=True))
    if live["team"].nunique() != clubs["team"].nunique():
        raise ValueError("Not every Eredivisie club received a live squad")
    temporary = LIVE_SQUADS.with_suffix(".csv.tmp")
    live.to_csv(temporary, index=False)
    temporary.replace(LIVE_SQUADS)
    LEAGUE_HTML.write_bytes(league.content)
    return f"ok:{len(live)}:{live['team'].nunique()}clubs:performance={performance_status}"


def refresh_injuries(force_squads: bool = False) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": USER_AGENT, "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.7"}
    )
    injury_status = "not_requested"
    suspension_status = "not_requested"
    squad_status = "not_requested"
    try:
        response = _get(session, TRANSFERMARKT_INJURY_URL)
        injuries = parse_transfermarkt_injuries(response.content)
        temporary = INJURY_HTML.with_suffix(".html.tmp")
        temporary.write_bytes(response.content)
        temporary.replace(INJURY_HTML)
        injuries.to_csv(INJURY_CSV, index=False)
        injury_status = f"ok:{len(injuries)}"
    except Exception as exc:
        injury_status = f"unavailable:{type(exc).__name__}"
    try:
        response = _get(session, TRANSFERMARKT_SUSPENSION_URL)
        suspensions = parse_transfermarkt_suspensions(response.content)
        temporary = SUSPENSION_HTML.with_suffix(".html.tmp")
        temporary.write_bytes(response.content)
        temporary.replace(SUSPENSION_HTML)
        suspensions.to_csv(SUSPENSION_CSV, index=False)
        suspension_status = f"ok:{len(suspensions)}"
    except Exception as exc:
        suspension_status = f"unavailable:{type(exc).__name__}"
    try:
        squad_status = refresh_current_squads(session, force=force_squads)
    except Exception as exc:
        squad_status = f"unavailable:{type(exc).__name__}"
    return {
        "injury_status": injury_status,
        "injury_rows": _read_row_count(INJURY_CSV),
        "suspension_status": suspension_status,
        "suspension_rows": _read_row_count(SUSPENSION_CSV),
        "current_squad_status": squad_status,
        "current_squad_rows": _read_row_count(LIVE_SQUADS),
        "injury_updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def ensure_manual_template() -> None:
    MANUAL_AVAILABILITY.parent.mkdir(parents=True, exist_ok=True)
    if not MANUAL_AVAILABILITY.exists():
        pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(MANUAL_AVAILABILITY, index=False)


def _automatic_availability() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path, absence_type in ((INJURY_CSV, "injury"), (SUSPENSION_CSV, "suspension")):
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        if "absence_type" not in frame:
            frame["absence_type"] = absence_type
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[*MANUAL_COLUMNS, "injury", "expected_return", "expected_return_date", "market_value_m", "source"]
        )

    automatic = pd.concat(frames, ignore_index=True, sort=False)
    automatic["team"] = automatic["team"].map(canonical_team)
    automatic["player_key"] = automatic["player"].map(name_key)
    automatic["player_id"] = pd.to_numeric(automatic.get("player_id"), errors="coerce").astype("Int64")
    automatic["availability"] = pd.to_numeric(automatic.get("availability"), errors="coerce").fillna(0.0)
    automatic["availability_key"] = np.where(
        automatic["player_id"].notna(),
        "id:" + automatic["player_id"].astype(str),
        automatic["team"].map(name_key) + "|" + automatic["player_key"],
    )

    combined: list[dict[str, Any]] = []
    for _, group in automatic.groupby("availability_key", sort=False):
        row = group.iloc[0].to_dict()
        for column in ("injury", "source", "absence_type", "since", "expected_return"):
            values = [
                str(value).strip() for value in group.get(column, pd.Series(dtype=str)).dropna()
                if str(value).strip()
            ]
            row[column] = " + ".join(dict.fromkeys(values))
        return_dates = group.get("expected_return_date", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
        parsed_dates = pd.to_datetime(return_dates.where(return_dates.ne("")), errors="coerce")
        row["expected_return_date"] = (
            "" if return_dates.eq("").any() or parsed_dates.notna().sum() != len(group)
            else parsed_dates.max().strftime("%Y-%m-%d")
        )
        row["availability"] = float(pd.to_numeric(group["availability"], errors="coerce").fillna(0.0).min())
        values = pd.to_numeric(group.get("market_value_m"), errors="coerce")
        row["market_value_m"] = float(values.max()) if values.notna().any() else np.nan
        combined.append(row)
    return pd.DataFrame(combined).drop(columns="availability_key", errors="ignore")


def load_availability() -> pd.DataFrame:
    ensure_manual_template()
    automatic = _automatic_availability()
    if "player_key" not in automatic:
        automatic["player_key"] = automatic["player"].map(name_key)

    manual = pd.read_csv(MANUAL_AVAILABILITY)
    if not manual.empty:
        for column in MANUAL_COLUMNS:
            if column not in manual:
                manual[column] = np.nan
        manual["team"] = manual["team"].map(canonical_team)
        manual["player_key"] = manual["player"].map(name_key)
        manual["availability"] = pd.to_numeric(manual["availability"], errors="coerce")
        status_availability = {
            "out": 0.0,
            "geblesseerd": 0.0,
            "geschorst": 0.0,
            "doubtful": 0.5,
            "twijfel": 0.5,
            "fit": 1.0,
            "available": 1.0,
            "beschikbaar": 1.0,
        }
        manual["availability"] = manual["availability"].fillna(
            manual["status"].astype(str).str.strip().str.lower().map(status_availability)
        ).fillna(0.0).clip(0.0, 1.0)
        manual["source"] = "handmatig"
        manual["injury"] = manual["status"]
        manual["expected_return"] = ""
        manual["expected_return_date"] = pd.to_datetime(manual["available_from"], errors="coerce").dt.strftime("%Y-%m-%d")
        manual["market_value_m"] = np.nan
        manual["player_id"] = pd.to_numeric(manual["player_id"], errors="coerce")
        manual["absence_type"] = "manual"

        automatic_key = automatic["team"].map(name_key) + "|" + automatic["player_key"]
        for row in manual.itertuples(index=False):
            key = f"{name_key(row.team)}|{row.player_key}"
            matches = (
                automatic["player_id"].eq(int(row.player_id))
                if pd.notna(row.player_id)
                else automatic_key.eq(key)
            )
            if matches.any():
                for column in (
                    "availability", "source", "injury", "note", "player_id",
                    "expected_return_date", "absence_type",
                ):
                    value = getattr(row, column, np.nan)
                    if pd.notna(value) and str(value).strip():
                        automatic.loc[matches, column] = value
            else:
                automatic = pd.concat([automatic, pd.DataFrame([row._asdict()])], ignore_index=True)

    automatic["availability"] = pd.to_numeric(automatic["availability"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    automatic["player_id"] = pd.to_numeric(automatic.get("player_id"), errors="coerce").astype("Int64")
    return automatic.reset_index(drop=True)


def load_current_roster() -> pd.DataFrame:
    source = LIVE_SQUADS if LIVE_SQUADS.exists() else CURRENT_SQUADS
    if not source.exists():
        return pd.DataFrame()
    roster = pd.read_csv(source)
    roster["team"] = roster["team"].map(canonical_team)
    roster["player_key"] = roster["player"].map(name_key)
    for column in (
        "market_value_m", "age", "minutes_per_app", "goals_per90", "assists_per90",
        "cards_per90", "days_since", "selected_proxy",
    ):
        if column not in roster:
            roster[column] = np.nan
    if "position_group" not in roster:
        roster["position_group"] = roster.get("position", "").map(_position_group)
    return roster


def _select_lineup(roster: pd.DataFrame) -> pd.DataFrame:
    available = roster[roster["availability"].gt(0.05)].copy()
    market_value = pd.to_numeric(available["market_value_m"], errors="coerce").fillna(0.0)
    season_minutes = pd.to_numeric(
        available.get("season_minutes", pd.Series(0.0, index=available.index)), errors="coerce"
    ).fillna(0.0)
    value_scale = market_value.groupby(available["team"]).transform("max").replace(0.0, 1.0)
    minutes_scale = season_minutes.groupby(available["team"]).transform("max").replace(0.0, 1.0)
    available["selection_priority"] = available["availability"] * (
        0.70 * season_minutes / minutes_scale + 0.30 * np.log1p(market_value) / np.log1p(value_scale)
    )
    available = available.sort_values(["selection_priority", "market_value_m"], ascending=False)
    selected_indices: list[int] = []
    for group_name, count in {"gk": 1, "def": 4, "mid": 3, "att": 3}.items():
        selected_indices.extend(available[available["position_group"].eq(group_name)].head(count).index.tolist())
    if len(selected_indices) < 11:
        selected_indices.extend(
            [index for index in available.index if index not in selected_indices][: 11 - len(selected_indices)]
        )
    return available.loc[selected_indices[:11]].copy()


def _lineup_features(roster: pd.DataFrame) -> dict[str, float]:
    selected = _select_lineup(roster)
    if selected.empty:
        return {}
    baseline = set(roster.loc[roster["selected_proxy"].eq(1), "player_id"].dropna().astype(int))
    selected_ids = set(selected["player_id"].dropna().astype(int))
    effective_values = selected["market_value_m"].fillna(0.0) * selected["availability"]
    count = float(len(selected))

    def weighted_mean(column: str) -> float:
        values = pd.to_numeric(selected[column], errors="coerce") * selected["availability"]
        return float(values.mean()) if values.notna().any() else np.nan

    return {
        "prior_lineup_available": 1.0,
        "prior_starter_count": count,
        "prior_lineup_value_sum_m": float(effective_values.sum()),
        "prior_lineup_value_avg_m": float(effective_values.mean()),
        "prior_lineup_value_top3_m": float(effective_values.nlargest(3).sum()),
        "prior_lineup_value_coverage": float(selected["market_value_m"].notna().mean()),
        "prior_lineup_age": float(pd.to_numeric(selected["age"], errors="coerce").mean()),
        "prior_lineup_gk_share": float(selected["position_group"].eq("gk").mean()),
        "prior_lineup_def_share": float(selected["position_group"].eq("def").mean()),
        "prior_lineup_mid_share": float(selected["position_group"].eq("mid").mean()),
        "prior_lineup_att_share": float(selected["position_group"].eq("att").mean()),
        "prior_lineup_stability": float(len(baseline & selected_ids) / max(len(baseline), 1)),
        "prior_players_minutes_per_app": weighted_mean("minutes_per_app"),
        "prior_players_goals_per90": weighted_mean("goals_per90"),
        "prior_players_assists_per90": weighted_mean("assists_per90"),
        "prior_players_cards_per90": weighted_mean("cards_per90"),
        "prior_players_days_since": float(pd.to_numeric(selected["days_since"], errors="coerce").mean()),
    }


def apply_current_availability(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roster = load_current_roster()
    if roster.empty:
        return frame.copy(), pd.DataFrame()
    availability = load_availability()

    availability["expected_return_date"] = pd.to_datetime(
        availability.get("expected_return_date"), errors="coerce"
    )

    dated_roster_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    def roster_for_date(match_date: pd.Timestamp) -> pd.DataFrame:
        match_date = pd.Timestamp(match_date).normalize()
        if match_date in dated_roster_cache:
            return dated_roster_cache[match_date]
        dated = roster.copy()
        dated["availability"] = 1.0
        dated["availability_source"] = ""
        for injury in availability.itertuples(index=False):
            player_id = getattr(injury, "player_id", pd.NA)
            if pd.notna(player_id):
                matches = dated["player_id"].eq(int(player_id))
            else:
                matches = dated["team"].eq(injury.team) & dated["player_key"].eq(injury.player_key)
            return_date = getattr(injury, "expected_return_date", pd.NaT)
            effective_availability = (
                1.0 if pd.notna(return_date) and match_date >= pd.Timestamp(return_date)
                else float(injury.availability)
            )
            if matches.any():
                dated.loc[matches, "availability"] = effective_availability
                dated.loc[matches, "availability_source"] = str(injury.source)
        dated_roster_cache[match_date] = dated
        return dated

    output = frame.copy()
    profile_cache: dict[tuple[pd.Timestamp, str], dict[str, float]] = {}
    for index, match in output.iterrows():
        match_date = pd.Timestamp(match["date"]).normalize()
        dated_roster = roster_for_date(match_date)
        names: set[str] = set()
        for side in ("home", "away"):
            team = canonical_team(match[f"{side}_team"])
            cache_key = (match_date, team)
            if cache_key not in profile_cache:
                profile_cache[cache_key] = _lineup_features(dated_roster[dated_roster["team"].eq(team)])
            for name, value in profile_cache[cache_key].items():
                output.at[index, f"tm_{side}_{name}"] = value
                names.add(name)
        for name in names:
            home = pd.to_numeric(pd.Series([output.at[index, f"tm_home_{name}"]]), errors="coerce").iloc[0]
            away = pd.to_numeric(pd.Series([output.at[index, f"tm_away_{name}"]]), errors="coerce").iloc[0]
            output.at[index, f"tm_diff_{name}"] = home - away

    today = pd.Timestamp.today().normalize()
    current_roster = roster_for_date(today)
    details_rows: list[dict[str, Any]] = []
    for absence in availability.itertuples(index=False):
        return_date = getattr(absence, "expected_return_date", pd.NaT)
        effective_availability = (
            1.0 if pd.notna(return_date) and today >= pd.Timestamp(return_date)
            else float(absence.availability)
        )
        if effective_availability >= 1.0:
            continue
        player_id = getattr(absence, "player_id", pd.NA)
        if pd.notna(player_id):
            matches = current_roster["player_id"].eq(int(player_id))
        else:
            matches = current_roster["team"].eq(absence.team) & current_roster["player_key"].eq(absence.player_key)
        matched = current_roster.loc[matches].head(1)
        row = absence._asdict()
        row["availability"] = effective_availability
        row["matched_roster"] = int(not matched.empty)
        if not matched.empty:
            player = matched.iloc[0]
            row["team"] = player["team"]
            row["player"] = player["player"]
            row["selected_proxy"] = float(player.get("selected_proxy", 0.0) or 0.0)
            row["market_value_m"] = (
                player["market_value_m"] if pd.notna(player.get("market_value_m"))
                else row.get("market_value_m", np.nan)
            )
        else:
            row["selected_proxy"] = 0.0
        row["estimated_value_impact_m"] = (
            float(row.get("market_value_m", 0.0) or 0.0)
            * (1.0 - effective_availability)
            * float(row["selected_proxy"])
        )
        details_rows.append(row)
    details = pd.DataFrame(details_rows)
    if details.empty:
        return output, details
    return output, details.sort_values(
        ["matched_roster", "estimated_value_impact_m", "market_value_m"], ascending=False
    ).reset_index(drop=True)


def save_injury_report(details: pd.DataFrame) -> Path:
    destination = RAW_DIR / "current_availability_report.json"
    report = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_absences": int(len(details)),
        "unavailable_players_matched": int(
            details.get("matched_roster", pd.Series(dtype=float)).fillna(0).sum()
        ),
        "unavailable_players_unmatched": int(
            len(details) - details.get("matched_roster", pd.Series(dtype=float)).fillna(0).sum()
        ),
        "likely_starters_affected": int(details.get("selected_proxy", pd.Series(dtype=float)).fillna(0).sum()),
        "teams_affected": int(details.get("team", pd.Series(dtype=str)).nunique()),
        "live_squad_players": _read_row_count(LIVE_SQUADS),
        "live_squad_teams": int(load_current_roster().get("team", pd.Series(dtype=str)).nunique()),
    }
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination
