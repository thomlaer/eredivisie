from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FOOTBALL_DATA_DIR = RAW_DIR / "football_data"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"
OUTPUT_DIR = ROOT / "outputs" / "latest"
MODEL_DIR = ROOT / "models"
DASHBOARD_PUBLIC = ROOT / "dashboard" / "public"

CURRENT_SEASON_START = 2026
FIRST_SEASON_START = 1993
MODEL_TRAIN_FROM = 2012
VALIDATION_SEASONS = {2023, 2024}
TEST_SEASON = 2025
CURRENT_SEASON = 2026

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/N1.csv"
FIXTURE_FEED_URL = "https://fixturedownload.azurewebsites.net/feed/json/eredivisie-2026"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/ned.1/scoreboard"
FOOTBALL_DATA_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

RANDOM_SEED = 42
RECENCY_HALF_LIFE_DAYS = 1460
RECENCY_MIN_WEIGHT = 0.15
SIMULATIONS = 10_000

# Fixed on rolling seasons through 2024/25, before the untouched 2025/26 test.
OUTCOME_ODDS_WEIGHT = 1.0
OUTCOME_DRAW_MULTIPLIER_WITH_ODDS = 1.30
OUTCOME_DRAW_MULTIPLIER_NO_ODDS = 0.75


def ensure_directories() -> None:
    for path in (
        RAW_DIR,
        FOOTBALL_DATA_DIR,
        PROCESSED_DIR,
        MANUAL_DIR,
        OUTPUT_DIR,
        MODEL_DIR,
        DASHBOARD_PUBLIC / "data",
        DASHBOARD_PUBLIC / "files",
    ):
        path.mkdir(parents=True, exist_ok=True)
