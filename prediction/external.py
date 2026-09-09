from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRANSFERMARKT_CONTEXT = ROOT / "data" / "external" / "transfermarkt_eredivisie_context.csv"


def attach_transfermarkt_context(frame: pd.DataFrame) -> pd.DataFrame:
    if not TRANSFERMARKT_CONTEXT.exists() or "match_key" not in frame:
        return frame.copy()
    context = pd.read_csv(TRANSFERMARKT_CONTEXT)
    context = context.drop(
        columns=[column for column in ("date", "home_team", "away_team") if column in context],
        errors="ignore",
    ).drop_duplicates("match_key", keep="last")
    return frame.merge(context, on="match_key", how="left", validate="one_to_one")
