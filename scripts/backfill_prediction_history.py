from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction.config import PREDICTION_HISTORY


DASHBOARD_PATH = "dashboard/public/data/dashboard.json"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def committed_dashboard_snapshots() -> list[tuple[pd.Timestamp, dict[str, Any]]]:
    lines = _git(
        "log",
        "--reverse",
        "--format=%H|%cI",
        "--",
        DASHBOARD_PATH,
    ).splitlines()
    snapshots: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for line in lines:
        commit, committed_at = line.split("|", maxsplit=1)
        raw = _git("show", f"{commit}:{DASHBOARD_PATH}")
        snapshots.append((pd.Timestamp(committed_at).tz_convert("UTC"), json.loads(raw)))
    return snapshots


def collect_published_predictions(
    snapshots: list[tuple[pd.Timestamp, dict[str, Any]]],
) -> pd.DataFrame:
    latest: dict[str, dict[str, Any]] = {}
    for snapshot_at, dashboard in snapshots:
        for row in dashboard.get("upcoming", []):
            match_key = str(row.get("match_key") or "").strip()
            if not match_key:
                continue
            kickoff = pd.to_datetime(row.get("kickoff_utc"), utc=True, errors="coerce")
            if pd.isna(kickoff):
                kickoff = pd.to_datetime(row.get("date"), utc=True, errors="coerce") + pd.Timedelta(days=1)
            if pd.isna(kickoff) or kickoff <= snapshot_at:
                continue
            saved = dict(row)
            saved["snapshot_at_utc"] = snapshot_at.isoformat()
            latest[match_key] = saved
    return pd.DataFrame(latest.values())


def main() -> None:
    snapshots = committed_dashboard_snapshots()
    published = collect_published_predictions(snapshots)
    frames = [published]
    if PREDICTION_HISTORY.exists():
        try:
            frames.append(pd.read_csv(PREDICTION_HISTORY))
        except pd.errors.EmptyDataError:
            pass
    combined = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in frames if not frame.empty],
        ignore_index=True,
        sort=False,
    )
    combined["snapshot_at_utc"] = pd.to_datetime(
        combined["snapshot_at_utc"], utc=True, errors="coerce"
    )
    combined = combined.sort_values("snapshot_at_utc").drop_duplicates("match_key", keep="last")
    combined["snapshot_at_utc"] = combined["snapshot_at_utc"].map(
        lambda value: value.isoformat() if pd.notna(value) else ""
    )
    PREDICTION_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    temporary = PREDICTION_HISTORY.with_suffix(".csv.tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(PREDICTION_HISTORY)
    print(
        json.dumps(
            {
                "snapshots": len(snapshots),
                "published_predictions": len(published),
                "history_rows": len(combined),
                "output": str(PREDICTION_HISTORY),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
