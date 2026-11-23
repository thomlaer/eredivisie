from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from prediction.data import (
    apply_upcoming_odds_sources,
    append_espn_results,
    append_fixture_results,
    parse_espn_completed_results,
    parse_pinnacle_upcoming_odds,
)
from prediction.dashboard import _json_default
from prediction.features import build_historical_features
from prediction.injuries import (
    parse_transfermarkt_clubs,
    parse_transfermarkt_club_performance,
    parse_transfermarkt_injuries,
    parse_transfermarkt_minutes,
    parse_transfermarkt_scorers,
    parse_transfermarkt_squad,
    parse_transfermarkt_suspensions,
)
from prediction.modeling import (
    base_feature_columns,
    best_score_for_outcome,
    blend_probabilities,
    score_feature_columns,
)
from prediction.names import canonical_team
from rebuild import attach_fixture_metadata, played_prediction_metrics, snapshot_predictions
from scripts.backfill_prediction_history import collect_published_predictions


class DashboardSerializationTests(unittest.TestCase):
    def test_mixed_object_timestamp_is_json_serializable(self) -> None:
        self.assertEqual(_json_default(pd.Timestamp("2026-09-16")), "2026-09-16T00:00:00")
        self.assertEqual(_json_default(np.int64(6)), 6)
        self.assertIsNone(_json_default(pd.NaT))


class EspnResultTests(unittest.TestCase):
    def test_completed_result_is_canonicalized_for_cross_check(self) -> None:
        payload = {
            "events": [
                {
                    "date": "2026-08-30T12:30Z",
                    "status": {"type": {"completed": True}},
                    "competitions": [
                        {
                            "competitors": [
                                {
                                    "homeAway": "home",
                                    "score": "2",
                                    "team": {"displayName": "Feyenoord Rotterdam"},
                                    "statistics": [
                                        {"name": "totalShots", "displayValue": "14"},
                                        {"name": "shotsOnTarget", "displayValue": "6"},
                                        {"name": "wonCorners", "displayValue": "7"},
                                    ],
                                },
                                {
                                    "homeAway": "away",
                                    "score": "1",
                                    "team": {"displayName": "ADO Den Haag"},
                                },
                            ]
                        }
                    ],
                }
            ]
        }
        result = parse_espn_completed_results(payload)
        self.assertEqual(result.loc[0, "home_team"], "Feyenoord")
        self.assertEqual(result.loc[0, "away_team"], "ADO Den Haag")
        self.assertEqual(result.loc[0, "pair_key"], "feyenoord|ado den haag")
        self.assertEqual(result.loc[0, "home_shots"], 14)
        self.assertEqual(result.loc[0, "home_shots_target"], 6)
        self.assertEqual(result.loc[0, "home_corners"], 7)

    def test_espn_only_appends_missing_completed_fixture(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-11-21"),
                    "season_start": 2026,
                    "season": "2026/27",
                    "home_team": "NEC",
                    "away_team": "PSV",
                    "home_goals": 1,
                    "away_goals": 2,
                    "home_shots": 10.0,
                    "away_shots": 14.0,
                    "match_key": "2026-11-21|nec|psv",
                    "source_file": "N1_2627.csv",
                    "odds_source": "closing_average",
                    "has_odds": 1,
                    "score_market_has_totals": 1,
                    "score_market_has_handicap": 1,
                }
            ]
        )
        fixtures = pd.DataFrame(
            [
                {
                    "match_key": "2026-11-21|nec|psv",
                    "date": pd.Timestamp("2026-11-21"),
                    "season_start": 2026,
                    "season": "2026/27",
                    "home_team": "NEC",
                    "away_team": "PSV",
                },
                {
                    "match_key": "2026-11-22|ajax|ado den haag",
                    "date": pd.Timestamp("2026-11-22"),
                    "season_start": 2026,
                    "season": "2026/27",
                    "home_team": "Ajax",
                    "away_team": "ADO Den Haag",
                },
            ]
        )
        espn = pd.DataFrame(
            [
                {
                    "pair_key": "nec|psv",
                    "home_goals": 9,
                    "away_goals": 9,
                    "home_shots": 1,
                    "away_shots": 1,
                },
                {
                    "pair_key": "ajax|ado den haag",
                    "home_goals": 3,
                    "away_goals": 0,
                    "home_shots": 18,
                    "away_shots": 5,
                },
            ]
        )
        result = append_espn_results(matches, fixtures, espn)
        self.assertEqual(len(result), 2)
        existing = result[result["match_key"].eq("2026-11-21|nec|psv")].iloc[0]
        self.assertEqual(existing["home_goals"], 1)
        self.assertEqual(existing["source_file"], "N1_2627.csv")
        fallback = result[result["match_key"].eq("2026-11-22|ajax|ado den haag")].iloc[0]
        self.assertEqual(fallback["home_goals"], 3)
        self.assertEqual(fallback["home_shots"], 18)
        self.assertEqual(fallback["source_file"], "espn_completed_fallback")


class UpcomingOddsPriorityTests(unittest.TestCase):
    def test_consensus_overrides_pinnacle_and_manual_overrides_both(self) -> None:
        fixtures = pd.DataFrame(
            [
                {
                    "match_key": "2026-11-21|groningen|heerenveen",
                    "odds_home": np.nan,
                    "odds_source": "",
                    "has_odds": 0,
                }
            ]
        )
        pinnacle = pd.DataFrame(
            [{"match_key": fixtures.loc[0, "match_key"], "odds_home": 2.8, "odds_source": "pinnacle_live", "has_odds": 1}]
        )
        consensus = pd.DataFrame(
            [{"match_key": fixtures.loc[0, "match_key"], "odds_home": 2.4, "odds_source": "weekly_market_consensus", "has_odds": 1}]
        )
        columns = ["odds_home", "odds_source", "has_odds"]
        result = apply_upcoming_odds_sources(
            fixtures,
            columns,
            pinnacle_odds=pinnacle,
            weekly_odds=consensus,
            manual_odds=pd.DataFrame(),
        )
        self.assertEqual(result.loc[0, "odds_source"], "weekly_market_consensus")
        self.assertEqual(result.loc[0, "odds_home"], 2.4)

        manual = pd.DataFrame(
            [{"match_key": fixtures.loc[0, "match_key"], "odds_home": 2.2, "odds_source": "manual", "has_odds": 1}]
        )
        result = apply_upcoming_odds_sources(
            fixtures,
            columns,
            pinnacle_odds=pinnacle,
            weekly_odds=consensus,
            manual_odds=manual,
        )
        self.assertEqual(result.loc[0, "odds_source"], "manual")
        self.assertEqual(result.loc[0, "odds_home"], 2.2)


def match_frame(first_home_shots: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-08-10"),
                "season_start": 2024,
                "season": "2024/25",
                "home_team": "Ajax",
                "away_team": "PSV",
                "home_goals": 1,
                "away_goals": 2,
                "home_shots": first_home_shots,
                "away_shots": 8.0,
                "match_key": "2024-08-10|ajax|psv",
            },
            {
                "date": pd.Timestamp("2024-08-18"),
                "season_start": 2024,
                "season": "2024/25",
                "home_team": "Ajax",
                "away_team": "Feyenoord",
                "home_goals": 2,
                "away_goals": 0,
                "home_shots": 12.0,
                "away_shots": 6.0,
                "match_key": "2024-08-18|ajax|feyenoord",
            },
        ]
    )


class FeatureLeakageTests(unittest.TestCase):
    def test_match_stats_only_affect_later_matches(self) -> None:
        low_stats, _ = build_historical_features(match_frame(5.0))
        high_stats, _ = build_historical_features(match_frame(20.0))

        pre_match_columns = [column for column in low_stats.columns if column.startswith(("home_", "away_", "diff_"))]
        pd.testing.assert_series_equal(
            low_stats.loc[0, pre_match_columns],
            high_stats.loc[0, pre_match_columns],
            check_names=False,
        )
        self.assertEqual(low_stats.loc[1, "home_form5_shots_for"], 5.0)
        self.assertEqual(high_stats.loc[1, "home_form5_shots_for"], 20.0)

    def test_played_predictions_receive_round_without_changing_pick(self) -> None:
        predictions = pd.DataFrame(
            [{"match_key": "2026-09-12|ajax|psv", "round": np.nan, "predicted_outcome": "home_win"}]
        )
        fixtures = pd.DataFrame(
            [
                {
                    "match_key": "2026-09-12|ajax|psv",
                    "round": 6,
                    "kickoff_utc": "2026-09-12T18:00:00Z",
                    "venue": "Johan Cruijff ArenA",
                }
            ]
        )

        enriched = attach_fixture_metadata(predictions, fixtures)

        self.assertEqual(enriched.loc[0, "round"], 6)
        self.assertEqual(enriched.loc[0, "venue"], "Johan Cruijff ArenA")
        self.assertEqual(enriched.loc[0, "predicted_outcome"], "home_win")

    def test_fixture_feed_only_appends_missing_completed_results(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-09-05"),
                    "season_start": 2026,
                    "season": "2026/27",
                    "home_team": "Ajax",
                    "away_team": "PSV",
                    "home_goals": 1,
                    "away_goals": 3,
                    "match_key": "2026-09-05|ajax|psv",
                    "source_file": "N1_2627.csv",
                    "has_odds": 1,
                }
            ]
        )
        fixtures = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-09-05"),
                    "season_start": 2026,
                    "season": "2026/27",
                    "home_team": "Ajax",
                    "away_team": "PSV",
                    "match_key": "2026-09-05|ajax|psv",
                    "feed_home_goals": 1,
                    "feed_away_goals": 3,
                },
                {
                    "date": pd.Timestamp("2026-09-08"),
                    "season_start": 2026,
                    "season": "2026/27",
                    "home_team": "NEC",
                    "away_team": "Excelsior",
                    "match_key": "2026-09-08|nec|excelsior",
                    "feed_home_goals": 2,
                    "feed_away_goals": 2,
                },
            ]
        )

        output = append_fixture_results(matches, fixtures)

        self.assertEqual(len(output), 2)
        ajax = output[output["match_key"].eq("2026-09-05|ajax|psv")].iloc[0]
        nec = output[output["match_key"].eq("2026-09-08|nec|excelsior")].iloc[0]
        self.assertEqual(ajax["source_file"], "N1_2627.csv")
        self.assertEqual(ajax["has_odds"], 1)
        self.assertEqual(nec["source_file"], "fixture_feed_current")
        self.assertEqual(nec["home_goals"], 2)

    def test_prediction_history_keeps_latest_pre_match_snapshot(self) -> None:
        first = pd.DataFrame(
            [
                {
                    "match_key": "2026-11-21|ajax|psv",
                    "date": "2026-11-21",
                    "kickoff_utc": "2026-11-21T19:00:00Z",
                    "home_team": "Ajax",
                    "away_team": "PSV",
                    "predicted_outcome": "home_win",
                }
            ]
        )
        second = first.copy()
        second.loc[0, "predicted_outcome"] = "away_win"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            snapshot_predictions(first, path, "2026-11-18T08:00:00Z")
            history = snapshot_predictions(second, path, "2026-11-20T08:00:00Z")

        self.assertEqual(len(history), 1)
        self.assertEqual(history.loc[0, "predicted_outcome"], "away_win")
        self.assertEqual(history.loc[0, "snapshot_at_utc"], "2026-11-20T08:00:00+00:00")

    def test_git_backfill_uses_last_published_pick_before_kickoff(self) -> None:
        row = {
            "match_key": "2026-11-21|ajax|psv",
            "date": "2026-11-21",
            "kickoff_utc": "2026-11-21T19:00:00Z",
            "predicted_outcome": "home_win",
        }
        changed = dict(row, predicted_outcome="away_win")
        too_late = dict(row, predicted_outcome="draw")
        snapshots = [
            (pd.Timestamp("2026-11-18T08:00:00Z"), {"upcoming": [row]}),
            (pd.Timestamp("2026-11-20T08:00:00Z"), {"upcoming": [changed]}),
            (pd.Timestamp("2026-11-21T20:00:00Z"), {"upcoming": [too_late]}),
        ]

        history = collect_published_predictions(snapshots)

        self.assertEqual(len(history), 1)
        self.assertEqual(history.loc[0, "predicted_outcome"], "away_win")

    def test_archived_accuracy_uses_saved_probabilities(self) -> None:
        played = pd.DataFrame(
            [
                {
                    "actual_outcome": "away_win",
                    "prob_away_win": 0.6,
                    "prob_draw": 0.2,
                    "prob_home_win": 0.2,
                },
                {
                    "actual_outcome": "draw",
                    "prob_away_win": 0.2,
                    "prob_draw": 0.3,
                    "prob_home_win": 0.5,
                },
            ]
        )

        metrics = played_prediction_metrics(played)

        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)

    def test_archived_and_reconstructed_dates_can_be_normalized_together(self) -> None:
        combined = pd.DataFrame(
            {
                "date": ["2026-09-12", pd.Timestamp("2026-09-19")],
                "forecast_status": ["archived", "reconstructed"],
            }
        )

        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        ordered = combined.sort_values("date", ascending=False)

        self.assertEqual(ordered.iloc[0]["forecast_status"], "reconstructed")


class UpcomingOddsTests(unittest.TestCase):
    def test_pinnacle_main_moneyline_is_converted_and_canonicalized(self) -> None:
        matchups = [
            {
                "id": 123,
                "type": "matchup",
                "parentId": None,
                "startTime": "2026-11-21T15:30:00Z",
                "participants": [
                    {"alignment": "home", "name": "NEC Nijmegen"},
                    {"alignment": "away", "name": "Cambuur Leeuwaarden"},
                ],
            },
            {
                "id": 124,
                "type": "special",
                "parentId": None,
                "startTime": "2026-11-21T15:30:00Z",
                "participants": [],
            },
        ]
        markets = [
            {
                "matchupId": 123,
                "type": "moneyline",
                "period": 0,
                "isAlternate": False,
                "status": "open",
                "prices": [
                    {"designation": "home", "price": 150},
                    {"designation": "draw", "price": 250},
                    {"designation": "away", "price": -125},
                ],
            },
            {
                "matchupId": 123,
                "type": "moneyline",
                "period": 1,
                "isAlternate": False,
                "status": "open",
                "prices": [],
            },
        ]

        parsed = parse_pinnacle_upcoming_odds(matchups, markets)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed.loc[0, "home_team"], "NEC")
        self.assertEqual(parsed.loc[0, "away_team"], "SC Cambuur")
        self.assertAlmostEqual(parsed.loc[0, "odds_home"], 2.5)
        self.assertAlmostEqual(parsed.loc[0, "odds_draw"], 3.5)
        self.assertAlmostEqual(parsed.loc[0, "odds_away"], 1.8)
        self.assertEqual(parsed.loc[0, "odds_source"], "pinnacle_live")
        self.assertEqual(parsed.loc[0, "match_key"], "2026-11-21|nec|sc cambuur")
        self.assertAlmostEqual(
            parsed.loc[0, ["market_prob_home", "market_prob_draw", "market_prob_away"]].sum(),
            1.0,
        )


class ProbabilityLayerTests(unittest.TestCase):
    def test_odds_and_no_odds_use_separate_draw_settings(self) -> None:
        base = np.asarray([[0.25, 0.25, 0.50], [0.25, 0.25, 0.50]], dtype=float)
        frame = pd.DataFrame(
            {
                "market_prob_away": [0.20, np.nan],
                "market_prob_draw": [0.30, np.nan],
                "market_prob_home": [0.50, np.nan],
            }
        )
        result = blend_probabilities(
            base,
            frame,
            odds_weight=0.5,
            draw_multiplier=0.5,
            draw_multiplier_no_odds=2.0,
        )
        self.assertLess(result[0, 1], 0.20)
        self.assertGreater(result[1, 1], 0.35)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_scoreline_is_consistent_with_selected_outcome(self) -> None:
        home, away, _ = best_score_for_outcome(1.35, 1.10, "home_win")
        self.assertGreater(home, away)
        home, away, _ = best_score_for_outcome(1.35, 1.10, "draw")
        self.assertEqual(home, away)
        home, away, _ = best_score_for_outcome(1.35, 1.10, "away_win")
        self.assertLess(home, away)


class ModelFeatureTests(unittest.TestCase):
    def test_club_aliases_are_canonical(self) -> None:
        self.assertEqual(canonical_team("RKC Waalwijk"), "Waalwijk")
        self.assertEqual(canonical_team("FC Volendam"), "Volendam")
        self.assertEqual(canonical_team("AFC Ajax Amsterdam"), "Ajax")
        self.assertEqual(canonical_team("Sparta"), "Sparta Rotterdam")
        self.assertEqual(canonical_team("Feyenoord Rotterdam"), "Feyenoord")

    def test_outcome_and_score_models_use_separate_odds_features(self) -> None:
        frame = pd.DataFrame(
            {
                "elo_diff": [20.0, -10.0],
                "market_prob_home": [0.55, 0.35],
                "market_prob_draw": [0.25, 0.30],
                "market_prob_away": [0.20, 0.35],
                "tm_diff_prior_lineup_value_sum_m": [10.0, -5.0],
                "tm_home_manager_matches": [12.0, 4.0],
            }
        )
        outcome_numeric, outcome_categorical = base_feature_columns(frame)
        score_numeric, score_categorical = score_feature_columns(frame)

        self.assertIn("elo_diff", outcome_numeric)
        self.assertIn("tm_diff_prior_lineup_value_sum_m", outcome_numeric)
        self.assertNotIn("tm_home_manager_matches", outcome_numeric)
        self.assertNotIn("market_prob_home", outcome_numeric)
        self.assertIn("market_prob_home", score_numeric)
        self.assertEqual(outcome_categorical, [])
        self.assertEqual(score_categorical, [])


class InjurySourceTests(unittest.TestCase):
    def test_transfermarkt_injury_row_is_parsed_with_stable_ids(self) -> None:
        html = b"""
        <table class="items"><tbody><tr>
          <td><a title="Test Speler" href="/test/profil/spieler/123">Test Speler</a>Centrumspits</td>
          <td><a title="PSV Eindhoven" href="/psv/startseite/verein/383"></a></td>
          <td>24</td><td></td><td>Knieblessure</td><td>1 sep. 2026</td><td>1 okt. 2026</td><td>5,00 mln. EUR</td>
        </tr></tbody></table>
        """
        parsed = parse_transfermarkt_injuries(html)

        self.assertEqual(parsed.loc[0, "player_id"], 123)
        self.assertEqual(parsed.loc[0, "team"], "PSV")
        self.assertEqual(parsed.loc[0, "expected_return_date"], "2026-10-01")
        self.assertEqual(parsed.loc[0, "market_value_m"], 5.0)

    def test_transfermarkt_suspension_row_is_parsed(self) -> None:
        html = b"""
        <table class="items"><tbody><tr>
          <td><table><tr><td class="hauptlink"><a title="Test Speler" href="/test/profil/spieler/123">Test Speler</a></td></tr><tr><td>Centrumspits</td></tr></table></td>
          <td><a title="Fortuna Sittard" href="/fortuna/startseite/verein/385"></a></td>
          <td>24</td><td>Geschorst door een rode kaart</td><td>24 aug. 2026</td><td>7 sep. 2026</td><td>2</td>
        </tr></tbody></table>
        """
        parsed = parse_transfermarkt_suspensions(html)

        self.assertEqual(parsed.loc[0, "player_id"], 123)
        self.assertEqual(parsed.loc[0, "team"], "Fortuna Sittard")
        self.assertEqual(parsed.loc[0, "absence_type"], "suspension")
        self.assertEqual(parsed.loc[0, "expected_return_date"], "2026-09-07")

    def test_transfermarkt_current_club_and_squad_are_parsed(self) -> None:
        club_rows = "".join(
            f'<tr><td><a title="Club {index}" href="/club-{index}/startseite/verein/{1000 + index}/saison_id/2026"></a></td>'
            f'<td><a title="Club {index}" href="/club-{index}/kader/verein/{1000 + index}/saison_id/2026">25</a></td></tr>'
            for index in range(18)
        )
        clubs = parse_transfermarkt_clubs(f'<table class="items"><tbody>{club_rows}</tbody></table>'.encode())
        self.assertEqual(len(clubs), 18)
        self.assertTrue(clubs.loc[0, "squad_url"].endswith("/plus/1"))

        player_rows = "".join(
            f'<tr><td>{index}</td><td><table><tr><td class="hauptlink"><a href="/speler-{index}/profil/spieler/{2000 + index}">Speler {index}</a></td></tr><tr><td>{"Keeper" if index == 0 else "Centrumspits"}</td></tr></table></td>'
            f'<td>1 jan. 2000 (26)</td><td></td><td></td><td></td><td></td><td></td><td></td><td><a href="/speler-{index}/marktwertverlauf/spieler/{2000 + index}">1,00 mln. EUR</a></td></tr>'
            for index in range(11)
        )
        squad = parse_transfermarkt_squad(
            f'<table class="items"><tbody>{player_rows}</tbody></table>'.encode(), "Ajax", 610
        )
        self.assertEqual(len(squad), 11)
        self.assertEqual(squad.loc[0, "position_group"], "gk")
        self.assertEqual(squad.loc[0, "market_value_m"], 1.0)

    def test_transfermarkt_current_performance_is_parsed(self) -> None:
        player_cell = """
        <td><table><tr><td class="hauptlink"><a title="Test Speler" href="/test/profil/spieler/123">Test Speler</a></td></tr><tr><td>Centrumspits</td></tr></table></td>
        <td></td><td>1 jan. 2000 (26)</td>
        <td><a title="PSV" href="/psv/startseite/verein/383"></a></td>
        """
        minutes_html = f"""
        <table class="items"><tbody><tr><td>1</td>{player_cell}<td>4</td><td>-</td><td>Geen minuut gemist</td><td>360</td><td>3</td></tr></tbody></table>
        """.encode()
        scorer_html = f"""
        <table class="items"><tbody><tr><td>1</td><td><table><tr><td class="hauptlink"><a title="Test Speler" href="/test/profil/spieler/123">Test Speler</a></td></tr><tr><td>Centrumspits</td></tr></table></td><td><a title="PSV" href="/psv/startseite/verein/383"></a></td><td></td><td>26</td><td>4</td><td>0</td><td>1</td><td>3</td><td>2</td><td>5</td></tr></tbody></table>
        """.encode()

        minutes = parse_transfermarkt_minutes(minutes_html)
        scorers = parse_transfermarkt_scorers(scorer_html)
        self.assertEqual(minutes.loc[0, "season_minutes"], 360)
        self.assertEqual(minutes.loc[0, "season_goals"], 3)
        self.assertEqual(scorers.loc[0, "season_assists"], 2)

    def test_transfermarkt_club_performance_is_parsed(self) -> None:
        html = b"""
        <table class="items"><tbody><tr>
          <td>9</td><td><a title="Test Speler" href="/test/profil/spieler/123">Test Speler</a></td>
          <td>25</td><td></td><td>5</td><td>4</td><td>3</td><td>2</td>
          <td>1</td><td>0</td><td>0</td><td>1</td><td>2</td><td>2,00</td><td>315'</td>
        </tr></tbody></table>
        """
        parsed = parse_transfermarkt_club_performance(html, "PSV", 383)
        self.assertEqual(parsed.loc[0, "player_id"], 123)
        self.assertEqual(parsed.loc[0, "season_appearances"], 4)
        self.assertEqual(parsed.loc[0, "season_minutes"], 315)
        self.assertEqual(parsed.loc[0, "season_goals"], 3)
        self.assertEqual(parsed.loc[0, "season_assists"], 2)
        self.assertEqual(parsed.loc[0, "season_yellow_cards"], 1)
        self.assertEqual(parsed.loc[0, "season_second_yellow"], 0)
        self.assertEqual(parsed.loc[0, "season_red_cards"], 0)


if __name__ == "__main__":
    unittest.main()
