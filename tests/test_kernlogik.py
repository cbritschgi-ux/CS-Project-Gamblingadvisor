"""
Automatische Tests der Kernlogik. Ausführen mit:  python -m pytest

Getestet wird mit erfundenen Beispieldaten in einer temporären Datenbank,
die echte Datenbank bleibt unberührt.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_sample_data import write_sample_csv  # noqa: E402
from src import database as db  # noqa: E402
from src.data_loader import map_team_names, parse_csv, season_from_date  # noqa: E402
from src.elo import EloParams, expected_home_score, run_elo  # noqa: E402
from src.features import FEATURE_COLUMNS, training_table  # noqa: E402
from src.model import predict_fixtures, train_models  # noqa: E402
from src.simulation import (build_prob_matrix, remaining_phase1_fixtures,  # noqa: E402
                            simulate_season)
from src.standings import compute_table  # noqa: E402


@pytest.fixture(scope="module")
def matches(tmp_path_factory) -> pd.DataFrame:
    folder = tmp_path_factory.mktemp("daten")
    csv = write_sample_csv(folder / "beispiel.csv")
    return parse_csv(csv)


# --------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------
def test_parse_csv_basics(matches):
    assert {"season", "date", "home", "away", "result", "odds_home"} <= set(matches.columns)
    assert set(matches["result"]) <= {"H", "D", "A"}
    assert matches["date"].is_monotonic_increasing
    # Resultat passt zu den Toren
    home_wins = matches["home_goals"] > matches["away_goals"]
    assert (matches.loc[home_wins, "result"] == "H").all()


def test_season_from_date():
    assert season_from_date(pd.Timestamp("2026-10-15")) == "2026/2027"
    assert season_from_date(pd.Timestamp("2027-03-15")) == "2026/2027"


def test_database_roundtrip(matches, tmp_path):
    path = tmp_path / "test.db"
    assert db.replace_matches(matches, path) == len(matches)
    loaded = db.load_matches(path)
    assert len(loaded) == len(matches)
    db.upsert_fixtures(pd.DataFrame({"date": [pd.Timestamp("2099-01-01")], "home": ["A"], "away": ["B"]}),
                       "manuell", path)
    assert len(db.load_fixtures(db_path=path)) == 1


def test_team_name_mapping():
    known = ["St. Gallen", "Grasshoppers", "Young Boys", "Lugano"]
    mapping = map_team_names(["FC St. Gallen 1879", "Grasshopper Club Zürich", "BSC Young Boys"], known)
    assert mapping == {"FC St. Gallen 1879": "St. Gallen",
                       "Grasshopper Club Zürich": "Grasshoppers",
                       "BSC Young Boys": "Young Boys"}


# --------------------------------------------------------------------------
# Elo und Tabelle
# --------------------------------------------------------------------------
def test_elo_expected_score_symmetry():
    assert expected_home_score(1500, 1500, 0) == pytest.approx(0.5)
    assert expected_home_score(1500, 1500, 60) > 0.5


def test_elo_is_zero_sum_within_season(matches):
    first = matches[matches["season"] == matches["season"].min()]
    result = run_elo(first, EloParams())
    assert np.mean(list(result.ratings.values())) == pytest.approx(1500)


def test_table_points(matches):
    season = matches[matches["season"] == "2025/2026"]
    table = compute_table(season)
    assert table["Pkt"].sum() == 3 * (season["result"] != "D").sum() + 2 * (season["result"] == "D").sum()
    assert table["Rang"].tolist() == list(range(1, len(table) + 1))


# --------------------------------------------------------------------------
# Kein Datenleck: Merkmale eines Spiels dürfen nicht von seinem eigenen
# Resultat oder späteren Spielen abhängen.
# --------------------------------------------------------------------------
def test_no_leakage(matches):
    table, _ = training_table(matches)
    cut = len(matches) // 2
    changed = matches.copy()
    # Ab dem Spiel «cut» alle Resultate umdrehen
    later = changed.index >= cut
    changed.loc[later, ["home_goals", "away_goals"]] = changed.loc[later, ["away_goals", "home_goals"]].to_numpy()
    changed.loc[later, "result"] = changed.loc[later, "result"].map({"H": "A", "A": "H", "D": "D"})
    table_changed, _ = training_table(changed)
    # Spiele bis und mit «cut» (gleiches Datum!) müssen identische Merkmale haben
    same_day = table["date"] < table.loc[cut, "date"]
    same_day |= table.index == cut
    pd.testing.assert_frame_equal(table.loc[same_day, FEATURE_COLUMNS],
                                  table_changed.loc[same_day, FEATURE_COLUMNS])


def test_features_have_no_missing_values(matches):
    table, _ = training_table(matches)
    assert not table[FEATURE_COLUMNS].isna().any().any()


# --------------------------------------------------------------------------
# Modell und Simulation
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(matches):
    return train_models(matches, path=None)


def test_model_beats_baseline(trained):
    m = trained.metrics.set_index("Modell")
    baseline = m.loc[[n for n in m.index if n.startswith("Baseline")][0], "Log-Loss"]
    assert m.loc[trained.name, "Log-Loss"] < baseline


def test_predictions_are_probabilities(trained, matches):
    elo = run_elo(matches, EloParams(**trained.elo_params))
    teams = sorted(set(matches[matches["season"] == "2026/2027"]["home"]))
    fixtures = pd.DataFrame({"date": [pd.Timestamp("2026-10-20")] * 2,
                             "home": teams[:2], "away": teams[2:4], "season": "2026/2027"})
    out = predict_fixtures(trained, matches, fixtures, elo)
    sums = out[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(sums, 1.0)
    assert set(out["tip"]) <= {"H", "D", "A"}


def test_remaining_fixtures_complete_phase1():
    teams = [f"T{i}" for i in range(12)]
    remaining = remaining_phase1_fixtures(pd.DataFrame(columns=["home", "away"]), teams)
    assert len(remaining) == 198                     # 12 × 33 / 2
    games_per_team = pd.Series([t for pair in remaining for t in pair]).value_counts()
    assert (games_per_team == 33).all()


def test_simulation_probabilities(trained, matches):
    season = matches[matches["season"] == "2026/2027"]
    teams = sorted(set(season["home"]) | set(season["away"]))
    elo = run_elo(matches, EloParams(**trained.elo_params))
    start = season["date"].max() + pd.Timedelta(days=1)
    matrix = build_prob_matrix(
        lambda pairs: predict_fixtures(trained, matches, pairs.assign(date=start, season="2026/2027"), elo)
        [["p_home", "p_draw", "p_away"]], teams)
    result = simulate_season(season, matrix, teams, n_sims=500)
    assert result.summary["Meister"].sum() == pytest.approx(1.0)
    assert result.summary["Obere Gruppe (Top 6)"].sum() == pytest.approx(6.0)
    assert np.allclose(result.rank_probs.sum(axis=1), 1.0)
    assert np.allclose(result.rank_probs.sum(axis=0), 1.0)


def test_api_fixtures_are_parsed_and_mapped(tmp_path, monkeypatch):
    """API-Antwort wird gelesen, Teamnamen werden zugeordnet und gespeichert (ohne echten Aufruf)."""
    from src import data_loader

    tomorrow = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%dT16:00:00+00:00")
    payload = {"errors": [], "response": [
        {"fixture": {"date": tomorrow},
         "teams": {"home": {"name": "FC St. Gallen 1879"}, "away": {"name": "BSC Young Boys"}}},
    ]}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(data_loader.requests, "get", lambda *a, **k: FakeResponse())
    path = tmp_path / "api.db"
    n, unmapped = data_loader.import_api_fixtures("test-key", 2026, ["St. Gallen", "Young Boys"],
                                                  db_path=path)
    assert n == 1 and unmapped == []
    stored = db.load_fixtures(db_path=path)
    assert stored.iloc[0]["home"] == "St. Gallen" and stored.iloc[0]["away"] == "Young Boys"


def test_api_error_message(tmp_path, monkeypatch):
    from src import data_loader

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"errors": {"plan": "Free plans do not have access to this season"}, "response": []}

    monkeypatch.setattr(data_loader.requests, "get", lambda *a, **k: FakeResponse())
    with pytest.raises(data_loader.ApiError, match="Free plans"):
        data_loader.fetch_upcoming_fixtures("key", 2026, db_path=tmp_path / "x.db")
