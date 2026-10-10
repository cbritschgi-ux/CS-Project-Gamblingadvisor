"""
Feature Engineering: Aus den Resultaten werden Merkmale für das ML-Modell.

Wichtigste Regel – kein Datenleck: Für jedes Spiel dürfen nur Informationen
verwendet werden, die VOR dem Anpfiff bekannt waren. Darum holen wir jeden
Zustand (Form, Saisonpunkte, Direktvergleich) per `merge_asof` mit
`allow_exact_matches=False`: Es zählt nur das letzte Spiel STRIKT vor dem Datum.

Derselbe Code berechnet die Merkmale für das Training (gespielte Spiele) und
für die Prognose (kommende Spiele). So können sich die beiden nicht
unbemerkt unterscheiden.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (AWAY_PRIOR_PPG, FORM_WINDOW, H2H_WINDOW, HOME_PRIOR_PPG,
                    NEUTRAL_PPG, SHRINK_GAMES)
from src.elo import EloParams, EloResult, run_elo
from src.standings import team_rows

# Reihenfolge der Merkmale, mit der das Modell trainiert wird
FEATURE_COLUMNS = [
    "elo_diff", "elo_home", "elo_away",
    "form_pts_home", "form_pts_away",
    "form_gd_home", "form_gd_away",
    "venue_ppg_home", "venue_ppg_away",
    "season_ppg_home", "season_ppg_away",
    "h2h_pts_home",
]

# Verständliche Bezeichnungen für die App
FEATURE_LABELS = {
    "elo_diff": "Elo-Differenz (Heim − Auswärts)",
    "elo_home": "Elo Heimteam",
    "elo_away": "Elo Auswärtsteam",
    "form_pts_home": "Form Heimteam (Ø Punkte, letzte 5)",
    "form_pts_away": "Form Auswärtsteam (Ø Punkte, letzte 5)",
    "form_gd_home": "Form Heimteam (Ø Tordifferenz, letzte 5)",
    "form_gd_away": "Form Auswärtsteam (Ø Tordifferenz, letzte 5)",
    "venue_ppg_home": "Heimstärke Heimteam (Punkte/Heimspiel, Saison)",
    "venue_ppg_away": "Auswärtsstärke Auswärtsteam (Punkte/Auswärtsspiel, Saison)",
    "season_ppg_home": "Saisonleistung Heimteam (Punkte/Spiel)",
    "season_ppg_away": "Saisonleistung Auswärtsteam (Punkte/Spiel)",
    "h2h_pts_home": "Direktvergleich (Ø Punkte Heimteam, letzte 3 Duelle)",
}


def _shrink(points_sum, games, prior):
    """Glättung: kleine Stichproben werden Richtung Startwert gezogen.

    Nach 1 Spiel mit 3 Punkten wäre der Schnitt 3.0 – das ist unrealistisch.
    Mit 3 fiktiven Spielen à 1.35 Punkten ergibt sich (3 + 4.05) / 4 = 1.76.
    """
    return (points_sum + prior * SHRINK_GAMES) / (games + SHRINK_GAMES)


def _team_states(played: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Berechnet für jedes Team den Zustand NACH jedem gespielten Spiel."""
    long = team_rows(played)
    long["gd"] = long["gf"] - long["ga"]
    by_team = long.groupby("team", sort=False)

    # Form über die letzten FORM_WINDOW Spiele (über Saisongrenzen hinweg)
    long["form_pts"] = by_team["pts"].transform(
        lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean())
    long["form_gd"] = by_team["gd"].transform(
        lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean())

    # Kumulierte Werte innerhalb der Saison
    by_season = long.groupby(["team", "season"], sort=False)
    long["season_pts"] = by_season["pts"].cumsum()
    long["season_n"] = by_season.cumcount() + 1

    # Kumulierte Werte nur Heim- bzw. nur Auswärtsspiele innerhalb der Saison
    by_venue = long.groupby(["team", "season", "is_home"], sort=False)
    long["venue_pts"] = by_venue["pts"].cumsum()
    long["venue_n"] = by_venue.cumcount() + 1

    # Direktvergleich: Punkte gegen genau diesen Gegner, letzte H2H_WINDOW Duelle
    long["h2h_pts"] = long.groupby(["team", "opp"], sort=False)["pts"].transform(
        lambda s: s.rolling(H2H_WINDOW, min_periods=1).mean())

    keep = ["date", "season", "team"]
    return {
        "general": long[keep + ["form_pts", "form_gd", "season_pts", "season_n"]],
        "home": long[long["is_home"]][keep + ["venue_pts", "venue_n"]],
        "away": long[~long["is_home"]][keep + ["venue_pts", "venue_n"]],
        "h2h": long[keep + ["opp", "h2h_pts"]],
    }


def _asof(queries: pd.DataFrame, state: pd.DataFrame, team_col: str,
          by_extra: list[tuple[str, str]] | None = None, prefix: str = "") -> pd.DataFrame:
    """Holt für jede Abfragezeile den letzten Zustand des Teams STRIKT vor dem Datum."""
    left_by = [team_col] + [q for q, _ in (by_extra or [])]
    right_by = ["team"] + [s for _, s in (by_extra or [])]
    state = state.sort_values("date").rename(columns={"season": "state_season"})
    merged = pd.merge_asof(
        queries.sort_values("date"), state,
        on="date", left_by=left_by, right_by=right_by,
        direction="backward", allow_exact_matches=False,
    )
    value_cols = [c for c in state.columns if c not in ("date", "team", *right_by)]
    out = merged.set_index("_qid")[value_cols]
    return out.add_prefix(prefix)


def build_features(played: pd.DataFrame, queries: pd.DataFrame,
                   elo: EloResult, use_pre_match_elo: bool) -> pd.DataFrame:
    """Berechnet alle Merkmale für die Abfragezeilen.

    played  : gespielte Spiele (Basis aller Zustände)
    queries : Spiele, für die Merkmale gesucht sind (Spalten date, season, home, away)
    elo     : Resultat von run_elo(played)
    use_pre_match_elo : True beim Training (queries == played, Elo vor dem Spiel),
                        False bei kommenden Spielen (aktuelle Ratings)
    """
    q = queries[["date", "season", "home", "away"]].copy()
    q["date"] = pd.to_datetime(q["date"]).astype("datetime64[ns]")
    # Einheitliche Datentypen, sonst verweigert merge_asof den Abgleich
    q = q.astype({"season": str, "home": str, "away": str})
    q["_qid"] = np.arange(len(q))

    played = played.copy()
    played["date"] = pd.to_datetime(played["date"]).astype("datetime64[ns]")
    played = played.astype({"season": str, "home": str, "away": str})
    states = _team_states(played)
    for key in states:
        states[key] = states[key].astype({"date": "datetime64[ns]"})

    home_gen = _asof(q, states["general"], "home", prefix="h_")
    away_gen = _asof(q, states["general"], "away", prefix="a_")
    home_venue = _asof(q, states["home"], "home", prefix="hv_")
    away_venue = _asof(q, states["away"], "away", prefix="av_")
    h2h = _asof(q, states["h2h"], "home", by_extra=[("away", "opp")], prefix="x_")

    f = q.set_index("_qid").join([home_gen, away_gen, home_venue, away_venue, h2h])

    # Saisonwerte gelten nur, wenn der letzte Zustand aus derselben Saison stammt
    same_h = f["h_state_season"] == f["season"]
    same_a = f["a_state_season"] == f["season"]
    same_hv = f["hv_state_season"] == f["season"]
    same_av = f["av_state_season"] == f["season"]

    out = pd.DataFrame(index=f.index)
    out["form_pts_home"] = f["h_form_pts"].fillna(NEUTRAL_PPG)
    out["form_pts_away"] = f["a_form_pts"].fillna(NEUTRAL_PPG)
    out["form_gd_home"] = f["h_form_gd"].fillna(0.0)
    out["form_gd_away"] = f["a_form_gd"].fillna(0.0)
    out["season_ppg_home"] = _shrink(f["h_season_pts"].where(same_h, 0).fillna(0),
                                     f["h_season_n"].where(same_h, 0).fillna(0), NEUTRAL_PPG)
    out["season_ppg_away"] = _shrink(f["a_season_pts"].where(same_a, 0).fillna(0),
                                     f["a_season_n"].where(same_a, 0).fillna(0), NEUTRAL_PPG)
    out["venue_ppg_home"] = _shrink(f["hv_venue_pts"].where(same_hv, 0).fillna(0),
                                    f["hv_venue_n"].where(same_hv, 0).fillna(0), HOME_PRIOR_PPG)
    out["venue_ppg_away"] = _shrink(f["av_venue_pts"].where(same_av, 0).fillna(0),
                                    f["av_venue_n"].where(same_av, 0).fillna(0), AWAY_PRIOR_PPG)
    out["h2h_pts_home"] = f["x_h2h_pts"].fillna(NEUTRAL_PPG)

    # Elo: beim Training der Wert vor dem Spiel, sonst das aktuelle Rating
    if use_pre_match_elo:
        out["elo_home"] = elo.pre_match["elo_home"].to_numpy()
        out["elo_away"] = elo.pre_match["elo_away"].to_numpy()
    else:
        fallback = (np.mean(list(elo.ratings.values())) if elo.ratings else 1500.0) - 100
        out["elo_home"] = q["home"].map(elo.ratings).fillna(fallback).to_numpy()
        out["elo_away"] = q["away"].map(elo.ratings).fillna(fallback).to_numpy()
    out["elo_diff"] = out["elo_home"] - out["elo_away"]

    out = out.sort_index()
    out.index = queries.index
    return out[FEATURE_COLUMNS].astype(float)


def training_table(played: pd.DataFrame, params: EloParams | None = None) -> tuple[pd.DataFrame, EloResult]:
    """Merkmale + Zielgrösse für alle gespielten Spiele (Basis für das Training)."""
    played = played.sort_values("date", kind="stable").reset_index(drop=True)
    elo = run_elo(played, params)
    features = build_features(played, played, elo, use_pre_match_elo=True)
    table = pd.concat([played, features], axis=1)
    return table, elo


def fixture_features(played: pd.DataFrame, fixtures: pd.DataFrame, elo: EloResult) -> pd.DataFrame:
    """Merkmale für kommende Spiele (Stand: alle bisher gespielten Partien)."""
    return build_features(played, fixtures, elo, use_pre_match_elo=False)
