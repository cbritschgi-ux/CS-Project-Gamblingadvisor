"""
Eigenes Elo-Rating für die Super League.

Idee: Jedes Team hat eine Stärke-Zahl (Start 1500). Nach jedem Spiel gibt der
Verlierer Punkte an den Sieger ab. Ein Sieg gegen ein starkes Team bringt mehr
als ein Sieg gegen ein schwaches. Höhere Siege zählen etwas mehr (Tordifferenz-
Faktor wie im «World Football Elo»).

Formeln:
    Erwartung E_heim = 1 / (1 + 10^((R_auswärts - R_heim - Heimvorteil) / 400))
    Ergebnis  S_heim = 1 (Sieg), 0.5 (Unentschieden), 0 (Niederlage)
    Änderung  Δ      = K × Tordifferenz-Faktor × (S_heim - E_heim)
    R_heim += Δ,  R_auswärts -= Δ

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from config import (ELO_HOME_ADVANTAGE, ELO_K, ELO_PROMOTED_OFFSET,
                    ELO_SEASON_REGRESSION, ELO_START)


@dataclass
class EloParams:
    """Alle Stellschrauben des Elo-Systems an einem Ort (in der App einstellbar)."""
    k: float = ELO_K
    home_advantage: float = ELO_HOME_ADVANTAGE
    start: float = ELO_START
    promoted_offset: float = ELO_PROMOTED_OFFSET
    season_regression: float = ELO_SEASON_REGRESSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EloResult:
    pre_match: pd.DataFrame     # Ratings VOR jedem Spiel (Index = Index der Spiele)
    ratings: dict[str, float]   # aktuelle Ratings nach dem letzten Spiel
    history: pd.DataFrame       # Verlauf: date, season, team, elo (nach jedem Spiel)


def expected_home_score(r_home: float, r_away: float, home_advantage: float) -> float:
    """Erwartete Punktausbeute des Heimteams zwischen 0 und 1."""
    return 1.0 / (1.0 + 10 ** ((r_away - r_home - home_advantage) / 400.0))


def goal_difference_factor(goal_diff: int) -> float:
    """Höhere Siege zählen mehr: 1 Tor -> 1.0, 2 Tore -> 1.5, 3+ Tore -> (11 + n) / 8."""
    goal_diff = abs(int(goal_diff))
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def run_elo(matches: pd.DataFrame, params: EloParams | None = None) -> EloResult:
    """Berechnet die Elo-Ratings Spiel für Spiel in zeitlicher Reihenfolge.

    Saisonwechsel:
      - Teams, die schon in der Vorsaison dabei waren, rücken um
        `season_regression` Richtung Ligamittel (Kaderwechsel im Sommer).
      - Aufsteiger starten `promoted_offset` Punkte unter dem Ligamittel.
    """
    params = params or EloParams()
    matches = matches.sort_values(["date"], kind="stable")
    teams_by_season = {
        season: set(group["home"]) | set(group["away"])
        for season, group in matches.groupby("season", sort=False)
    }

    ratings: dict[str, float] = {}
    pre_home, pre_away = np.empty(len(matches)), np.empty(len(matches))
    history = []
    current_season, previous_teams = None, set()

    for i, row in enumerate(matches.itertuples()):
        # Saisonwechsel erkennen und Ratings anpassen
        if row.season != current_season:
            new_teams = teams_by_season[row.season]
            if current_season is None:
                ratings = {team: params.start for team in new_teams}
            else:
                league_mean = np.mean([ratings[t] for t in previous_teams if t in ratings])
                adjusted = {}
                for team in new_teams:
                    if team in previous_teams and team in ratings:
                        adjusted[team] = league_mean + (1 - params.season_regression) * (
                            ratings[team] - league_mean)
                    else:   # Aufsteiger
                        adjusted[team] = league_mean + params.promoted_offset
                # Absteiger behalten ihr letztes Rating (falls sie später zurückkehren,
                # werden sie trotzdem wie Aufsteiger behandelt).
                ratings.update(adjusted)
            previous_teams = new_teams
            current_season = row.season

        r_home, r_away = ratings[row.home], ratings[row.away]
        pre_home[i], pre_away[i] = r_home, r_away

        expected = expected_home_score(r_home, r_away, params.home_advantage)
        actual = {"H": 1.0, "D": 0.5, "A": 0.0}[row.result]
        delta = params.k * goal_difference_factor(row.home_goals - row.away_goals) * (actual - expected)
        ratings[row.home] = r_home + delta
        ratings[row.away] = r_away - delta

        history.append((row.date, row.season, row.home, ratings[row.home]))
        history.append((row.date, row.season, row.away, ratings[row.away]))

    pre_match = pd.DataFrame({"elo_home": pre_home, "elo_away": pre_away}, index=matches.index)
    history_df = pd.DataFrame(history, columns=["date", "season", "team", "elo"])
    current_ratings = {t: ratings[t] for t in previous_teams}   # nur Teams der aktuellen Saison
    return EloResult(pre_match=pre_match, ratings=current_ratings, history=history_df)
