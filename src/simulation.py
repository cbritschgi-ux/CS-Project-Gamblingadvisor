"""
Monte-Carlo-Simulation der laufenden Saison.

Die restlichen Spiele werden viele Tausend Mal «ausgewürfelt». Für jedes Spiel
liefert das ML-Modell die Wahrscheinlichkeiten für Heimsieg, Unentschieden und
Auswärtssieg. Am Ende zählen wir, wie oft jedes Team Meister wird, die obere
Hälfte erreicht oder auf Platz 11 bzw. 12 landet.

Liga-Format seit 2023/24 (12 Teams):
  - Phase 1: 33 Runden, jedes Team spielt dreimal gegen jedes andere.
  - Danach Teilung: Plätze 1–6 und 7–12 bilden zwei Gruppen. Punkte werden
    mitgenommen, jedes Team spielt noch einmal gegen jedes Team seiner Gruppe
    (5 Runden). Die obere Gruppe belegt immer die Plätze 1–6.

Vereinfachungen (im Video erwähnen!):
  - Die Wahrscheinlichkeiten werden mit dem heutigen Stand berechnet und
    während der Simulation nicht angepasst (keine Form-Updates).
  - Heimrecht der noch nicht angesetzten Spiele wird ausgeglichen geschätzt.
  - Tore werden nur grob simuliert (für die Tordifferenz bei Punktgleichheit).

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from config import GROUP_SIZE, PHASE1_ROUNDS, RANDOM_STATE, TEAMS_PER_SEASON
from src.standings import compute_table

PHASE1_MATCHES = TEAMS_PER_SEASON * PHASE1_ROUNDS // 2          # 198 Spiele
MEETINGS_PHASE1 = PHASE1_ROUNDS // (TEAMS_PER_SEASON - 1)       # 3 Duelle pro Paar


@dataclass
class SimulationResult:
    summary: pd.DataFrame       # pro Team: Wahrscheinlichkeiten und Erwartungswerte
    rank_probs: pd.DataFrame    # Team × Schlussrang: Wahrscheinlichkeit
    n_sims: int
    remaining_matches: int
    split_done: bool            # True, wenn die Teilung schon stattgefunden hat


# --------------------------------------------------------------------------
# Spielplan der Restsaison
# --------------------------------------------------------------------------
def remaining_phase1_fixtures(played: pd.DataFrame, teams: list[str]) -> list[tuple[str, str]]:
    """Alle noch fehlenden Spiele der 33 Runden.

    Jedes Paar trifft dreimal aufeinander. Fehlende Duelle bekommen das Heimrecht
    so, dass beide Teams möglichst gleich viele Heimspiele gegeneinander haben.
    """
    home_count = played.groupby(["home", "away"]).size().to_dict()
    fixtures = []
    for a, b in combinations(sorted(teams), 2):
        a_home, b_home = home_count.get((a, b), 0), home_count.get((b, a), 0)
        for _ in range(max(0, MEETINGS_PHASE1 - a_home - b_home)):
            if a_home <= b_home:
                fixtures.append((a, b))
                a_home += 1
            else:
                fixtures.append((b, a))
                b_home += 1
    return fixtures


def _group_pairs_home_first(group_ranked: list) -> list[tuple]:
    """Spiele innerhalb einer Gruppe nach der Teilung (jedes Paar einmal).
    Heimrecht abwechselnd nach Rang, damit es ausgeglichen ist."""
    pairs = []
    for i, j in combinations(range(len(group_ranked)), 2):
        home, away = (group_ranked[i], group_ranked[j]) if (i + j) % 2 == 0 else (group_ranked[j], group_ranked[i])
        pairs.append((home, away))
    return pairs


# --------------------------------------------------------------------------
# Würfeln
# --------------------------------------------------------------------------
def _sample_matches(probs: np.ndarray, rng: np.random.Generator):
    """Zieht Ausgänge und grobe Torzahlen.

    probs : Array (n_sims, 3) mit P(H), P(D), P(A)
    Rückgabe: Punkte und Tore für Heim- und Auswärtsteam, je (n_sims,)
    """
    u = rng.random(len(probs))
    home_win = u < probs[:, 0]
    draw = (~home_win) & (u < probs[:, 0] + probs[:, 1])
    away_win = ~(home_win | draw)

    # Torzahlen: nur für die Tordifferenz, darum einfache Verteilungen
    loser_goals = rng.choice([0, 1, 2], size=len(probs), p=[0.5, 0.35, 0.15])
    margin = rng.choice([1, 2, 3], size=len(probs), p=[0.55, 0.3, 0.15])
    draw_goals = rng.choice([0, 1, 2], size=len(probs), p=[0.35, 0.45, 0.2])

    home_goals = np.where(home_win, loser_goals + margin, np.where(draw, draw_goals, loser_goals))
    away_goals = np.where(away_win, loser_goals + margin, np.where(draw, draw_goals, loser_goals))
    home_pts = np.where(home_win, 3, np.where(draw, 1, 0))
    away_pts = np.where(away_win, 3, np.where(draw, 1, 0))
    return home_pts, away_pts, home_goals, away_goals


def _ranking_score(pts, gd, gf, rng) -> np.ndarray:
    """Ein einziger Wert pro Team, der die Tabellenregeln abbildet:
    Punkte vor Tordifferenz vor Toren; bei Gleichstand entscheidet der Zufall."""
    return pts * 1e6 + (gd + 500) * 1e3 + gf + rng.random(pts.shape) * 0.5


def simulate_season(played_season: pd.DataFrame, prob_matrix: np.ndarray, teams: list[str],
                    n_sims: int = 5000, seed: int = RANDOM_STATE) -> SimulationResult:
    """Simuliert die Restsaison n_sims-mal.

    played_season : bereits gespielte Spiele der laufenden Saison
    prob_matrix   : Array (12, 12, 3); prob_matrix[h, a] = P(H, D, A) für Heimteam h gegen a
    teams         : Teamnamen in der Reihenfolge der prob_matrix
    """
    if len(teams) != TEAMS_PER_SEASON:
        raise ValueError(f"Die Simulation ist für {TEAMS_PER_SEASON} Teams gebaut, "
                         f"die laufende Saison hat {len(teams)}.")
    rng = np.random.default_rng(seed)
    idx = {team: i for i, team in enumerate(teams)}
    played_season = played_season.sort_values("date", kind="stable")

    # Ausgangslage: bisherige Punkte, Tore, Gegentore
    table = compute_table(played_season, teams).set_index("Team").loc[teams]
    pts = np.tile(table["Pkt"].to_numpy(float), (n_sims, 1))
    gf = np.tile(table["Tore"].to_numpy(float), (n_sims, 1))
    ga = np.tile(table["Gegentore"].to_numpy(float), (n_sims, 1))

    def play(home_idx: np.ndarray, away_idx: np.ndarray) -> None:
        """Simuliert ein Spiel in allen Simulationen gleichzeitig (Indizes je Simulation)."""
        probs = prob_matrix[home_idx, away_idx]                      # (n_sims, 3)
        hp, ap, hg, ag = _sample_matches(probs, rng)
        rows = np.arange(n_sims)
        pts[rows, home_idx] += hp
        pts[rows, away_idx] += ap
        gf[rows, home_idx] += hg
        ga[rows, home_idx] += ag
        gf[rows, away_idx] += ag
        ga[rows, away_idx] += hg

    split_done = len(played_season) >= PHASE1_MATCHES
    remaining = 0

    if not split_done:
        # Phase 1 zu Ende spielen
        fixtures = remaining_phase1_fixtures(played_season, teams)
        remaining += len(fixtures)
        for home, away in fixtures:
            play(np.full(n_sims, idx[home]), np.full(n_sims, idx[away]))
        # Teilung hängt vom simulierten Tabellenstand ab -> pro Simulation verschieden
        order = np.argsort(-_ranking_score(pts, gf - ga, gf, rng), axis=1)   # (n_sims, 12)
        phase2_pairs = [(slice_start, i, j)
                        for slice_start in (0, GROUP_SIZE)
                        for i, j in combinations(range(GROUP_SIZE), 2)]
        for start, i, j in phase2_pairs:
            a, b = order[:, start + i], order[:, start + j]
            home_idx = np.where((i + j) % 2 == 0, a, b)
            away_idx = np.where((i + j) % 2 == 0, b, a)
            play(home_idx, away_idx)
        remaining += len(phase2_pairs)
        top_group = np.zeros((n_sims, len(teams)), dtype=bool)
        np.put_along_axis(top_group, order[:, :GROUP_SIZE], True, axis=1)
    else:
        # Teilung ist bekannt: Tabelle nach den ersten 198 Spielen
        phase1 = played_season.iloc[:PHASE1_MATCHES]
        phase2_played = played_season.iloc[PHASE1_MATCHES:]
        split_table = compute_table(phase1, teams)
        groups = [list(split_table["Team"].iloc[:GROUP_SIZE]), list(split_table["Team"].iloc[GROUP_SIZE:])]
        already = {frozenset(p) for p in zip(phase2_played["home"], phase2_played["away"])}
        for group in groups:
            for home, away in _group_pairs_home_first(group):
                if frozenset((home, away)) in already:
                    continue
                remaining += 1
                play(np.full(n_sims, idx[home]), np.full(n_sims, idx[away]))
        top_group = np.zeros((n_sims, len(teams)), dtype=bool)
        top_group[:, [idx[t] for t in groups[0]]] = True

    # Schlussrangliste: obere Gruppe immer vor der unteren
    score = _ranking_score(pts, gf - ga, gf, rng) + top_group * 1e12
    final_order = np.argsort(-score, axis=1)
    ranks = np.empty_like(final_order)
    np.put_along_axis(ranks, final_order, np.arange(1, len(teams) + 1)[None, :].repeat(n_sims, 0), axis=1)

    rank_probs = pd.DataFrame(
        [[np.mean(ranks[:, t] == r) for r in range(1, len(teams) + 1)] for t in range(len(teams))],
        index=teams, columns=range(1, len(teams) + 1),
    )
    summary = pd.DataFrame({
        "Team": teams,
        "Aktuelle Punkte": table["Pkt"].to_numpy(),
        "Erwartete Punkte": pts.mean(axis=0),
        "Ø Schlussrang": ranks.mean(axis=0),
        "Meister": rank_probs[1].to_numpy(),
        "Obere Gruppe (Top 6)": top_group.mean(axis=0),
        "Platz 11": rank_probs[11].to_numpy(),
        "Platz 12": rank_probs[12].to_numpy(),
    }).sort_values(["Erwartete Punkte"], ascending=False).reset_index(drop=True)

    return SimulationResult(summary=summary, rank_probs=rank_probs.loc[summary["Team"]],
                            n_sims=n_sims, remaining_matches=remaining, split_done=split_done)


def build_prob_matrix(predict_fn, teams: list[str]) -> np.ndarray:
    """Erstellt die 12×12×3-Matrix aller Paarungen mit einer Prognosefunktion.

    predict_fn erhält ein DataFrame mit Spalten home/away und gibt
    p_home, p_draw, p_away zurück.
    """
    pairs = pd.DataFrame([(h, a) for h in teams for a in teams if h != a], columns=["home", "away"])
    probs = predict_fn(pairs)
    matrix = np.zeros((len(teams), len(teams), 3))
    idx = {t: i for i, t in enumerate(teams)}
    for row, (_, p) in zip(pairs.itertuples(index=False), probs.iterrows()):
        matrix[idx[row.home], idx[row.away]] = [p["p_home"], p["p_draw"], p["p_away"]]
    return matrix
