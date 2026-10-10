"""
Tabellenberechnung aus einer Liste von Spielen.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

import pandas as pd


def team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Macht aus jedem Spiel zwei Zeilen – eine pro Team («langes Format»).

    Spalten: match_id, date, season, team, opp, is_home, gf, ga, pts
    Dieses Format brauchen Tabelle, Form und Direktvergleich.
    """
    base = matches.reset_index().rename(columns={"index": "match_id"})
    home = pd.DataFrame({
        "match_id": base["match_id"], "date": base["date"], "season": base["season"],
        "team": base["home"], "opp": base["away"], "is_home": True,
        "gf": base["home_goals"], "ga": base["away_goals"],
    })
    away = pd.DataFrame({
        "match_id": base["match_id"], "date": base["date"], "season": base["season"],
        "team": base["away"], "opp": base["home"], "is_home": False,
        "gf": base["away_goals"], "ga": base["home_goals"],
    })
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = 0
    long.loc[long["gf"] > long["ga"], "pts"] = 3
    long.loc[long["gf"] == long["ga"], "pts"] = 1
    return long.sort_values(["date", "match_id", "is_home"], kind="stable").reset_index(drop=True)


def compute_table(matches: pd.DataFrame, teams: list[str] | None = None) -> pd.DataFrame:
    """Berechnet die Tabelle: Punkte, dann Tordifferenz, dann erzielte Tore.

    `teams` ergänzt Teams ohne Spiel (z. B. vor dem 1. Spieltag) mit 0 Punkten.
    """
    columns = ["Rang", "Team", "Sp", "S", "U", "N", "Tore", "Gegentore", "TD", "Pkt"]
    if matches.empty and not teams:
        return pd.DataFrame(columns=columns)

    long = team_rows(matches) if not matches.empty else pd.DataFrame(
        columns=["team", "gf", "ga", "pts"])
    table = long.groupby("team").agg(
        Sp=("pts", "size"),
        S=("pts", lambda p: int((p == 3).sum())),
        U=("pts", lambda p: int((p == 1).sum())),
        N=("pts", lambda p: int((p == 0).sum())),
        Tore=("gf", "sum"),
        Gegentore=("ga", "sum"),
        Pkt=("pts", "sum"),
    )
    if teams:
        table = table.reindex(sorted(set(teams) | set(table.index)), fill_value=0)
    table["TD"] = table["Tore"] - table["Gegentore"]
    table = table.sort_values(["Pkt", "TD", "Tore"], ascending=False)
    table = table.reset_index().rename(columns={"team": "Team", "index": "Team"})
    table.insert(0, "Rang", range(1, len(table) + 1))
    return table[columns].astype({c: int for c in columns if c != "Team"})
