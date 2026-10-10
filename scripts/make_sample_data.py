"""
Erzeugt ERFUNDENE Beispieldaten im Format von football-data.co.uk.

Nur zum Testen der App, wenn die echten Daten (noch) nicht verfügbar sind!
Die Teams sind fiktiv, die Resultate zufällig. Nicht für die Präsentation
oder das Video verwenden – die App zeigt bei Beispieldaten einen Warnhinweis.

Aufruf:  python scripts/make_sample_data.py

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR  # noqa: E402

SAMPLE_CSV = DATA_DIR / "raw" / "beispieldaten.csv"

# 15 erfundene Vereine
TEAMS = ["FC Säntis", "FC Rheintal", "FC Bodensee", "BSC Aare", "FC Jura", "FC Limmat",
         "FC Reuss", "FC Ticino", "FC Léman", "FC Pilatus", "FC Emme", "FC Thur",
         "FC Rigi", "FC Saane", "FC Glarus"]


def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Kreis-Methode: jedes Team spielt einmal gegen jedes andere (Runden-Liste)."""
    teams = list(teams)
    rounds = []
    for r in range(len(teams) - 1):
        pairs = []
        for i in range(len(teams) // 2):
            a, b = teams[i], teams[-1 - i]
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds


def outcome_probs(lam_h: float, lam_a: float) -> tuple[float, float, float]:
    """P(Heimsieg, Unentschieden, Auswärtssieg) bei Poisson-verteilten Toren."""
    from math import exp, factorial
    ph = [exp(-lam_h) * lam_h ** k / factorial(k) for k in range(11)]
    pa = [exp(-lam_a) * lam_a ** k / factorial(k) for k in range(11)]
    m = np.outer(ph, pa)
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def match_dates(start: pd.Timestamp, n_rounds: int) -> list[pd.Timestamp]:
    """Samstage ab Saisonstart, mit Winterpause (20.12. bis Ende Januar)."""
    dates, d = [], start
    while len(dates) < n_rounds:
        if not ((d.month == 12 and d.day > 20) or d.month == 1):
            dates.append(d)
        d += pd.Timedelta(days=7)
    return dates


def simulate(rng: np.random.Generator) -> pd.DataFrame:
    strength = {t: rng.normal(0, 0.3) for t in TEAMS}
    league = TEAMS[:10]
    rows = []
    for year in range(2013, 2027):
        season = f"{year}/{year + 1}"
        n_teams = 10 if year < 2023 else 12
        if year == 2023:                               # Aufstockung auf 12 Teams
            league = league + [t for t in TEAMS if t not in league][:2]
        for t in strength:                             # Stärke ändert sich über den Sommer
            strength[t] = 0.75 * strength[t] + rng.normal(0, 0.15)

        def play(home, away, date):
            lam_h = np.exp(0.25 + strength[home] - strength[away] * 0.8 + 0.1)
            lam_a = np.exp(0.05 + strength[away] - strength[home] * 0.8)
            hg, ag = rng.poisson(lam_h), rng.poisson(lam_a)
            p = outcome_probs(lam_h, lam_a)
            odds = [round(1 / (pi * 1.06), 2) for pi in p]
            rows.append({"Country": "Switzerland", "League": "Super League", "Season": season,
                         "Date": date.strftime("%d/%m/%Y"), "Time": "18:00",
                         "Home": home, "Away": away, "HG": hg, "AG": ag,
                         "Res": "H" if hg > ag else ("A" if hg < ag else "D"),
                         "AvgCH": odds[0], "AvgCD": odds[1], "AvgCA": odds[2]})
            return hg, ag

        rr = round_robin(rng.permutation(league).tolist())
        if n_teams == 10:
            schedule = rr + [[(b, a) for a, b in r] for r in rr]
            schedule = schedule + schedule                                  # 36 Runden
        else:
            schedule = rr + [[(b, a) for a, b in r] for r in rr] + rr       # 33 Runden
        dates = match_dates(pd.Timestamp(f"{year}-07-20"), len(schedule) + 5)
        # Laufende Saison 2026/27: nur bis Anfang Oktober gespielt
        cutoff = pd.Timestamp("2026-10-05") if year == 2026 else pd.Timestamp("2100-01-01")

        for r, pairs in enumerate(schedule):
            if dates[r] > cutoff:
                break
            for home, away in pairs:
                play(home, away, dates[r])

        if n_teams == 12 and dates[len(schedule) - 1] <= cutoff:
            # Teilung nach 33 Runden: je 5 Runden in der oberen und unteren Gruppe
            season_rows = pd.DataFrame([x for x in rows if x["Season"] == season])
            pts = {t: 0 for t in league}
            for x in season_rows.itertuples():
                pts[x.Home] += 3 if x.HG > x.AG else (1 if x.HG == x.AG else 0)
                pts[x.Away] += 3 if x.AG > x.HG else (1 if x.HG == x.AG else 0)
            ranked = sorted(league, key=lambda t: -pts[t])
            for group in (ranked[:6], ranked[6:]):
                for r, pairs in enumerate(round_robin(group)):
                    for home, away in pairs:
                        play(home, away, dates[len(schedule) + r])

        # Ab-/Aufstieg: schwächstes Team (nach Stärke) tauscht mit einem Nicht-Ligisten
        if year < 2026:
            weakest = min(league, key=lambda t: strength[t])
            outside = [t for t in TEAMS if t not in league]
            league = [t for t in league if t != weakest] + [outside[rng.integers(len(outside))]]
    return pd.DataFrame(rows)


def write_sample_csv(path: Path = SAMPLE_CSV, seed: int = 7) -> Path:
    """Erzeugt die Beispieldaten und speichert sie als CSV."""
    df = simulate(np.random.default_rng(seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = write_sample_csv()
    print(f"Erfundene Spiele gespeichert: {out}")
