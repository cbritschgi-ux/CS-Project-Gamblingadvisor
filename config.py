"""
Zentrale Einstellungen der Super-League-Prognose-App.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Pfade: Alle Daten liegen im Ordner «data» neben dieser Datei.
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "raw" / "SWZ.csv"          # heruntergeladene Original-CSV
DB_PATH = DATA_DIR / "superleague.db"           # SQLite-Datenbank
MODEL_PATH = DATA_DIR / "modell.joblib"         # trainiertes ML-Modell

# --------------------------------------------------------------------------
# Datenquellen
# --------------------------------------------------------------------------
# Historische Resultate inkl. Wettquoten (eine CSV mit allen Saisons).
CSV_URL = "https://www.football-data.co.uk/new/SWZ.csv"

# API-Football (optional) für kommende Spiele. Gratis-Plan: 100 Anfragen/Tag.
# Die Liga-ID der Super League bitte im API-Dashboard prüfen (/leagues?country=Switzerland).
API_BASE_URL = "https://v3.football.api-sports.io"
API_LEAGUE_ID = 207
API_CACHE_HOURS = 12        # so lange werden API-Antworten in der DB zwischengespeichert

# --------------------------------------------------------------------------
# Elo-Rating (eigene Implementierung in src/elo.py)
# --------------------------------------------------------------------------
ELO_START = 1500            # Startwert aller Teams in der ersten Saison
ELO_K = 20                  # Lernrate: wie stark ein einzelnes Spiel das Rating verändert
ELO_HOME_ADVANTAGE = 60     # Heimvorteil in Elo-Punkten
ELO_PROMOTED_OFFSET = -100  # Aufsteiger starten 100 Punkte unter dem Ligaschnitt
ELO_SEASON_REGRESSION = 0.2 # zu Saisonbeginn rücken alle Ratings 20 % zum Mittel

# --------------------------------------------------------------------------
# Merkmale (Feature Engineering in src/features.py)
# --------------------------------------------------------------------------
FORM_WINDOW = 5             # Form = Durchschnitt der letzten 5 Spiele
H2H_WINDOW = 3              # Direktvergleich = letzte 3 Duelle
NEUTRAL_PPG = 1.35          # neutraler Wert für Punkte pro Spiel (ohne Daten)
HOME_PRIOR_PPG = 1.6        # typische Punkte pro Heimspiel (Startwert)
AWAY_PRIOR_PPG = 1.1        # typische Punkte pro Auswärtsspiel (Startwert)
SHRINK_GAMES = 3            # «Glättung»: so viele fiktive Spiele mit Startwert

# --------------------------------------------------------------------------
# Liga-Format seit 2023/24: 12 Teams, 33 Runden, danach Teilung in zwei Gruppen
# --------------------------------------------------------------------------
TEAMS_PER_SEASON = 12
PHASE1_ROUNDS = 33
GROUP_SIZE = 6
# Anzahl Spiele einer vollständigen Saison je nach Anzahl Teams
# (10 Teams: 4 × Hin-/Rückrunde = 180; 12 Teams: 33 Runden + 5 Runden nach Teilung = 228)
SEASON_LENGTH = {10: 180, 12: 228}

RANDOM_STATE = 42
