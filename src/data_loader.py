"""
Daten laden: historische Resultate (CSV von football-data.co.uk) und
kommende Spiele (API-Football, optional).

Ablauf:
    1. download_csv()        – lädt die CSV aus dem Internet
    2. parse_csv()           – vereinheitlicht Spaltennamen, Datum, Saison, Quoten
    3. import_csv_to_db()    – schreibt alles in die SQLite-Datenbank
    4. import_api_fixtures() – holt kommende Spiele über die API (mit Cache)

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

import difflib
import io
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from config import (API_BASE_URL, API_CACHE_HOURS, API_LEAGUE_ID, CSV_URL,
                    DB_PATH, RAW_CSV)
from src import database as db

# Mögliche Spaltennamen in den CSV-Dateien von football-data.co.uk.
# Die «neuen Ligen» (dazu gehört die Schweiz) nutzen «Home/Away/HG/AG/Res»,
# die grossen Ligen «HomeTeam/AwayTeam/FTHG/FTAG/FTR». Wir akzeptieren beides.
COLUMN_ALIASES = {
    "season": ["Season"],
    "date": ["Date"],
    "home": ["Home", "HomeTeam"],
    "away": ["Away", "AwayTeam"],
    "home_goals": ["HG", "FTHG"],
    "away_goals": ["AG", "FTAG"],
    "result": ["Res", "FTR"],
}

# Quoten in der Reihenfolge, in der wir sie bevorzugen
# (Schlussquoten-Durchschnitt, Pinnacle, Bet365, ...).
ODDS_PREFERENCE = [
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
]


class DataError(Exception):
    """Fehler beim Laden oder Verarbeiten der Daten (mit verständlicher Meldung)."""


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------
def season_from_date(date: pd.Timestamp) -> str:
    """Saison aus einem Datum ableiten. Eine Saison beginnt im Juli.

    Beispiel: 15.10.2026 -> «2026/2027», 15.03.2027 -> «2026/2027».
    """
    date = pd.Timestamp(date)
    start_year = date.year if date.month >= 7 else date.year - 1
    return f"{start_year}/{start_year + 1}"


def _normalize_season(value) -> str | None:
    """Vereinheitlicht Saison-Schreibweisen: «2012/13» oder «2012-2013» -> «2012/2013»."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.match(r"^(\d{4})\D+(\d{2,4})$", text)
    if match:
        start = int(match.group(1))
        return f"{start}/{start + 1}"
    if re.match(r"^\d{4}$", text):          # nur ein Jahr angegeben
        start = int(text)
        return f"{start}/{start + 1}"
    return None


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Gibt den ersten vorhandenen Spaltennamen aus der Kandidatenliste zurück."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


# --------------------------------------------------------------------------
# 1. CSV herunterladen
# --------------------------------------------------------------------------
def download_csv(url: str = CSV_URL, target: Path = RAW_CSV, timeout: int = 30) -> Path:
    """Lädt die CSV herunter und speichert sie unter data/raw/."""
    try:
        response = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "HSG-Studienprojekt/1.0"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataError(
            f"Download fehlgeschlagen ({exc}). Lade die Datei im Browser unter {url} "
            "herunter und lade sie auf der Seite «Daten & Einstellungen» hoch."
        ) from exc

    # Grobe Plausibilitätsprüfung: eine HTML-Fehlerseite ist keine CSV.
    head = response.content[:500].decode("utf-8", errors="ignore")
    if "Home" not in head or "<html" in head.lower():
        raise DataError("Die heruntergeladene Datei sieht nicht wie eine Resultate-CSV aus.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target


# --------------------------------------------------------------------------
# 2. CSV einlesen und vereinheitlichen
# --------------------------------------------------------------------------
def _read_raw_csv(source) -> pd.DataFrame:
    """Liest die CSV robust ein (Datei-Pfad, Bytes oder hochgeladene Datei)."""
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif hasattr(source, "read"):              # z. B. Streamlit-Upload
        raw = source.read()
    else:
        raw = Path(source).read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DataError("Die CSV-Datei konnte nicht gelesen werden (unbekannte Zeichenkodierung).")


def parse_csv(source) -> pd.DataFrame:
    """Liest eine CSV von football-data.co.uk und gibt ein einheitliches DataFrame zurück.

    Spalten: season, date, home, away, home_goals, away_goals, result,
             odds_home, odds_draw, odds_away
    """
    raw = _read_raw_csv(source)
    raw.columns = [str(c).strip() for c in raw.columns]

    # Falls die Datei mehrere Ligen enthält: nur die Super League behalten.
    if "League" in raw.columns:
        league = raw["League"].astype(str).str.strip().str.lower()
        if (league == "super league").any():
            raw = raw[league == "super league"]

    # Pflichtspalten suchen
    found = {key: _find_column(raw, names) for key, names in COLUMN_ALIASES.items()}
    missing = [key for key in ("date", "home", "away", "home_goals", "away_goals")
               if found[key] is None]
    if missing:
        raise DataError(f"In der CSV fehlen Pflichtspalten: {', '.join(missing)}. "
                        f"Vorhanden sind: {', '.join(raw.columns[:15])} ...")

    df = pd.DataFrame({
        "home": raw[found["home"]].astype(str).str.strip(),
        "away": raw[found["away"]].astype(str).str.strip(),
        "home_goals": pd.to_numeric(raw[found["home_goals"]], errors="coerce"),
        "away_goals": pd.to_numeric(raw[found["away_goals"]], errors="coerce"),
    })

    # Datum: football-data.co.uk schreibt TT/MM/JJJJ (manchmal TT/MM/JJ).
    dates = raw[found["date"]].astype(str).str.strip()
    parsed = pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(dates, format="%d/%m/%y", errors="coerce")
    df["date"] = parsed.fillna(fallback)

    # Spiele ohne Resultat oder Datum (z. B. abgesagt) entfernen
    df = df.dropna(subset=["date", "home_goals", "away_goals"]).copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # Resultat immer aus den Toren berechnen (zuverlässiger als die Res-Spalte)
    df["result"] = "D"
    df.loc[df["home_goals"] > df["away_goals"], "result"] = "H"
    df.loc[df["home_goals"] < df["away_goals"], "result"] = "A"

    # Saison: aus der CSV übernehmen, sonst aus dem Datum ableiten.
    # (Wichtig für 2019/20: die Corona-Saison endete erst im August 2020.)
    if found["season"] is not None:
        season = raw.loc[df.index, found["season"]].map(_normalize_season)
    else:
        season = pd.Series(index=df.index, dtype=object)
    derived = df["date"].map(season_from_date)
    df["season"] = season.where(season.notna(), derived)

    # Quoten: pro Zeile die erste vollständig vorhandene Quelle verwenden
    df["odds_home"] = float("nan")
    df["odds_draw"] = float("nan")
    df["odds_away"] = float("nan")
    for cols in ODDS_PREFERENCE:
        if not all(c in raw.columns for c in cols):
            continue
        values = raw.loc[df.index, list(cols)].apply(pd.to_numeric, errors="coerce")
        usable = values.notna().all(axis=1) & (values > 1).all(axis=1) & df["odds_home"].isna()
        df.loc[usable, "odds_home"] = values.loc[usable, cols[0]]
        df.loc[usable, "odds_draw"] = values.loc[usable, cols[1]]
        df.loc[usable, "odds_away"] = values.loc[usable, cols[2]]

    df = df.drop_duplicates(subset=["date", "home", "away"])
    df = df.sort_values(["date", "home"]).reset_index(drop=True)
    if df.empty:
        raise DataError("Die CSV enthält keine gespielten Partien.")
    return df[["season", "date", "home", "away", "home_goals", "away_goals",
               "result", "odds_home", "odds_draw", "odds_away"]]


# --------------------------------------------------------------------------
# 3. In die Datenbank schreiben
# --------------------------------------------------------------------------
def import_csv_to_db(source, data_source: str = "football-data.co.uk",
                     db_path: Path | str = DB_PATH) -> int:
    """Liest eine CSV ein und ersetzt die Spiele in der Datenbank. Gibt die Anzahl Spiele zurück."""
    matches = parse_csv(source)
    count = db.replace_matches(matches, db_path)
    db.set_meta("data_source", data_source, db_path)
    db.set_meta("last_update", datetime.now().isoformat(timespec="minutes"), db_path)
    return count


def refresh_from_web(db_path: Path | str = DB_PATH) -> int:
    """Lädt die aktuelle CSV herunter und importiert sie (für den Button in der App)."""
    path = download_csv()
    return import_csv_to_db(path, db_path=db_path)


# --------------------------------------------------------------------------
# 4. API-Football: kommende Spiele
# --------------------------------------------------------------------------
class ApiError(Exception):
    """Fehlermeldung der API (z. B. ungültiger Schlüssel oder Saison nicht im Gratis-Plan)."""


def _api_get(endpoint: str, params: dict, api_key: str,
             db_path: Path | str = DB_PATH, force: bool = False) -> dict:
    """Ruft die API auf. Antworten werden API_CACHE_HOURS Stunden in der DB gespeichert,
    damit das Gratis-Limit von 100 Anfragen pro Tag reicht."""
    cache_key = endpoint + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    if not force:
        cached = db.cache_get(cache_key, API_CACHE_HOURS, db_path)
        if cached is not None:
            return json.loads(cached)

    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params,
                                headers={"x-apisports-key": api_key}, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ApiError(f"API nicht erreichbar: {exc}") from exc

    # Die API meldet Fehler im Feld «errors» (Liste oder Dictionary), nicht per HTTP-Status.
    errors = data.get("errors")
    if errors:
        message = "; ".join(errors.values()) if isinstance(errors, dict) else "; ".join(map(str, errors))
        raise ApiError(f"API-Fehler: {message}")

    db.cache_set(cache_key, json.dumps(data), db_path)
    return data


def normalize_team_name(name: str) -> str:
    """Vereinfacht Teamnamen für den Vergleich: «FC St. Gallen 1879» -> «st gallen»."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    stop_words = {"fc", "bsc", "sc", "ac", "club", "sport", "1879", "1886", "1896", "1893", "1890"}
    return " ".join(t for t in text.split() if t not in stop_words)


def map_team_names(api_names: list[str], known_names: list[str]) -> dict[str, str]:
    """Ordnet API-Teamnamen den Namen aus der CSV zu (unscharfer Vergleich).

    Nicht zuordenbare Namen fehlen im Ergebnis und werden in der App gemeldet.
    """
    known_norm = {normalize_team_name(k): k for k in known_names}
    mapping = {}
    for api_name in set(api_names):
        norm = normalize_team_name(api_name)
        best, best_score = None, 0.0
        for k_norm, original in known_norm.items():
            score = difflib.SequenceMatcher(None, norm, k_norm).ratio()
            # Bonus, wenn ein markantes Wort übereinstimmt (z. B. «grasshopper» ~ «grasshoppers»)
            for token in norm.split():
                if len(token) >= 4 and any(t.startswith(token) or token.startswith(t)
                                           for t in k_norm.split() if len(t) >= 4):
                    score = max(score, 0.85)
            if score > best_score:
                best, best_score = original, score
        if best is not None and best_score >= 0.6:
            mapping[api_name] = best
    return mapping


def fetch_upcoming_fixtures(api_key: str, season_year: int,
                            db_path: Path | str = DB_PATH, force: bool = False) -> pd.DataFrame:
    """Holt die noch nicht gespielten Spiele (Status «NS») einer Saison von API-Football."""
    data = _api_get("fixtures",
                    {"league": API_LEAGUE_ID, "season": season_year, "status": "NS"},
                    api_key, db_path, force)
    rows = []
    for item in data.get("response", []):
        rows.append({
            "date": pd.to_datetime(item["fixture"]["date"]).tz_convert("Europe/Zurich")
                    .tz_localize(None).normalize(),
            "home_api": item["teams"]["home"]["name"],
            "away_api": item["teams"]["away"]["name"],
        })
    return pd.DataFrame(rows, columns=["date", "home_api", "away_api"])


def import_api_fixtures(api_key: str, season_year: int, known_teams: list[str],
                        days_ahead: int = 21, db_path: Path | str = DB_PATH,
                        force: bool = False) -> tuple[int, list[str]]:
    """Holt kommende Spiele, übersetzt die Teamnamen und speichert sie in der DB.

    Gibt (Anzahl neu gespeicherter Spiele, Liste nicht zuordenbarer Teamnamen) zurück.
    """
    raw = fetch_upcoming_fixtures(api_key, season_year, db_path, force)
    if raw.empty:
        return 0, []
    today = pd.Timestamp.today().normalize()
    raw = raw[(raw["date"] >= today) & (raw["date"] <= today + pd.Timedelta(days=days_ahead))]

    mapping = map_team_names(list(raw["home_api"]) + list(raw["away_api"]), known_teams)
    unmapped = sorted({n for n in list(raw["home_api"]) + list(raw["away_api"]) if n not in mapping})
    raw = raw[raw["home_api"].isin(mapping) & raw["away_api"].isin(mapping)]
    fixtures = pd.DataFrame({
        "date": raw["date"],
        "home": raw["home_api"].map(mapping),
        "away": raw["away_api"].map(mapping),
    })
    return db.upsert_fixtures(fixtures, source="api", db_path=db_path), unmapped
