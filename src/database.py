"""
Datenbank-Schicht (SQLite): Spiele, kommende Spiele, Tipps, API-Cache.

Alle anderen Module greifen nur über die Funktionen in dieser Datei auf die
Datenbank zu. So steht das SQL an einem einzigen Ort.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config import DB_PATH

# Tabellen-Definitionen. «IF NOT EXISTS» erlaubt es, init_db() beliebig oft aufzurufen.
SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (          -- gespielte Partien (aus der CSV)
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    season      TEXT NOT NULL,                -- z. B. «2025/2026»
    date        TEXT NOT NULL,                -- ISO-Datum JJJJ-MM-TT
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    home_goals  INTEGER NOT NULL,
    away_goals  INTEGER NOT NULL,
    result      TEXT NOT NULL CHECK (result IN ('H', 'D', 'A')),
    odds_home   REAL,                         -- Buchmacherquoten (falls vorhanden)
    odds_draw   REAL,
    odds_away   REAL,
    UNIQUE (date, home, away)
);

CREATE TABLE IF NOT EXISTS fixtures (         -- kommende Spiele (API oder manuell)
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    date    TEXT NOT NULL,
    home    TEXT NOT NULL,
    away    TEXT NOT NULL,
    source  TEXT NOT NULL,                    -- «api» oder «manuell»
    UNIQUE (date, home, away)
);

CREATE TABLE IF NOT EXISTS user_tips (        -- Tipps der Nutzerinnen und Nutzer
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player      TEXT NOT NULL,
    date        TEXT NOT NULL,
    home        TEXT NOT NULL,
    away        TEXT NOT NULL,
    tip         TEXT NOT NULL CHECK (tip IN ('H', 'D', 'A')),
    model_tip   TEXT NOT NULL CHECK (model_tip IN ('H', 'D', 'A')),
    p_home      REAL,
    p_draw      REAL,
    p_away      REAL,
    created_at  TEXT NOT NULL,
    UNIQUE (player, date, home, away)
);

CREATE TABLE IF NOT EXISTS api_cache (        -- zwischengespeicherte API-Antworten
    endpoint    TEXT PRIMARY KEY,
    response    TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (             -- einfache Schlüssel-Wert-Infos
    key     TEXT PRIMARY KEY,
    value   TEXT
);
"""


@contextmanager
def _connect(db_path: Path | str = DB_PATH):
    """Öffnet eine Verbindung, bestätigt Änderungen (commit) und schliesst sie wieder."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Legt alle Tabellen an (falls sie noch nicht existieren)."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# Gespielte Partien
# --------------------------------------------------------------------------
def replace_matches(matches: pd.DataFrame, db_path: Path | str = DB_PATH) -> int:
    """Ersetzt alle gespielten Partien durch den neuen Datenstand.

    Die CSV enthält immer alle Saisons, darum wird die Tabelle komplett neu
    geschrieben statt einzelne Zeilen zu ergänzen.
    """
    init_db(db_path)
    rows = matches.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    columns = ["season", "date", "home", "away", "home_goals", "away_goals",
               "result", "odds_home", "odds_draw", "odds_away"]
    # NaN (fehlende Quoten) als NULL speichern
    records = [tuple(None if pd.isna(v) else v for v in row)
               for row in rows[columns].itertuples(index=False)]
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM matches")
        conn.executemany(
            f"INSERT OR IGNORE INTO matches ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})",
            records,
        )
    return len(records)


def load_matches(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Liest alle gespielten Partien, chronologisch sortiert."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    init_db(db_path)
    with _connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM matches ORDER BY date, id", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------
# Kommende Spiele
# --------------------------------------------------------------------------
def upsert_fixtures(fixtures: pd.DataFrame, source: str,
                    db_path: Path | str = DB_PATH) -> int:
    """Speichert kommende Spiele. Bereits vorhandene (gleiches Datum + Teams) werden übersprungen."""
    init_db(db_path)
    rows = [(pd.Timestamp(r.date).strftime("%Y-%m-%d"), r.home, r.away, source)
            for r in fixtures.itertuples(index=False)]
    with _connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO fixtures (date, home, away, source) VALUES (?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes - before


def load_fixtures(only_upcoming: bool = True, today: pd.Timestamp | None = None,
                  db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Liest kommende Spiele. Mit only_upcoming=True nur Spiele ab heute."""
    if not Path(db_path).exists():
        return pd.DataFrame(columns=["id", "date", "home", "away", "source"])
    init_db(db_path)
    query = "SELECT * FROM fixtures"
    params: tuple = ()
    if only_upcoming:
        today = today or pd.Timestamp.today().normalize()
        query += " WHERE date >= ?"
        params = (today.strftime("%Y-%m-%d"),)
    with _connect(db_path) as conn:
        df = pd.read_sql_query(query + " ORDER BY date, home", conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


def delete_fixture(fixture_id: int, db_path: Path | str = DB_PATH) -> None:
    """Löscht ein einzelnes kommendes Spiel (z. B. falsch erfasst)."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM fixtures WHERE id = ?", (int(fixture_id),))


# --------------------------------------------------------------------------
# Tippspiel
# --------------------------------------------------------------------------
def save_tip(player: str, date, home: str, away: str, tip: str, model_tip: str,
             probs: tuple[float, float, float], db_path: Path | str = DB_PATH) -> None:
    """Speichert einen Tipp. Ein erneuter Tipp derselben Person überschreibt den alten."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO user_tips (player, date, home, away, tip, model_tip,
                                      p_home, p_draw, p_away, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (player, date, home, away) DO UPDATE SET
                   tip = excluded.tip, model_tip = excluded.model_tip,
                   p_home = excluded.p_home, p_draw = excluded.p_draw,
                   p_away = excluded.p_away, created_at = excluded.created_at""",
            (player.strip(), pd.Timestamp(date).strftime("%Y-%m-%d"), home, away,
             tip, model_tip, *map(float, probs), datetime.now().isoformat(timespec="seconds")),
        )


def load_tips(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Liest alle abgegebenen Tipps."""
    if not Path(db_path).exists():
        return pd.DataFrame()
    init_db(db_path)
    with _connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM user_tips ORDER BY date, player", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------
# API-Cache und Meta-Infos
# --------------------------------------------------------------------------
def cache_get(endpoint: str, max_age_hours: float,
              db_path: Path | str = DB_PATH) -> str | None:
    """Gibt eine gespeicherte API-Antwort zurück, wenn sie jünger als max_age_hours ist."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT response, fetched_at FROM api_cache WHERE endpoint = ?", (endpoint,)
        ).fetchone()
    if row is None:
        return None
    response, fetched_at = row
    if datetime.now() - datetime.fromisoformat(fetched_at) > timedelta(hours=max_age_hours):
        return None
    return response


def cache_set(endpoint: str, response: str, db_path: Path | str = DB_PATH) -> None:
    """Speichert eine API-Antwort mit Zeitstempel."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (endpoint, response, fetched_at) VALUES (?, ?, ?)",
            (endpoint, response, datetime.now().isoformat(timespec="seconds")),
        )


def set_meta(key: str, value: str, db_path: Path | str = DB_PATH) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def get_meta(key: str, default: str | None = None,
             db_path: Path | str = DB_PATH) -> str | None:
    if not Path(db_path).exists():
        return default
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default
