"""
Gemeinsame Bausteine aller Seiten: Daten laden (mit Cache), Farben, Grafiken.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH, MODEL_PATH
from src import database as db
from src.elo import EloParams, EloResult, run_elo
from src.model import TrainedModel, current_season, load_model, predict_fixtures

# Farben: Heim = Blau, Auswärts = Orange, Unentschieden = neutrales Grau.
# (Kategorische Palette, auf Farbenblindheit geprüft; Grau als neutrale Mitte.)
COLOR_HOME = "#2a78d6"
COLOR_AWAY = "#eb6834"
COLOR_DRAW = "#9a9893"
OUTCOME_COLORS = {"H": COLOR_HOME, "D": COLOR_DRAW, "A": COLOR_AWAY}
OUTCOME_SYMBOL = {"H": "1", "D": "X", "A": "2"}
OUTCOME_TEXT = {"H": "Heimsieg", "D": "Unentschieden", "A": "Auswärtssieg"}

DATA_PAGE = "views/daten.py"


@dataclass
class AppContext:
    """Alles, was eine Seite für Prognosen braucht."""
    matches: pd.DataFrame          # alle gespielten Spiele
    model: TrainedModel
    elo: EloResult
    season: str                    # laufende Saison
    season_matches: pd.DataFrame   # gespielte Spiele der laufenden Saison
    teams: list[str]               # Teams der laufenden Saison


# --------------------------------------------------------------------------
# Laden mit Cache. Der «version»-Parameter ist der Änderungszeitpunkt der Datei:
# Ändert sich die Datenbank oder das Modell, wird automatisch neu geladen.
# --------------------------------------------------------------------------
def _file_version(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_matches(version: float) -> pd.DataFrame:
    return db.load_matches()


@st.cache_resource(show_spinner=False)
def _load_model(version: float) -> TrainedModel | None:
    return load_model()


@st.cache_data(show_spinner=False)
def _run_elo(version: float, params: tuple) -> EloResult:
    return run_elo(db.load_matches(), EloParams(**dict(params)))


def data_version() -> float:
    return _file_version(DB_PATH)


def model_version() -> float:
    return _file_version(MODEL_PATH)


def get_matches() -> pd.DataFrame:
    return _load_matches(data_version())


def get_model() -> TrainedModel | None:
    return _load_model(model_version())


def get_context() -> AppContext | None:
    """Gibt den App-Kontext zurück oder None, wenn Daten oder Modell fehlen."""
    matches = get_matches()
    model = get_model()
    if matches.empty or model is None:
        return None
    elo = _run_elo(data_version(), tuple(sorted(model.elo_params.items())))
    season = current_season(matches)
    season_matches = matches[matches["season"] == season]
    teams = sorted(set(season_matches["home"]) | set(season_matches["away"]))
    return AppContext(matches, model, elo, season, season_matches, teams)


def require_context() -> AppContext:
    """Bricht die Seite mit einem Hinweis ab, wenn noch keine Daten/kein Modell da sind."""
    ctx = get_context()
    if ctx is None:
        st.info("Noch keine Daten oder kein trainiertes Modell vorhanden. "
                "Lade zuerst die Resultate und trainiere das Modell.", icon=":material/info:")
        st.page_link(DATA_PAGE, label="Zu «Daten & Einstellungen»", icon=":material/database:")
        st.stop()
    return ctx


def predict(ctx: AppContext, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Prognose für beliebige Paarungen (Spalten date, home, away)."""
    fixtures = fixtures.copy()
    fixtures["season"] = ctx.season
    return predict_fixtures(ctx.model, ctx.matches, fixtures, ctx.elo)


def api_key_default() -> str:
    """API-Schlüssel aus .streamlit/secrets.toml oder Umgebungsvariable (falls vorhanden)."""
    try:
        if "API_FOOTBALL_KEY" in st.secrets:
            return st.secrets["API_FOOTBALL_KEY"]
    except Exception:          # keine secrets.toml vorhanden
        pass
    return os.environ.get("API_FOOTBALL_KEY", "")


def sample_data_warning() -> None:
    """Deutlicher Hinweis, wenn erfundene Testdaten geladen sind."""
    if db.get_meta("data_source") == "beispieldaten":
        st.warning("**Beispieldaten aktiv:** Teams und Resultate sind erfunden und dienen nur "
                   "zum Testen. Für Präsentation und Video echte Daten laden.",
                   icon=":material/science:")


# --------------------------------------------------------------------------
# Grafiken
# --------------------------------------------------------------------------
def style_figure(fig: go.Figure, height: int) -> go.Figure:
    """Einheitlicher, zurückhaltender Stil für alle Grafiken."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(font_size=13),
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def probability_chart(predictions: pd.DataFrame) -> go.Figure:
    """Gestapelte 100-%-Balken: eine Zeile pro Spiel, Heim | X | Auswärts."""
    labels = [f"{r.home} – {r.away}" for r in predictions.itertuples()]
    fig = go.Figure()
    for key, col in (("H", "p_home"), ("D", "p_draw"), ("A", "p_away")):
        values = predictions[col].to_numpy()
        fig.add_bar(
            y=labels, x=values, orientation="h", name=OUTCOME_TEXT[key],
            marker=dict(color=OUTCOME_COLORS[key], line=dict(width=0)),
            text=[f"{v:.0%}" if v >= 0.08 else "" for v in values],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white"),
            hovertemplate="%{y}<br>" + OUTCOME_TEXT[key] + ": %{x:.1%}<extra></extra>",
        )
    fig.update_layout(barmode="stack", bargap=0.35, legend_traceorder="normal")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=False)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return style_figure(fig, height=90 + 46 * len(predictions))


def percent(value: float) -> str:
    return f"{value:.0%}"


def fmt_int(n: int) -> str:
    """Ganze Zahl mit Schweizer Tausendertrennzeichen, z. B. 2'556."""
    return f"{int(n):,}".replace(",", "'")


def pct_column(label: str, help_text: str | None = None):
    """Tabellenspalte mit Balken für Werte zwischen 0 und 100 (%)."""
    return st.column_config.ProgressColumn(label, help=help_text, format="%.0f%%",
                                           min_value=0, max_value=100)


def table_height(n_rows: int) -> int:
    """Höhe, damit eine Tabelle alle Zeilen ohne Scrollen zeigt."""
    return 35 * (n_rows + 1) + 3


WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def short_date(ts) -> str:
    """Datum auf Deutsch, z. B. «Sa 17.10.»."""
    ts = pd.Timestamp(ts)
    return f"{WEEKDAYS[ts.weekday()]} {ts:%d.%m.}"
