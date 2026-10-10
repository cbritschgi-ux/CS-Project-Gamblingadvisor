"""
Seite «Daten & Einstellungen»: Resultate laden, Modell trainieren,
kommende Spiele erfassen (API oder von Hand), Datenbank ansehen.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

import pandas as pd
import streamlit as st

from config import CSV_URL, ELO_HOME_ADVANTAGE, ELO_K, ELO_SEASON_REGRESSION
from src import database as db
from src.data_loader import (ApiError, DataError, import_api_fixtures,
                             import_csv_to_db, refresh_from_web)
from src.elo import EloParams
from src.model import current_season, train_models
from views.common import api_key_default, fmt_int, get_matches, get_model, short_date

st.title("Daten & Einstellungen")

# Meldungen über einen Neustart der Seite hinweg anzeigen (nach Import/Training
# wird die Seite neu geladen, damit überall der neue Datenstand gilt).
if "flash" in st.session_state:
    kind, message = st.session_state.pop("flash")
    getattr(st, kind)(message)


def flash(kind: str, message: str) -> None:
    """Meldung merken und Seite neu laden."""
    st.session_state["flash"] = (kind, message)
    st.rerun()


def train_and_report(params: EloParams) -> None:
    """Trainiert das Modell und zeigt das Ergebnis kurz an."""
    with st.spinner("Modell wird trainiert …"):
        try:
            trained = train_models(db.load_matches(), params)
        except ValueError as exc:
            st.error(f"Training nicht möglich: {exc}", icon=":material/error:")
            return
    best = trained.metrics.set_index("Modell").loc[trained.name]
    flash("success", f"Modell trainiert: **{trained.name}** · Trefferquote auf "
                     f"{trained.test_season}: {best['Trefferquote']:.1%}")


# --------------------------------------------------------------------------
# 1. Resultate (Datenbank)
# --------------------------------------------------------------------------
st.header("1 · Resultate", divider="gray")
matches = get_matches()
if matches.empty:
    st.caption("Die Datenbank ist noch leer.")
else:
    per_season = matches.groupby("season").size()
    m1, m2, m3 = st.columns(3)
    m1.metric("Spiele", fmt_int(len(matches)))
    m2.metric("Saisons", len(per_season), help=f"{per_season.index[0]} bis {per_season.index[-1]}")
    m3.metric("Letztes Spiel", matches["date"].max().strftime("%d.%m.%Y"))
    st.caption(f"Quelle: {db.get_meta('data_source', '–')} · "
               f"aktualisiert {db.get_meta('last_update', '–').replace('T', ' ')}")

tab_web, tab_upload, tab_sample = st.tabs(["Aus dem Internet laden", "CSV hochladen", "Beispieldaten (Test)"])
with tab_web:
    st.markdown(f"Lädt alle Super-League-Resultate inkl. Wettquoten von [football-data.co.uk]({CSV_URL}).")
    if st.button("Resultate herunterladen und importieren", icon=":material/download:", type="primary"):
        try:
            with st.spinner("Lade Daten …"):
                n = refresh_from_web()
            flash("success", f"{fmt_int(n)} Spiele importiert. Trainiere jetzt das Modell (Abschnitt 2).")
        except DataError as exc:
            st.error(str(exc), icon=":material/error:")
with tab_upload:
    st.markdown(f"Falls der Download blockiert ist: Datei im Browser unter {CSV_URL} speichern und hier hochladen.")
    upload = st.file_uploader("CSV-Datei", type=["csv"])
    if upload is not None and st.button("Hochgeladene CSV importieren", icon=":material/upload:"):
        try:
            n = import_csv_to_db(upload, data_source="football-data.co.uk (Upload)")
            flash("success", f"{fmt_int(n)} Spiele importiert. Trainiere jetzt das Modell (Abschnitt 2).")
        except DataError as exc:
            st.error(str(exc), icon=":material/error:")
with tab_sample:
    st.markdown("Erzeugt **erfundene** Teams und Resultate, um die App ohne Internet zu testen. "
                "Ersetzt die vorhandenen Spiele und darf nicht für die Präsentation verwendet werden.")
    if st.button("Beispieldaten laden", icon=":material/science:"):
        from scripts.make_sample_data import write_sample_csv
        n = import_csv_to_db(write_sample_csv(), data_source="beispieldaten")
        flash("warning", f"{fmt_int(n)} erfundene Spiele geladen. Trainiere jetzt das Modell (Abschnitt 2).")

# --------------------------------------------------------------------------
# 2. Modell
# --------------------------------------------------------------------------
st.header("2 · Modell trainieren", divider="gray")
model = get_model()
if model is not None:
    st.caption(f"Aktuelles Modell: {model.name} · trainiert {model.trained_at.replace('T', ' ')} · "
               f"Elo-Einstellungen K={model.elo_params['k']:.0f}, "
               f"Heimvorteil={model.elo_params['home_advantage']:.0f}, "
               f"Saison-Regression={model.elo_params['season_regression']:.0%}")

with st.form("training"):
    st.markdown("Die Elo-Einstellungen beeinflussen die wichtigsten Merkmale. Probiert verschiedene "
                "Werte aus und vergleicht die Trefferquote auf der Seite «Wie gut ist das Modell?».")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("K-Faktor", 5, 50, int(model.elo_params["k"]) if model else ELO_K,
                  help="Wie stark ein einzelnes Spiel das Rating verändert.")
    home_adv = c2.slider("Heimvorteil (Elo-Punkte)", 0, 150,
                         int(model.elo_params["home_advantage"]) if model else ELO_HOME_ADVANTAGE)
    regression = c3.slider("Saison-Regression", 0.0, 0.6,
                           float(model.elo_params["season_regression"]) if model else ELO_SEASON_REGRESSION,
                           step=0.05, help="Wie stark die Ratings im Sommer zum Mittel zurückrücken.")
    if st.form_submit_button("Modell trainieren", icon=":material/model_training:", type="primary",
                             disabled=matches.empty):
        train_and_report(EloParams(k=k, home_advantage=home_adv, season_regression=regression))

# --------------------------------------------------------------------------
# 3. Kommende Spiele
# --------------------------------------------------------------------------
st.header("3 · Kommende Spiele", divider="gray")
if matches.empty:
    st.caption("Zuerst Resultate laden.")
    st.stop()

season = current_season(matches)
season_matches = matches[matches["season"] == season]
teams = sorted(set(season_matches["home"]) | set(season_matches["away"]))

tab_api, tab_manual = st.tabs(["Über API-Football laden", "Von Hand erfassen"])
with tab_api:
    st.markdown("Gratis-Schlüssel auf [api-football.com](https://www.api-football.com) "
                "(100 Anfragen pro Tag). Antworten werden 12 Stunden zwischengespeichert. "
                "Je nach Plan sind nicht alle Saisons freigeschaltet – die Fehlermeldung der API "
                "wird dann hier angezeigt.")
    with st.form("api"):
        key = st.text_input("API-Schlüssel", value=api_key_default(), type="password")
        a1, a2 = st.columns(2)
        season_year = a1.number_input("Saison (Startjahr)", min_value=2010, max_value=2100,
                                      value=int(season[:4]), step=1)
        days_ahead = a2.slider("Spiele der nächsten … Tage", 3, 60, 21)
        force = st.checkbox("Cache ignorieren (verbraucht eine Anfrage)")
        if st.form_submit_button("Kommende Spiele laden", icon=":material/cloud_download:"):
            if not key:
                st.error("Bitte API-Schlüssel eingeben.")
            else:
                try:
                    n, unmapped = import_api_fixtures(key, int(season_year), teams, days_ahead, force=force)
                    message = f"{n} neue Spiele gespeichert."
                    if unmapped:
                        message += (" Nicht zuordenbare Teamnamen: " + ", ".join(unmapped)
                                    + ". Diese Spiele bei Bedarf von Hand erfassen.")
                    flash("warning" if unmapped else "success", message)
                except ApiError as exc:
                    st.error(str(exc), icon=":material/error:")

with tab_manual:
    with st.form("manual", clear_on_submit=True):
        d1, d2, d3 = st.columns([1, 2, 2])
        date = d1.date_input("Datum", value=pd.Timestamp.today() + pd.Timedelta(days=1), format="DD.MM.YYYY")
        home = d2.selectbox("Heimteam", teams, index=None, placeholder="Team wählen")
        away = d3.selectbox("Auswärtsteam", teams, index=None, placeholder="Team wählen")
        if st.form_submit_button("Spiel hinzufügen", icon=":material/add:"):
            if not home or not away or home == away:
                st.error("Bitte zwei verschiedene Teams wählen.")
            else:
                n = db.upsert_fixtures(pd.DataFrame({"date": [date], "home": [home], "away": [away]}),
                                       source="manuell")
                flash("success", f"{home} – {away} hinzugefügt." if n else "Dieses Spiel ist bereits erfasst.")

fixtures = db.load_fixtures(only_upcoming=True)
if fixtures.empty:
    st.caption("Keine kommenden Spiele erfasst.")
else:
    view = fixtures.assign(Datum=fixtures["date"].map(short_date))[["id", "Datum", "home", "away", "source"]]
    view = view.rename(columns={"home": "Heim", "away": "Auswärts", "source": "Quelle"})
    event = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="multi-row", column_config={"id": None})
    selected = event.selection.rows
    if st.button(f"{len(selected)} markierte Spiele löschen", icon=":material/delete:",
                 disabled=not selected):
        for row in selected:
            db.delete_fixture(view.iloc[row]["id"])
        st.rerun()

# --------------------------------------------------------------------------
# 4. Datenbank ansehen
# --------------------------------------------------------------------------
st.header("4 · Datenbank ansehen", divider="gray")
seasons = sorted(matches["season"].unique(), reverse=True)
chosen = st.selectbox("Saison", seasons)
subset = matches[matches["season"] == chosen].sort_values("date", ascending=False)
st.dataframe(
    subset.drop(columns=["id"]).assign(date=subset["date"].dt.strftime("%d.%m.%Y")),
    hide_index=True, width="stretch",
    column_config={"season": "Saison", "date": "Datum", "home": "Heim", "away": "Auswärts",
                   "home_goals": "Tore H", "away_goals": "Tore A", "result": "Ausgang",
                   "odds_home": "Quote 1", "odds_draw": "Quote X", "odds_away": "Quote 2"},
)
