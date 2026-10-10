"""
Seite «Nächster Spieltag»: Prognosen für alle erfassten kommenden Spiele.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

import pandas as pd
import streamlit as st

from src import database as db
from views.common import (DATA_PAGE, OUTCOME_SYMBOL, OUTCOME_TEXT, pct_column,
                          predict, probability_chart, require_context,
                          short_date, table_height)

st.title("Nächster Spieltag")
st.caption("Wahrscheinlichkeiten für Heimsieg, Unentschieden und Auswärtssieg – "
           "berechnet aus Elo-Rating, Form, Saisonleistung und Direktvergleich.")

ctx = require_context()
fixtures = db.load_fixtures(only_upcoming=True)

if fixtures.empty:
    st.info("Es sind noch keine kommenden Spiele erfasst. Lade sie über API-Football "
            "oder erfasse sie von Hand.", icon=":material/event_busy:")
    st.page_link(DATA_PAGE, label="Kommende Spiele erfassen", icon=":material/add:")
    st.stop()

# Nur die nächsten 14 Tage als «Spieltag» zeigen (wählbar)
first_date = fixtures["date"].min()
days = st.segmented_control("Zeitraum", options=[7, 14, 28], default=14,
                            format_func=lambda d: f"{d} Tage")
window = fixtures[fixtures["date"] <= first_date + pd.Timedelta(days=days or 14)]

unknown = sorted((set(window["home"]) | set(window["away"])) - set(ctx.teams))
if unknown:
    st.warning(f"Unbekannte Teams (nicht in der laufenden Saison): {', '.join(unknown)}. "
               "Ihre Prognose ist unsicher.", icon=":material/warning:")

pred = predict(ctx, window)

# Grafik: alle Spiele auf einen Blick
st.plotly_chart(probability_chart(pred), width="stretch")

# Tabelle mit Wahrscheinlichkeiten (in %) und Modell-Tipp
table = pd.DataFrame({
    "Datum": pred["date"].map(short_date),
    "Heim": pred["home"],
    "Auswärts": pred["away"],
    "1": pred["p_home"] * 100,
    "X": pred["p_draw"] * 100,
    "2": pred["p_away"] * 100,
    "Tipp": pred["tip"].map(lambda t: f"{OUTCOME_SYMBOL[t]} ({OUTCOME_TEXT[t]})"),
    "Elo-Diff.": pred["elo_diff"].round(0).astype(int),
})
st.dataframe(
    table, hide_index=True, width="stretch", height=table_height(len(table)),
    column_config={
        "1": pct_column("1", "Heimsieg"),
        "X": pct_column("X", "Unentschieden"),
        "2": pct_column("2", "Auswärtssieg"),
        "Elo-Diff.": st.column_config.NumberColumn(
            help="Elo Heimteam minus Elo Auswärtsteam (ohne Heimvorteil)"),
    },
)

# Einzelnes Spiel genauer anschauen
st.subheader("Einzelnes Spiel analysieren")
choice = st.selectbox("Spiel", options=list(pred.index),
                      format_func=lambda i: f"{pred.at[i, 'home']} – {pred.at[i, 'away']} "
                                            f"({pred.at[i, 'date']:%d.%m.})")
if st.button("Zur Match-Analyse", icon=":material/compare_arrows:"):
    st.session_state["analyse_home"] = pred.at[choice, "home"]
    st.session_state["analyse_away"] = pred.at[choice, "away"]
    st.switch_page("views/match_analyse.py")

st.caption("Hinweis: Bei drei möglichen Ausgängen ist eine Trefferquote um 50 % bereits gut. "
           "Unentschieden sagt kaum ein Modell zuverlässig voraus.")
