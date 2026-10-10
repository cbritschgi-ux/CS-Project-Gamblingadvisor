"""
Super-League-Prognose – Startpunkt der Streamlit-App.

Starten mit:   streamlit run app.py

Diese Datei legt nur die Navigation und die Seitenleiste fest. Die einzelnen
Seiten liegen im Ordner «views», die Logik (Daten, Elo, ML, Simulation) in «src».

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

import streamlit as st

from src import database as db
from views.common import fmt_int, get_matches, get_model, sample_data_warning

st.set_page_config(page_title="Super-League-Prognose", page_icon="⚽", layout="wide")

pages = {
    "Prognosen": [
        st.Page("views/spieltag.py", title="Nächster Spieltag", icon=":material/event:", default=True),
        st.Page("views/match_analyse.py", title="Match-Analyse", icon=":material/compare_arrows:"),
        st.Page("views/saison_simulation.py", title="Saison-Simulation", icon=":material/casino:"),
    ],
    "Mitmachen": [
        st.Page("views/tippspiel.py", title="Tippspiel", icon=":material/how_to_vote:"),
    ],
    "Modell & Daten": [
        st.Page("views/modellguete.py", title="Wie gut ist das Modell?", icon=":material/insights:"),
        st.Page("views/daten.py", title="Daten & Einstellungen", icon=":material/database:"),
    ],
}
navigation = st.navigation(pages)

# Seitenleiste: Datenstand auf einen Blick
with st.sidebar:
    st.markdown("### ⚽ Super-League-Prognose")
    st.caption("Wer gewinnt am Wochenende? Datenbasierte Zweitmeinung für dein Tippspiel.")
    matches = get_matches()
    model = get_model()
    if matches.empty:
        st.caption("Noch keine Daten geladen.")
    else:
        last = matches["date"].max().strftime("%d.%m.%Y")
        st.caption(f"**{fmt_int(len(matches))}** Spiele · **{matches['season'].nunique()}** Saisons "
                   f"· letztes Spiel {last}")
        st.caption(f"Quelle: {db.get_meta('data_source', '–')}")
    if model is not None:
        st.caption(f"Modell: {model.name} · trainiert {model.trained_at.replace('T', ' ')}")

sample_data_warning()
navigation.run()
