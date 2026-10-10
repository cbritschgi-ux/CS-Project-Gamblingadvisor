"""
Seite «Tippspiel»: eigene Tipps abgeben und gegen das Modell antreten.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import database as db
from views.common import (COLOR_AWAY, COLOR_HOME, DATA_PAGE, OUTCOME_SYMBOL,
                          OUTCOME_TEXT, pct_column, predict, require_context,
                          short_date, style_figure)

st.title("Tippspiel: Du gegen das Modell")
st.caption("Tippe die kommenden Spiele. Sobald die Resultate in der Datenbank sind, "
           "zeigt die Rangliste, wer besser lag. Pro richtigem Tipp (1, X oder 2) gibt es einen Punkt.")

ctx = require_context()

# --------------------------------------------------------------------------
# 1. Tipps abgeben
# --------------------------------------------------------------------------
st.subheader("Tipps abgeben")
player = st.text_input("Dein Name", key="player_name", max_chars=40,
                       placeholder="z. B. Christian").strip()
fixtures = db.load_fixtures(only_upcoming=True)

if fixtures.empty:
    st.info("Keine kommenden Spiele erfasst.", icon=":material/event_busy:")
    st.page_link(DATA_PAGE, label="Kommende Spiele erfassen", icon=":material/add:")
elif not player:
    st.caption("Gib zuerst deinen Namen ein.")
else:
    pred = predict(ctx, fixtures)
    existing = db.load_tips()
    mine = {} if existing.empty else {
        (r.date, r.home, r.away): r.tip for r in existing[existing["player"] == player].itertuples()
    }
    show_model = st.toggle("Modell-Prognose anzeigen", value=False,
                           help="Erst selbst überlegen, dann vergleichen.")

    options = ["H", "D", "A"]
    with st.form("tipps"):
        choices = {}
        for i, r in pred.iterrows():
            previous = mine.get((r["date"], r["home"], r["away"]))
            label = f"{short_date(r['date'])} · **{r['home']}** – **{r['away']}**"
            if show_model:
                label += (f"  \nModell: 1 {r['p_home']:.0%} · X {r['p_draw']:.0%} · "
                          f"2 {r['p_away']:.0%}")
            choices[i] = st.radio(
                label, options, horizontal=True, key=f"tip_{i}",
                index=options.index(previous) if previous else None,
                format_func=lambda o, r=r: {"H": f"1 · {r['home']}", "D": "X",
                                            "A": f"2 · {r['away']}"}[o],
            )
        submitted = st.form_submit_button("Tipps speichern", icon=":material/save:", type="primary")

    if submitted:
        saved = 0
        for i, tip in choices.items():
            if tip is None:
                continue
            r = pred.loc[i]
            db.save_tip(player, r["date"], r["home"], r["away"], tip, r["tip"],
                        (r["p_home"], r["p_draw"], r["p_away"]))
            saved += 1
        st.success(f"{saved} Tipp(s) gespeichert.", icon=":material/check:")

# --------------------------------------------------------------------------
# 2. Auswertung
# --------------------------------------------------------------------------
st.subheader("Rangliste")
tips = db.load_tips()
if tips.empty:
    st.caption("Noch keine Tipps abgegeben.")
    st.stop()

# Tipps mit Resultaten verbinden. Das Datum darf um bis zu 3 Tage abweichen
# (Verschiebungen oder ungenaue Erfassung).
results = ctx.matches[["date", "home", "away", "result", "home_goals", "away_goals"]]
merged = tips.merge(results, on=["home", "away"], suffixes=("", "_played"))
merged = merged[(merged["date_played"] - merged["date"]).abs() <= pd.Timedelta(days=3)]
merged = merged.drop_duplicates(subset=["id"])

if merged.empty:
    st.caption("Zu deinen Tipps gibt es noch keine Resultate. Aktualisiere die Daten nach dem Spieltag.")
    st.page_link(DATA_PAGE, label="Daten aktualisieren", icon=":material/refresh:")
    st.stop()

merged["Punkt Mensch"] = (merged["tip"] == merged["result"]).astype(int)
merged["Punkt Modell"] = (merged["model_tip"] == merged["result"]).astype(int)

board = (merged.groupby("player")
         .agg(Tipps=("id", "size"), Punkte=("Punkt Mensch", "sum"), Modell=("Punkt Modell", "sum"))
         .reset_index().rename(columns={"player": "Name"}))
board["Trefferquote"] = board["Punkte"] / board["Tipps"] * 100
board["Bilanz vs. Modell"] = board["Punkte"] - board["Modell"]
board = board.sort_values(["Trefferquote", "Punkte"], ascending=False)

st.dataframe(
    board, hide_index=True, width="stretch",
    column_config={
        "Trefferquote": pct_column("Trefferquote"),
        "Modell": st.column_config.NumberColumn("Punkte Modell", help="Punkte des Modells auf denselben Spielen"),
        "Bilanz vs. Modell": st.column_config.NumberColumn(format="%+d"),
    },
)

# Grafik: Mensch und Modell auf denselben Spielen
fig = go.Figure()
fig.add_bar(x=board["Name"], y=board["Punkte"], name="Mensch", marker=dict(color=COLOR_HOME),
            hovertemplate="%{x}: %{y} Punkte<extra></extra>")
fig.add_bar(x=board["Name"], y=board["Modell"], name="Modell (gleiche Spiele)",
            marker=dict(color=COLOR_AWAY), hovertemplate="Modell: %{y} Punkte<extra></extra>")
fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.06)
fig.update_yaxes(title="Punkte")
st.plotly_chart(style_figure(fig, 320), width="stretch")

with st.expander("Alle ausgewerteten Tipps"):
    detail = merged.assign(
        Datum=merged["date_played"].map(short_date),
        Spiel=merged["home"] + " – " + merged["away"],
        Resultat=merged["home_goals"].astype(str) + ":" + merged["away_goals"].astype(str),
        Tipp=merged["tip"].map(OUTCOME_SYMBOL),
        Modelltipp=merged["model_tip"].map(OUTCOME_SYMBOL),
        Ausgang=merged["result"].map(OUTCOME_TEXT),
    )[["Datum", "player", "Spiel", "Resultat", "Ausgang", "Tipp", "Modelltipp"]]
    st.dataframe(detail.rename(columns={"player": "Name"}), hide_index=True, width="stretch")
