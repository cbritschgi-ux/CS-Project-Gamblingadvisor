"""
Seite «Match-Analyse»: zwei Teams wählen, Prognose sehen und verstehen, warum.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.common import (COLOR_AWAY, COLOR_HOME, OUTCOME_TEXT, predict,
                          require_context, style_figure)

st.title("Match-Analyse")
st.caption("Wähle zwei Teams der laufenden Saison. Die App zeigt die Prognose und die "
           "Merkmale, auf denen sie beruht.")

ctx = require_context()
teams = ctx.teams

# Vorauswahl: von der Seite «Nächster Spieltag» übergeben oder die ersten zwei Teams
default_home = st.session_state.pop("analyse_home", teams[0])
default_away = st.session_state.pop("analyse_away", teams[1])

col_home, col_away = st.columns(2)
home = col_home.selectbox("Heimteam", teams, index=teams.index(default_home) if default_home in teams else 0)
away_options = [t for t in teams if t != home]
away = col_away.selectbox("Auswärtsteam", away_options,
                          index=away_options.index(default_away) if default_away in away_options else 0)

match = pd.DataFrame({"date": [pd.Timestamp.today().normalize()], "home": [home], "away": [away]})
pred = predict(ctx, match).iloc[0]

# --------------------------------------------------------------------------
# Prognose
# --------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric(f"1 · Sieg {home}", f"{pred['p_home']:.0%}")
c2.metric("X · Unentschieden", f"{pred['p_draw']:.0%}")
c3.metric(f"2 · Sieg {away}", f"{pred['p_away']:.0%}")
st.markdown(f"**Modell-Tipp:** {OUTCOME_TEXT[pred['tip']]}")

# --------------------------------------------------------------------------
# Begründung: Merkmale im Vergleich
# --------------------------------------------------------------------------
st.subheader("Worauf beruht die Prognose?")

e1, e2, e3 = st.columns(3)
e1.metric(f"Elo {home}", f"{pred['elo_home']:.0f}")
e2.metric(f"Elo {away}", f"{pred['elo_away']:.0f}")
e3.metric("Elo-Differenz", f"{pred['elo_diff']:+.0f}",
          help="Positiv = Heimteam stärker. Dazu kommt im Modell der Heimvorteil.")

# Punkte-pro-Spiel-Merkmale haben dieselbe Skala (0–3) und lassen sich gemeinsam zeigen
categories = ["Form (Ø Punkte, letzte 5)", "Heim- bzw. Auswärtsstärke (Saison)",
              "Saisonleistung (Punkte/Spiel)", "Direktvergleich (Ø Punkte, letzte 3)"]
home_values = [pred["form_pts_home"], pred["venue_ppg_home"], pred["season_ppg_home"], pred["h2h_pts_home"]]
# Direktvergleich aus Sicht des Auswärtsteams: dieselbe Paarung mit vertauschten Rollen
swapped = predict(ctx, match.rename(columns={"home": "away", "away": "home"})).iloc[0]
away_values = [pred["form_pts_away"], pred["venue_ppg_away"], pred["season_ppg_away"], swapped["h2h_pts_home"]]

fig = go.Figure()
fig.add_bar(y=categories, x=home_values, orientation="h", name=home,
            marker=dict(color=COLOR_HOME), hovertemplate="%{y}<br>" + home + ": %{x:.2f}<extra></extra>")
fig.add_bar(y=categories, x=away_values, orientation="h", name=away,
            marker=dict(color=COLOR_AWAY), hovertemplate="%{y}<br>" + away + ": %{x:.2f}<extra></extra>")
fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
fig.update_xaxes(range=[0, 3], title="Punkte pro Spiel", showgrid=True,
                 gridcolor="rgba(128,128,128,0.18)")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style_figure(fig, 300), width="stretch")
st.caption("Alle Werte stammen aus Spielen vor heute. Kleine Stichproben (z. B. zu Saisonbeginn) "
           "werden zum Ligadurchschnitt hin geglättet. Ohne gemeinsame Duelle steht der "
           "Direktvergleich auf dem neutralen Wert 1.35.")

# --------------------------------------------------------------------------
# Elo-Verlauf
# --------------------------------------------------------------------------
st.subheader("Elo-Verlauf")
seasons = sorted(ctx.matches["season"].unique())
n_seasons = st.slider("Anzahl Saisons", 1, len(seasons), min(3, len(seasons)))
history = ctx.elo.history[ctx.elo.history["season"].isin(seasons[-n_seasons:])]

fig = go.Figure()
for team, color in ((home, COLOR_HOME), (away, COLOR_AWAY)):
    h = history[history["team"] == team].copy()
    # Lücke statt Verbindungslinie, wenn ein Team eine Saison nicht in der Liga war
    gap = h["date"].diff() > pd.Timedelta(days=150)
    if gap.any():
        breaks = h.loc[gap].assign(elo=None, date=lambda d: d["date"] - pd.Timedelta(days=1))
        h = pd.concat([h, breaks]).sort_values("date")
    fig.add_scatter(x=h["date"], y=h["elo"], mode="lines", name=team,
                    line=dict(color=color, width=2),
                    hovertemplate="%{x|%d.%m.%Y}<br>" + team + ": %{y:.0f}<extra></extra>")
fig.update_layout(hovermode="x unified")
fig.update_yaxes(title="Elo-Rating")
st.plotly_chart(style_figure(fig, 340), width="stretch")
st.caption("Lücken entstehen, wenn ein Team nicht in der Super League spielte.")

# --------------------------------------------------------------------------
# Letzte Spiele und Direktvergleich
# --------------------------------------------------------------------------
def last_games(team: str, n: int = 5) -> pd.DataFrame:
    """Die letzten n Spiele eines Teams mit Resultat aus seiner Sicht."""
    m = ctx.matches[(ctx.matches["home"] == team) | (ctx.matches["away"] == team)].tail(n)
    rows = []
    for r in m.iloc[::-1].itertuples():
        at_home = r.home == team
        gf, ga = (r.home_goals, r.away_goals) if at_home else (r.away_goals, r.home_goals)
        rows.append({
            "Datum": r.date.strftime("%d.%m.%Y"),
            "Gegner": (r.away if at_home else r.home) + (" (H)" if at_home else " (A)"),
            "Resultat": f"{gf}:{ga}",
            "": "S" if gf > ga else ("U" if gf == ga else "N"),
        })
    return pd.DataFrame(rows)


st.subheader("Letzte Spiele")
l1, l2 = st.columns(2)
with l1:
    st.markdown(f"**{home}**")
    st.dataframe(last_games(home), hide_index=True, width="stretch")
with l2:
    st.markdown(f"**{away}**")
    st.dataframe(last_games(away), hide_index=True, width="stretch")

st.subheader("Direktvergleich")
duels = ctx.matches[((ctx.matches["home"] == home) & (ctx.matches["away"] == away)) |
                    ((ctx.matches["home"] == away) & (ctx.matches["away"] == home))]
if duels.empty:
    st.caption("Diese Teams sind in den vorhandenen Daten noch nie aufeinandergetroffen.")
else:
    wins_home = int((((duels["home"] == home) & (duels["result"] == "H")) |
                     ((duels["away"] == home) & (duels["result"] == "A"))).sum())
    draws = int((duels["result"] == "D").sum())
    st.markdown(f"**{len(duels)} Duelle:** {wins_home} Siege {home} · {draws} Unentschieden · "
                f"{len(duels) - wins_home - draws} Siege {away}")
    st.dataframe(
        duels.iloc[::-1].head(8).assign(
            Datum=lambda d: d["date"].dt.strftime("%d.%m.%Y"),
            Resultat=lambda d: d["home_goals"].astype(str) + ":" + d["away_goals"].astype(str),
        )[["Datum", "season", "home", "away", "Resultat"]]
        .rename(columns={"season": "Saison", "home": "Heim", "away": "Auswärts"}),
        hide_index=True, width="stretch",
    )
