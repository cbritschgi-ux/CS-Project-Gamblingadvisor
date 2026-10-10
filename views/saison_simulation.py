"""
Seite «Saison-Simulation»: Restsaison tausendfach simulieren (Monte-Carlo).

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import TEAMS_PER_SEASON
from src.simulation import SimulationResult, build_prob_matrix, simulate_season
from src.standings import compute_table
from views.common import (data_version, fmt_int, get_context, model_version,
                          pct_column, predict, require_context, style_figure,
                          table_height)

st.title("Saison-Simulation")
st.caption("Die Restsaison wird viele tausend Mal ausgewürfelt – jedes Spiel mit den "
           "Wahrscheinlichkeiten des Modells. So entstehen Chancen auf Meistertitel, "
           "obere Gruppe und Abstiegsplätze.")

ctx = require_context()

if len(ctx.teams) != TEAMS_PER_SEASON:
    st.info(f"Die Simulation ist für das Format mit {TEAMS_PER_SEASON} Teams gebaut. "
            f"Die laufende Saison {ctx.season} hat {len(ctx.teams)} Teams.")
    st.stop()


@st.cache_data(show_spinner=False)
def run_simulation(data_v: float, model_v: float, n_sims: int, seed: int) -> SimulationResult:
    """Cache: dieselbe Simulation wird nur einmal gerechnet, solange Daten und Modell gleich sind."""
    context = get_context()
    start = context.season_matches["date"].max() + pd.Timedelta(days=1)
    matrix = build_prob_matrix(
        lambda pairs: predict(context, pairs.assign(date=start))[["p_home", "p_draw", "p_away"]],
        context.teams,
    )
    return simulate_season(context.season_matches, matrix, context.teams, n_sims=n_sims, seed=seed)


# --------------------------------------------------------------------------
# Aktuelle Tabelle
# --------------------------------------------------------------------------
with st.expander(f"Aktuelle Tabelle {ctx.season}", expanded=False):
    st.dataframe(compute_table(ctx.season_matches, ctx.teams), hide_index=True, width="stretch")

# --------------------------------------------------------------------------
# Einstellungen und Start
# --------------------------------------------------------------------------
c1, c2 = st.columns([3, 1])
n_sims = c1.select_slider("Anzahl Simulationen", options=[1000, 2000, 5000, 10000, 20000], value=5000,
                          help="Mehr Simulationen = stabilere Prozentwerte, aber längere Rechenzeit.")
seed = c2.number_input("Zufalls-Startwert", min_value=0, value=42, step=1,
                       help="Gleicher Startwert = gleiches Ergebnis (reproduzierbar).")

with st.spinner(f"Simuliere {fmt_int(n_sims)} Saisons …"):
    result = run_simulation(data_version(), model_version(), n_sims, int(seed))

phase = "nach der Teilung" if result.split_done else "vor der Teilung nach 33 Runden"
st.caption(f"{result.remaining_matches} verbleibende Spiele simuliert ({phase}), "
           f"{fmt_int(result.n_sims)} Durchläufe.")

# --------------------------------------------------------------------------
# Ergebnisse
# --------------------------------------------------------------------------
st.subheader("Chancen pro Team")
summary = result.summary.copy()
summary["Ø Schlussrang"] = summary["Ø Schlussrang"].round(1)
summary["Erwartete Punkte"] = summary["Erwartete Punkte"].round(1)
percent_cols = ["Meister", "Obere Gruppe (Top 6)", "Platz 11", "Platz 12"]
summary[percent_cols] = summary[percent_cols] * 100
st.dataframe(
    summary, hide_index=True, width="stretch", height=table_height(len(summary)),
    column_config={
        "Aktuelle Punkte": st.column_config.NumberColumn("Pkt. heute", width="small"),
        "Erwartete Punkte": st.column_config.NumberColumn("Pkt. erwartet", width="small"),
        "Ø Schlussrang": st.column_config.NumberColumn("Ø Rang", width="small"),
        "Meister": pct_column("Meister", "Anteil der Simulationen mit Platz 1"),
        "Obere Gruppe (Top 6)": pct_column("Top 6", "Platz 1–6 nach 33 Runden (obere Gruppe)"),
        "Platz 11": pct_column("Platz 11", "Anteil der Simulationen mit Schlussrang 11"),
        "Platz 12": pct_column("Platz 12", "Anteil der Simulationen mit Schlussrang 12 (letzter Platz)"),
    },
)

# Heatmap: Wahrscheinlichkeit jedes Schlussrangs pro Team
st.subheader("Verteilung der Schlussränge")
probs = result.rank_probs
fig = go.Figure(go.Heatmap(
    z=probs.to_numpy(), x=[str(c) for c in probs.columns], y=list(probs.index),
    colorscale=[[0, "rgba(205,226,251,0.15)"], [0.25, "#86b6ef"], [0.6, "#2a78d6"], [1, "#104281"]],
    zmin=0, zmax=max(0.5, float(probs.to_numpy().max())),
    xgap=2, ygap=2,
    text=[[f"{v:.0%}" if v >= 0.05 else "" for v in row] for row in probs.to_numpy()],
    texttemplate="%{text}", textfont=dict(size=11),
    hovertemplate="%{y}<br>Platz %{x}: %{z:.1%}<extra></extra>",
    colorbar=dict(title="Wahrsch.", tickformat=".0%", thickness=12),
))
fig.update_xaxes(title="Schlussrang", side="top", showgrid=False, dtick=1)
fig.update_yaxes(autorange="reversed", showgrid=False)
st.plotly_chart(style_figure(fig, 120 + 34 * len(probs)), width="stretch")
st.caption("Lesebeispiel: Eine Zelle mit 40 % bedeutet, dass das Team in 40 % der simulierten "
           "Saisons auf diesem Platz landet.")

with st.expander("Annahmen und Vereinfachungen"):
    st.markdown(
        "- Die Wahrscheinlichkeiten jedes Spiels stammen vom ML-Modell mit dem **heutigen** Stand "
        "(Elo, Form, Saisonleistung). Während der Simulation werden sie nicht nachgeführt.\n"
        "- Format: 33 Runden (jedes Paar dreimal), danach Teilung in Plätze 1–6 und 7–12 mit je "
        "5 weiteren Runden. Die Punkte werden mitgenommen.\n"
        "- Das Heimrecht noch nicht angesetzter Spiele wird ausgeglichen geschätzt.\n"
        "- Tore werden nur grob simuliert, damit bei Punktgleichheit die Tordifferenz entscheiden kann.\n"
        "- Was Platz 11 und 12 sportlich bedeuten (direkter Abstieg, Barrage), regelt das aktuelle "
        "Reglement der Swiss Football League."
    )
