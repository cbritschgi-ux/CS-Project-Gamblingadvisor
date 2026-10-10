"""
Seite «Wie gut ist das Modell?»: ehrliche Auswertung auf einer Saison, die das
Modell beim Training nicht gesehen hat.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import FEATURE_LABELS
from src.model import CLASSES
from views.common import (COLOR_AWAY, COLOR_DRAW, COLOR_HOME, OUTCOME_TEXT,
                          fmt_int, require_context, style_figure)

st.title("Wie gut ist das Modell?")
ctx = require_context()
model = ctx.model
metrics = model.metrics.copy()
preds = model.test_predictions

st.markdown(
    f"Getestet auf der Saison **{model.test_season}**, die beim Training ausgeschlossen war "
    f"(zeitliche Aufteilung, kein Blick in die Zukunft). Gewählt wurde das Modell mit dem "
    f"tiefsten Log-Loss: **{model.name}**. Für die Prognosen wurde es danach auf allen "
    f"{fmt_int(model.n_train)} Spielen neu trainiert."
)

# --------------------------------------------------------------------------
# Kennzahlen
# --------------------------------------------------------------------------
st.subheader("Kennzahlen auf der Testsaison")
st.dataframe(
    metrics, hide_index=True, width="stretch",
    column_config={
        "Trefferquote": st.column_config.NumberColumn(format="percent",
                                                      help="Anteil richtig vorhergesagter Ausgänge (höher = besser)"),
        "Log-Loss": st.column_config.NumberColumn(format="%.3f",
                                                  help="Bestraft sichere, aber falsche Prognosen (tiefer = besser)"),
        "Brier-Score": st.column_config.NumberColumn(format="%.3f",
                                                     help="Abstand zwischen Wahrscheinlichkeit und Ausgang (tiefer = besser)"),
    },
)

best = metrics.set_index("Modell")
baseline_name = next(n for n in best.index if n.startswith("Baseline"))
text = (f"Trefferquote {best.at[model.name, 'Trefferquote']:.1%} gegenüber "
        f"{best.at[baseline_name, 'Trefferquote']:.1%} der Baseline")
if "Buchmacher (Quoten)" in best.index:
    text += f" und {best.at['Buchmacher (Quoten)', 'Trefferquote']:.1%} der Buchmacher"
st.markdown(f"**Ergebnis:** {text}.")
st.caption("Einordnung: Bei drei möglichen Ausgängen sind rund 50 % Trefferquote im Fussball gut. "
           "Die Buchmacher sind die härteste Messlatte, weil ihre Quoten viel mehr Information "
           "enthalten (Verletzungen, Aufstellungen, Marktwissen).")

# Trefferquote und Log-Loss in zwei getrennten Grafiken (nie zwei Skalen in einem Diagramm).
# Trefferquote als Balken ab 0 %; Log-Loss als Punkte, weil die Unterschiede klein sind
# und ein Balken ab 0 sie unsichtbar machen würde.
colors = [COLOR_HOME if m == model.name else "#9a9893" for m in metrics["Modell"]]
c1, c2 = st.columns(2)
fig = go.Figure(go.Bar(
    x=metrics["Trefferquote"], y=metrics["Modell"], orientation="h", marker=dict(color=colors),
    text=[f"{v:.1%}" for v in metrics["Trefferquote"]], textposition="outside",
    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(tickformat=".0%", range=[0, max(0.7, metrics["Trefferquote"].max() * 1.2)])
c1.markdown("**Trefferquote** (höher = besser)")
c1.plotly_chart(style_figure(fig, 230), width="stretch")

span = metrics["Log-Loss"].max() - metrics["Log-Loss"].min()
fig = go.Figure(go.Scatter(
    x=metrics["Log-Loss"], y=metrics["Modell"], mode="markers+text",
    marker=dict(color=colors, size=12), text=[f"{v:.3f}" for v in metrics["Log-Loss"]],
    textposition="middle right", hovertemplate="%{y}: %{x:.3f}<extra></extra>",
))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(range=[metrics["Log-Loss"].min() - span * 0.3 - 0.005,
                        metrics["Log-Loss"].max() + span * 0.6 + 0.01], tickformat=".2f",
                 showgrid=True, gridcolor="rgba(128,128,128,0.18)")
c2.markdown("**Log-Loss** (tiefer = besser)")
c2.plotly_chart(style_figure(fig, 230), width="stretch")

# --------------------------------------------------------------------------
# Verlauf über die Testsaison
# --------------------------------------------------------------------------
st.subheader("Trefferquote im Saisonverlauf")
order = preds.sort_values("date")
fig = go.Figure()
series = [(model.name, COLOR_HOME), (baseline_name, COLOR_DRAW)]
if "Buchmacher (Quoten)|p_home" in preds.columns:
    series.append(("Buchmacher (Quoten)", COLOR_AWAY))
for name, color in series:
    probs = order[[f"{name}|p_home", f"{name}|p_draw", f"{name}|p_away"]]
    valid = probs.notna().all(axis=1)
    picked = np.array(CLASSES)[probs[valid].to_numpy().argmax(axis=1)]
    hits = pd.Series(picked == order.loc[valid, "result"].to_numpy(), index=order.index[valid])
    cumulative = hits.expanding().mean()
    fig.add_scatter(x=order.loc[valid, "date"], y=cumulative, mode="lines", name=name,
                    line=dict(color=color, width=2),
                    hovertemplate="%{x|%d.%m.%Y}<br>" + name + ": %{y:.1%}<extra></extra>")
fig.update_layout(hovermode="x unified")
fig.update_yaxes(tickformat=".0%", title="kumulierte Trefferquote")
st.plotly_chart(style_figure(fig, 320), width="stretch")
st.caption("Zu Saisonbeginn schwankt die Kurve stark, weil erst wenige Spiele gezählt sind.")

# --------------------------------------------------------------------------
# Kalibrierung und Verwechslungsmatrix
# --------------------------------------------------------------------------
st.subheader("Sind die Prozentwerte ehrlich?")
k1, k2 = st.columns(2)

probs = preds[[f"{model.name}|p_home", f"{model.name}|p_draw", f"{model.name}|p_away"]].to_numpy()
picked_prob = probs.max(axis=1)
picked = np.array(CLASSES)[probs.argmax(axis=1)]
hit = picked == preds["result"].to_numpy()
bins = pd.cut(picked_prob, bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0])
calib = (pd.DataFrame({"bin": bins, "p": picked_prob, "hit": hit})
         .groupby("bin", observed=True).agg(prognose=("p", "mean"), treffer=("hit", "mean"), n=("hit", "size"))
         .reset_index())
calib = calib[calib["n"] >= 10]          # Gruppen mit weniger als 10 Spielen sind zu zufällig

fig = go.Figure()
fig.add_scatter(x=[0.3, 0.9], y=[0.3, 0.9], mode="lines", name="perfekt kalibriert",
                line=dict(color="#b8b6b0", width=1.5, dash="dash"), hoverinfo="skip")
fig.add_scatter(x=calib["prognose"], y=calib["treffer"], mode="lines+markers", name=model.name,
                line=dict(color=COLOR_HOME, width=2), marker=dict(size=9),
                customdata=calib["n"],
                hovertemplate="Prognose Ø %{x:.0%}<br>tatsächlich %{y:.0%}<br>%{customdata} Spiele<extra></extra>")
fig.update_xaxes(tickformat=".0%", title="Wahrscheinlichkeit des Tipps", range=[0.3, 0.9])
fig.update_yaxes(tickformat=".0%", title="tatsächlich eingetroffen", range=[0, 1])
k1.markdown("**Kalibrierung**")
k1.plotly_chart(style_figure(fig, 320), width="stretch")
k1.caption("Liegen die Punkte nahe der Diagonale, stimmen die Prozentwerte: "
           "Von Tipps mit 60 % trifft etwa jeder sechste von zehn ein.")

confusion = pd.crosstab(pd.Series(preds["result"].to_numpy(), name="Tatsächlich"),
                        pd.Series(picked, name="Tipp")).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
labels = [OUTCOME_TEXT[c] for c in CLASSES]
fig = go.Figure(go.Heatmap(
    z=confusion.to_numpy(), x=labels, y=labels, xgap=2, ygap=2,
    colorscale=[[0, "rgba(205,226,251,0.15)"], [0.5, "#5598e7"], [1, "#104281"]],
    text=confusion.to_numpy(), texttemplate="%{text}", showscale=False,
    hovertemplate="Tatsächlich %{y}<br>Tipp %{x}<br>%{z} Spiele<extra></extra>",
))
fig.update_xaxes(title="Tipp des Modells", side="top", showgrid=False)
fig.update_yaxes(title="Tatsächlicher Ausgang", autorange="reversed", showgrid=False)
k2.markdown("**Verwechslungsmatrix**")
k2.plotly_chart(style_figure(fig, 320), width="stretch")
k2.caption("Die Spalte «Unentschieden» ist fast leer: Ein Remis ist selten der wahrscheinlichste "
           "Ausgang, darum tippt das Modell es kaum.")

# --------------------------------------------------------------------------
# Merkmalswichtigkeit
# --------------------------------------------------------------------------
st.subheader("Welche Merkmale zählen am meisten?")
imp = model.importance.assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
fig = go.Figure(go.Bar(
    x=imp["importance"], y=imp["label"], orientation="h", marker=dict(color=COLOR_HOME),
    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(tickformat=".0%", title="relativer Einfluss")
st.plotly_chart(style_figure(fig, 60 + 30 * len(imp)), width="stretch")
st.caption("Random Forest: Beitrag zur Verbesserung der Entscheidungsbäume. Logistische Regression: "
           "Betrag der Koeffizienten auf standardisierten Merkmalen. Merkmale hängen teilweise "
           "zusammen (z. B. Elo und Saisonleistung), daher die Werte nicht zu wörtlich nehmen.")
