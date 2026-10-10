"""
Machine Learning: Training, Evaluation und Prognose.

Zielgrösse: Ausgang eines Spiels – H (Heimsieg), D (Unentschieden), A (Auswärtssieg).

Vorgehen:
  1. Merkmale für alle gespielten Spiele berechnen (src/features.py).
  2. Zeitliche Aufteilung: Training auf allen Saisons VOR der Testsaison,
     Test auf der letzten abgeschlossenen Saison. Nie zufällig mischen –
     sonst «kennt» das Modell die Zukunft.
  3. Drei Modelle vergleichen: Baseline, logistische Regression, Random Forest.
     Messlatte: die Buchmacherquoten.
  4. Das Modell mit dem tiefsten Log-Loss wird auf ALLEN Daten neu trainiert
     und für die Prognosen gespeichert.

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_PATH, RANDOM_STATE, SEASON_LENGTH
from src.elo import EloParams, EloResult
from src.features import FEATURE_COLUMNS, fixture_features, training_table

CLASSES = ["H", "D", "A"]                         # feste Reihenfolge in der ganzen App
CLASS_LABELS = {"H": "Heimsieg", "D": "Unentschieden", "A": "Auswärtssieg"}


class BaselineModel(ClassifierMixin, BaseEstimator):
    """Einfachstes denkbares «Modell»: Es gibt immer die Häufigkeiten aus dem
    Training aus. Da Heimsiege am häufigsten sind, tippt es immer «Heimsieg».
    Ein echtes Modell muss diese Baseline schlagen."""

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        counts = pd.Series(y).value_counts(normalize=True)
        self.frequencies_ = np.array([counts.get(c, 0.0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        return np.tile(self.frequencies_, (len(X), 1))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_models() -> dict[str, BaseEstimator]:
    """Die drei Kandidaten. Die Hyperparameter sind bewusst vorsichtig gewählt,
    weil Fussballdaten verrauscht sind und schnell überangepasst werden."""
    return {
        "Baseline (immer Häufigkeiten)": BaselineModel(),
        "Logistische Regression": make_pipeline(
            StandardScaler(),                               # Merkmale auf gleiche Skala bringen
            LogisticRegression(C=0.5, max_iter=2000),       # C<1 = stärkere Regularisierung
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=25,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


# --------------------------------------------------------------------------
# Hilfsfunktionen für Wahrscheinlichkeiten und Kennzahlen
# --------------------------------------------------------------------------
def proba_frame(estimator, X: pd.DataFrame) -> pd.DataFrame:
    """Wahrscheinlichkeiten immer in der Reihenfolge H, D, A (sklearn sortiert alphabetisch)."""
    raw = estimator.predict_proba(X[FEATURE_COLUMNS])
    df = pd.DataFrame(raw, columns=list(estimator.classes_), index=X.index)
    for c in CLASSES:
        if c not in df:
            df[c] = 0.0
    return df[CLASSES].rename(columns={"H": "p_home", "D": "p_draw", "A": "p_away"})


def bookmaker_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Rechnet Quoten in Wahrscheinlichkeiten um: 1/Quote, normiert auf 100 %
    (die Buchmachermarge wird dabei gleichmässig herausgerechnet)."""
    inv = 1 / df[["odds_home", "odds_draw", "odds_away"]]
    probs = inv.div(inv.sum(axis=1), axis=0)
    probs.columns = ["p_home", "p_draw", "p_away"]
    return probs


def brier_score(y_true: pd.Series, probs: pd.DataFrame) -> float:
    """Brier-Score: mittlere quadrierte Abweichung zwischen Wahrscheinlichkeit
    und tatsächlichem Ausgang (0 = perfekt, tiefer = besser)."""
    onehot = pd.get_dummies(y_true).reindex(columns=CLASSES, fill_value=0).to_numpy(float)
    return float(np.mean(np.sum((probs.to_numpy() - onehot) ** 2, axis=1)))


def log_loss_score(y_true: pd.Series, probs: pd.DataFrame) -> float:
    """Log-Loss: Durchschnitt von −ln(Wahrscheinlichkeit des tatsächlichen Ausgangs).
    Bestraft sichere, aber falsche Prognosen stark (tiefer = besser).
    Eigene Umsetzung, weil sklearn die Spalten alphabetisch (A, D, H) erwartet."""
    column = y_true.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    p_true = probs.to_numpy()[np.arange(len(probs)), column]
    return float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0))))


def evaluate(y_true: pd.Series, probs: pd.DataFrame, name: str) -> dict:
    """Trefferquote, Log-Loss und Brier-Score für ein Modell."""
    predicted = probs.to_numpy().argmax(axis=1)
    predicted = np.array(CLASSES)[predicted]
    return {
        "Modell": name,
        "Trefferquote": accuracy_score(y_true, predicted),
        "Log-Loss": log_loss_score(y_true, probs),
        "Brier-Score": brier_score(y_true, probs),
        "Spiele": len(y_true),
    }


def choose_test_season(table: pd.DataFrame) -> str:
    """Testsaison = die letzte vollständig gespielte Saison."""
    complete = []
    for season, group in table.groupby("season"):
        n_teams = len(set(group["home"]) | set(group["away"]))
        expected = SEASON_LENGTH.get(n_teams, n_teams * (n_teams - 1))
        if len(group) >= expected - 3:          # kleine Toleranz für abgebrochene/fehlende Spiele
            complete.append(season)
    if len(complete) < 3:
        raise ValueError("Zu wenige abgeschlossene Saisons für Training und Test "
                         "(mindestens 3 nötig).")
    return sorted(complete)[-1]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """Alles, was die App vom Training braucht – wird als eine Datei gespeichert."""
    name: str                       # Name des gewählten Modells
    estimator: BaseEstimator        # auf allen Daten trainiertes Modell
    elo_params: dict
    test_season: str
    metrics: pd.DataFrame           # Kennzahlen aller Modelle + Buchmacher (Testsaison)
    test_predictions: pd.DataFrame  # Prognosen der Testsaison (für die Seite «Modellgüte»)
    importance: pd.DataFrame        # Bedeutung der Merkmale im gewählten Modell
    n_train: int
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="minutes"))


def _importance(estimator) -> pd.DataFrame:
    """Merkmalswichtigkeit: beim Random Forest direkt, bei der logistischen
    Regression der mittlere Betrag der (standardisierten) Koeffizienten."""
    if isinstance(estimator, RandomForestClassifier):
        values = estimator.feature_importances_
    else:
        coefs = estimator[-1].coef_                         # Form: (Klassen, Merkmale)
        values = np.abs(coefs).mean(axis=0)
        values = values / values.sum()
    return (pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def train_models(played: pd.DataFrame, params: EloParams | None = None,
                 path: Path | str | None = MODEL_PATH) -> TrainedModel:
    """Trainiert, vergleicht und speichert das beste Modell."""
    params = params or EloParams()
    table, _ = training_table(played, params)

    # Die erste Saison dient nur als «Anlaufphase»: Elo und Form sind dort noch
    # nicht aussagekräftig, darum trainieren wir nicht damit.
    first_season = sorted(table["season"].unique())[0]
    usable = table[table["season"] != first_season]

    test_season = choose_test_season(table)
    train = usable[usable["season"] < test_season]
    test = usable[usable["season"] == test_season]

    results, predictions, fitted = [], test[["date", "season", "home", "away", "result"]].copy(), {}
    for name, model in make_models().items():
        model.fit(train[FEATURE_COLUMNS], train["result"])
        probs = proba_frame(model, test)
        results.append(evaluate(test["result"], probs, name))
        predictions[[f"{name}|{c}" for c in probs.columns]] = probs.to_numpy()
        fitted[name] = model

    # Buchmacher als Messlatte (nur Spiele mit Quoten)
    with_odds = test.dropna(subset=["odds_home", "odds_draw", "odds_away"])
    if len(with_odds) > 0:
        book = bookmaker_probabilities(with_odds)
        results.append(evaluate(with_odds["result"], book, "Buchmacher (Quoten)"))
        for c in book.columns:
            predictions.loc[book.index, f"Buchmacher (Quoten)|{c}"] = book[c]

    metrics = pd.DataFrame(results)

    # Bestes ML-Modell (ohne Baseline/Buchmacher) nach Log-Loss wählen
    candidates = metrics[~metrics["Modell"].str.startswith(("Baseline", "Buchmacher"))]
    best_name = candidates.sort_values("Log-Loss").iloc[0]["Modell"]

    # Endgültiges Modell auf allen verfügbaren Daten (inkl. Testsaison und laufender Saison)
    final = clone(make_models()[best_name]).fit(usable[FEATURE_COLUMNS], usable["result"])

    trained = TrainedModel(
        name=best_name, estimator=final, elo_params=params.to_dict(),
        test_season=test_season, metrics=metrics, test_predictions=predictions,
        importance=_importance(final), n_train=len(usable),
    )
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained, path)
    return trained


def load_model(path: Path | str = MODEL_PATH) -> TrainedModel | None:
    """Lädt das gespeicherte Modell (oder None, falls noch keines trainiert wurde)."""
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)


# --------------------------------------------------------------------------
# Prognose für kommende Spiele
# --------------------------------------------------------------------------
def predict_fixtures(model: TrainedModel, played: pd.DataFrame, fixtures: pd.DataFrame,
                     elo: EloResult) -> pd.DataFrame:
    """Gibt für jedes kommende Spiel Merkmale, Wahrscheinlichkeiten und Tipp zurück."""
    if fixtures.empty:
        return fixtures.assign(p_home=[], p_draw=[], p_away=[], tip=[])
    fixtures = fixtures.copy()
    if "season" not in fixtures:
        fixtures["season"] = current_season(played)
    features = fixture_features(played, fixtures, elo)
    probs = proba_frame(model.estimator, features)
    out = pd.concat([fixtures, features, probs], axis=1)
    out["tip"] = np.array(CLASSES)[probs.to_numpy().argmax(axis=1)]
    return out


def current_season(played: pd.DataFrame) -> str:
    """Die laufende Saison = die Saison des letzten gespielten Spiels."""
    return str(played.sort_values("date")["season"].iloc[-1])
