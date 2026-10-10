"""
Einmalige Einrichtung über die Kommandozeile: Daten laden und Modell trainieren.
(Dasselbe geht auch in der App auf der Seite «Daten & Einstellungen».)

Aufrufe:
    python scripts/setup_data.py                 # echte Daten herunterladen
    python scripts/setup_data.py --csv pfad.csv  # bereits heruntergeladene CSV verwenden
    python scripts/setup_data.py --sample        # erfundene Beispieldaten (nur zum Testen)

KI-Hinweis (gemäss HSG-Regeln zur KI-Nutzung): Diese Datei wurde mit Claude
(Anthropic, Modell «Claude Opus 5.5», Chat vom 10.10.2026) generiert.
Prompt der Gruppe: «Erstelle den Programmcode für die Super-League-
Matchprognose (Streamlit-App) gemäss unserem Projektkonzept.»
Geprüft und angepasst durch: [Namen der Gruppe ergänzen]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import database as db  # noqa: E402
from src.data_loader import download_csv, import_csv_to_db  # noqa: E402
from src.model import train_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Daten laden und Modell trainieren")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--csv", type=Path, help="Pfad zu einer bereits heruntergeladenen CSV")
    group.add_argument("--sample", action="store_true", help="erfundene Beispieldaten verwenden")
    args = parser.parse_args()

    if args.sample:
        from scripts.make_sample_data import write_sample_csv
        n = import_csv_to_db(write_sample_csv(), data_source="beispieldaten")
        print(f"Beispieldaten importiert: {n} Spiele (ERFUNDEN, nur zum Testen)")
    else:
        path = args.csv or download_csv()
        n = import_csv_to_db(path)
        print(f"Daten importiert: {n} Spiele aus {path}")

    model = train_models(db.load_matches())
    print(f"\nTestsaison: {model.test_season}")
    print(model.metrics.round(3).to_string(index=False))
    print(f"\nGewähltes Modell: {model.name} (trainiert auf {model.n_train} Spielen)")


if __name__ == "__main__":
    main()
