# Super-League-Prognose ⚽

**Wer gewinnt am Wochenende?** Eine Streamlit-App, die für jedes Spiel der Schweizer
Super League die Wahrscheinlichkeiten für Heimsieg, Unentschieden und Auswärtssieg
berechnet – als datenbasierte Zweitmeinung für Büro- und Vereins-Tippspiele.

Gruppenprojekt im Kurs «Fundamentals and Methods of Computer Science», Universität St. Gallen, HS 2026.

---

## Installation und Start

Voraussetzung: Python 3.10 oder neuer.

```bash
# 1. In den Projektordner wechseln
cd superleague-prognose

# 2. Virtuelle Umgebung anlegen und aktivieren (empfohlen)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Pakete installieren
pip install -r requirements.txt

# 4. App starten – öffnet sich im Browser unter http://localhost:8501
streamlit run app.py
```

### Daten laden (einmalig)

In der App auf der Seite **«Daten & Einstellungen»**:

1. **Resultate herunterladen** – lädt alle Super-League-Spiele inkl. Wettquoten von
   [football-data.co.uk](https://www.football-data.co.uk/new/SWZ.csv) in die SQLite-Datenbank.
   Falls der Download blockiert ist: Datei im Browser speichern und unter «CSV hochladen» importieren.
2. **Modell trainieren** – dauert wenige Sekunden.
3. **Kommende Spiele erfassen** – über API-Football (optional) oder von Hand.

Alternativ über die Kommandozeile:

```bash
python scripts/setup_data.py                # herunterladen + trainieren
python scripts/setup_data.py --csv SWZ.csv  # bereits heruntergeladene Datei verwenden
```

### API-Football (optional)

Gratis-Schlüssel auf [api-football.com](https://www.api-football.com) (100 Anfragen pro Tag).
Den Schlüssel entweder in der App eingeben oder `.streamlit/secrets.toml.example` nach
`.streamlit/secrets.toml` kopieren und dort eintragen. **Die Datei `secrets.toml` nie abgeben.**
Antworten werden 12 Stunden in der Datenbank zwischengespeichert. Je nach Plan sind nicht alle
Saisons freigeschaltet; die App zeigt dann die Fehlermeldung der API. Ohne Schlüssel funktioniert
alles mit von Hand erfassten Spielen.

### Testen ohne Internet

`python scripts/setup_data.py --sample` erzeugt **erfundene** Teams und Resultate. Die App zeigt
dann einen gelben Warnhinweis. Nur zum Testen – nicht für Video oder Präsentation verwenden.

Automatische Tests der Kernlogik: `python -m pytest`

---

## Aufbau

```
app.py                  Startpunkt: Navigation und Seitenleiste
config.py               alle Einstellungen (Pfade, Elo-Parameter, Liga-Format)
src/
  database.py           SQLite: Spiele, kommende Spiele, Tipps, API-Cache
  data_loader.py        CSV-Download und -Import, API-Football, Teamnamen-Abgleich
  elo.py                eigenes Elo-Rating
  standings.py          Tabellenberechnung
  features.py           Merkmale für das ML-Modell (ohne Datenleck)
  model.py              Training, Evaluation, Prognose
  simulation.py         Monte-Carlo-Simulation der Restsaison
views/                  die sechs Seiten der App
scripts/                Einrichtung über die Kommandozeile, Beispieldaten
tests/                  automatische Tests
data/                   Datenbank, CSV und Modell (wird automatisch angelegt)
```

## Seiten der App

| Seite | Was sie zeigt | Interaktion |
|---|---|---|
| Nächster Spieltag | Wahrscheinlichkeiten aller kommenden Spiele | Zeitraum wählen, Spiel zur Analyse öffnen |
| Match-Analyse | Prognose zweier Teams und ihre Begründung, Elo-Verlauf, Formkurve, Direktvergleich | Teams und Zeitraum wählen |
| Saison-Simulation | Chancen auf Meistertitel, obere Gruppe, Platz 11 und 12 | Anzahl Simulationen, Zufalls-Startwert |
| Tippspiel | Eigene Tipps gegen das Modell, Rangliste | Tipps abgeben (werden in der DB gespeichert) |
| Wie gut ist das Modell? | Trefferquote, Log-Loss, Kalibrierung, Merkmalswichtigkeit | – |
| Daten & Einstellungen | Daten laden, Modell trainieren, Spiele erfassen | Elo-Parameter einstellen und neu trainieren |

## Abdeckung der Bewertungskriterien

| # | Anforderung | Umsetzung |
|---|---|---|
| 1 | Problemstellung | Tippspiel-Helfer für Fans: datenbasierte Zweitmeinung statt Bauchgefühl |
| 2 | Daten über API/Datenbank | CSV-Download in SQLite; kommende Spiele über API-Football mit Cache |
| 3 | Visualisierung | Wahrscheinlichkeitsbalken, Elo-Verläufe, Rang-Heatmap, Kalibrierung, Verwechslungsmatrix |
| 4 | Interaktion | Teamwahl, Tipps speichern, Simulation starten, Elo-Parameter ändern, Spiele erfassen |
| 5 | Machine Learning | Logistische Regression und Random Forest, gegen Baseline und Buchmacher getestet |
| 6 | Kommentare | Docstrings und Kommentare in jeder Datei |
| 7 | Beiträge + GenAI | im Video (siehe Checkliste unten) |
| 8 | Video | 4 Minuten, eigene Stimme (siehe Checkliste unten) |

## Machine Learning in Kürze

- **Zielgrösse:** Heimsieg (H), Unentschieden (D) oder Auswärtssieg (A).
- **Merkmale** (nur aus Spielen *vor* dem Anpfiff): Elo beider Teams und Differenz, Form der
  letzten 5 Spiele, Heim- bzw. Auswärtsstärke der Saison, Saisonleistung, Direktvergleich.
  Ein Test (`test_no_leakage`) prüft, dass kein Merkmal von späteren Resultaten abhängt.
- **Zeitliche Aufteilung:** Training auf allen Saisons vor der letzten abgeschlossenen, Test auf
  dieser. Die erste Saison dient nur als Anlaufphase für Elo und Form.
- **Vergleich:** Baseline (immer Häufigkeiten), logistische Regression, Random Forest, Buchmacher.
  Gewählt wird das Modell mit dem tiefsten Log-Loss; danach wird es auf allen Daten neu trainiert.
- **Erwartung:** Bei drei Ausgängen sind rund 50 % Trefferquote im Fussball gut. Unentschieden
  sagt kaum ein Modell zuverlässig voraus.

## Bekannte Grenzen

- Keine Informationen zu Verletzungen, Sperren oder Aufstellungen.
- Die Saison-Simulation hält die Wahrscheinlichkeiten über die Restsaison konstant, schätzt das
  Heimrecht noch nicht angesetzter Spiele und simuliert Tore nur grob.
- Die Teilung nach 33 Runden wird über die ersten 198 Spiele der Saison erkannt; stark
  verschobene Spiele können das verfälschen.
- Teamnamen von API-Football werden unscharf zugeordnet; nicht erkannte Namen meldet die App.

---

## Nutzung generativer KI

Der Programmcode wurde mit **Claude (Anthropic, Modell «Claude Opus 5.5»)** generiert, Chat vom
10.10.2026. Gemäss den Kursregeln ist das in **jeder Quelldatei** im Kopfkommentar zitiert.
Bitte in jedem Kopfkommentar die Zeile «Geprüft und angepasst durch» mit euren Namen ergänzen
und die genaue Zitierform mit den HSG-Richtlinien abgleichen (Link auf Folie 4 der Projektvorgaben).

## Checkliste vor der Abgabe (10.12.2026, 23:59, Canvas)

- [ ] Echte Daten geladen, Modell trainiert, Zahlen auf «Wie gut ist das Modell?» geprüft
- [ ] Code von allen gelesen und verstanden – in der Q&A am 11.12. wird nachgefragt
- [ ] Kopfkommentare ergänzt («Geprüft und angepasst durch»)
- [ ] Video (max. 4 Minuten) mit **eigener Stimme**: Problem, Demo, Reflexion Teamarbeit,
      Beitragsmatrix, Liste der Hilfsmittel inkl. KI-Nutzung
- [ ] Beitragsmatrix gibt wieder, wer tatsächlich was gemacht hat
- [ ] Eigenständigkeitserklärung (eine pro Gruppe)
- [ ] Abgabe als Dateien hochladen (ZIP mit Code inkl. `data/`-Ordner, Video, Erklärung) –
      Links auf GitHub oder YouTube genügen nicht
- [ ] `.streamlit/secrets.toml` **nicht** mit abgeben
