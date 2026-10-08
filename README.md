# 🌱 SaxSSL Bodenschätzung

**Bodeneigenschaften aus MIR-Spektren – in Sekunden statt Laborwochen.**

OPUS-Dateien eines Bruker Alpha II (DRIFT) hochladen, Tabelle mit geschätzten
Bodeneigenschaften herunterladen. Grundlage sind die sächsischen Bodenspektralbibliotheken
**SaxSSL** und **SaxTox**, zum Vergleich daneben die **OSSL/KSSL**.

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Was die App kann

- 📂 **Beliebig viele `.0`-Dateien** hochladen; Wiederholungsmessungen (`Probe7-1.0`, `Probe7-2.0`, …) werden automatisch gemittelt
- 📊 **Rund 30 Eigenschaften**: Kohlenstoff & Stickstoff, Textur, pH, KAK, Eisenoxide, Nährstoffe, Schwermetalle, PAK
- 🎯 **Ehrliche Unsicherheit**: ~68-%-Intervall aus Kreuzvalidierungsfehlern, breiter für untypische Spektren; Güteklasse 🟢🟡🟠🔴 je Eigenschaft
- ⚠️ **Ähnlichkeit zur Bibliothek**: Ampel je Probe ganz oben in der Ergebnistabelle
- 🌿 **Torfmodus** für organische Proben (wird ab geschätztem Corg > 12 % vorgeschlagen)
- 🌍 **OSSL/KSSL-Vergleich**: Schätzungen des US-Modells daneben
- 💾 **Export** als CSV oder Excel

## Schnellstart

```bash
git clone https://github.com/cojacoo/SaxSSL_Models.git
cd SaxSSL_Models
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Zum Ausprobieren: die Dateien aus [`beispiele/`](beispiele) hochladen (2 Proben × 4 Wiederholungen).

## Modellgüte

Klassen nach RPIQ auf einem unabhängigen Testset · alle Kennzahlen in [`models/library/metrics.csv`](models/library/metrics.csv)

| | Mineralböden (SaxSSL + SaxTox) |
|:-:|---|
| 🟢 **A** | Corg, Ct, Nt, TOC, TOC400, ROC · Ton, Schluff, Sand · pH (CaCl₂) · Fe_d · P (Königswasser) |
| 🟡 **B** | KAK pot. · Fe_o · Trockenrohdichte · Cr, Ni, Zn |
| 🟠 **C** | C/N · TIC900 · KAK eff. · P₂O₅ (CAL) · Mg (CaCl₂) · Cu · Benzo(a)pyren, PAK16 |
| 🔴 **D** | ausgeblendet: CaCO₃, K₂O (CAL), K, Mg, As, Cd, Hg, Pb (Königswasser) |

**Torf** (pmird): Nt, C/N (A) · **OSSL/KSSL**: 11 Modelle (Achtung: USDA-Sandgrenze 50 µm)

> **Gut zu wissen:** Schwermetalle und PAK haben kein eigenes MIR-Signal. Sie werden
> indirekt über organische Substanz, Ton und Eisenoxide geschätzt – gut fürs Screening,
> kein Ersatz für die Laboranalyse.

## Daten

| Quelle | Inhalt |
|---|---|
| **SaxSSL** – Adam, Julich, Benning & Jackisch (2026), [doi:10.1594/PANGAEA.984699](https://doi.org/10.1594/PANGAEA.984699) | Sächsische Boden-Dauerbeobachtung, Alpha II DRIFT + Labordaten |
| **SaxTox** – FIS Boden Sachsen, MSc-Arbeit TU Bergakademie Freiberg | Rückstellproben belasteter Auenböden mit Schwermetallen und PAK |
| **pmird** – Teickner & Knorr (2026), [doi:10.5194/soil-12-497-2026](https://doi.org/10.5194/soil-12-497-2026) | Torf-MIR-Datenbank |
| **OSSL** – [soilspectroscopy.org](https://soilspectroscopy.org) | KSSL-Ausschnitt der Open Soil Spectral Library |

## Aufbau

```
app.py · app_tables.py Streamlit-App
soilspec/              schlanker Laufzeitkern (OPUS lesen, Vorverarbeitung, Vorhersage)
models/library/        SaxSSL+SaxTox- und Torfmodelle, manifest.json, metrics.csv
models/ossl/           OSSL/KSSL-Modelle
beispiele/             Beispielspektren
tests/                 End-to-End-Test  →  pip install pytest && pytest
tools/strip_models.py  entfernt Probendaten aus den Modelldateien (vor jedem Commit)
```

<details>
<summary><b>Für Betreuende: Modelle aktualisieren</b></summary>

Trainiert wird im vollständigen `soilspec`-Paket (`scripts/rebuild_models.py`), dann:

```bash
SRC=<Pfad zum soilspec-Projekt>/models
rsync -a --delete --exclude 'dataset_*' --exclude metrics_detail.json $SRC/library $SRC/ossl models/
python tools/strip_models.py   # Testset-Daten (Proben-IDs, Laborwerte) aus den Modellen entfernen
pytest
```

Die festen Versionen in `requirements.txt` gehören zu den gespeicherten Modellen –
nur zusammen mit einem Neutraining ändern.
</details>

---

MIT-Lizenz · TU Bergakademie Freiberg
