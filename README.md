# SaxSSL Bodenschätzung

Bodeneigenschaften aus MIR-Spektren (Bruker Alpha II, DRIFT) schätzen.
OPUS-Dateien (`.0`, `.1`, …) hochladen → eine Tabelle pro Probe mit Schätzungen der
sächsischen Spektralbibliothek (**SaxSSL + Hannah**), optional **Torfmodellen (pmird)**
und **OSSL (KSSL, USA)** zum Vergleich. Jede Schätzung hat ein ~68-%-Intervall, eine
Güteklasse und eine Prüfung, ob das Spektrum zur Bibliothek passt (⚠). Download als CSV/Excel.

## Installation und Start

Python 3.12 empfohlen.

```bash
git clone <repo-url> saxssl-bodenschaetzung
cd saxssl-bodenschaetzung
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Der Browser öffnet sich unter <http://localhost:8501>. Zum Ausprobieren die Dateien
aus `beispiele/` hochladen (2 Proben × 4 Wiederholungen).

## Bedienung

- **Wiederholungsmessungen** `Probe7-1.0`, `Probe7-2.0`, … werden zur Probe `Probe7`
  gemittelt (Endung und `-<1–2 Ziffern>` samt allem danach werden abgeschnitten;
  Unterstriche bleiben: `Profil_2.0` → `Profil_2`). Die Zuordnung Datei → Probe wird
  angezeigt und mit exportiert.
- **Seitenleiste:** Eigenschaftsgruppen, 🌿 Torfmodelle (aus; die App schlägt sie vor,
  wenn Corg > 12 % geschätzt wird), OSSL ein/aus, Klasse-D-Modelle ein/aus, Intervall ein/aus.

## Modelle und Güte

Alle Kennzahlen: `models/library/metrics.csv`. Güte nach RPIQ auf einem
Kennard-Stone-Testset:

| Klasse | SaxSSL + Hannah (Mineralböden) |
|---|---|
| A (RPIQ ≥ 2,5) | Corg, Ct, Nt, TOC, TOC400, ROC, Ton, Schluff, Sand, pH (CaCl₂), Fe_d, P (Königswasser) |
| B (≥ 1,9) | KAK pot., Fe_o, Trockenrohdichte, Cr, Ni, Zn (Königswasser) |
| C (≥ 1,4, Screening) | C/N, TIC900, KAK eff., P₂O₅ (CAL), Mg (CaCl₂), Cu, Benzo(a)pyren, PAK16 |
| D (ausgeblendet) | CaCO₃, K₂O (CAL), K/Mg/As/Cd/Hg/Pb (Königswasser) |

Torf (pmird): Nt und C/N Klasse A. OSSL-KSSL: 11 Modelle (Corg, Ct, Nt, Ton, Schluff,
Sand (USDA-Grenze 50 µm!), pH, KAK, CaCO₃, Fe_d, Trockenrohdichte).

Schwermetalle und PAK haben kein eigenes MIR-Signal; sie werden indirekt über
Trägerphasen (organische Substanz, Ton, Fe-Oxide) geschätzt. Diese Werte eignen sich
zum Screening, nicht als Analysenersatz.

## Aufbau

| Pfad | Inhalt |
|---|---|
| `app.py`, `tables.py` | Streamlit-App und Tabellenfunktionen |
| `soilspec/` | minimaler Laufzeit-Ausschnitt des `soilspec`-Pakets (OPUS lesen, Vorverarbeitung, Vorhersage) |
| `models/library/` | SaxSSL+Hannah- und Torfmodelle, `manifest.json`, `metrics.csv` |
| `models/ossl/` | OSSL-KSSL-Modelle |
| `beispiele/` | Beispielspektren aus der SaxSSL |
| `tests/test_app.py` | End-to-End-Test (`pip install pytest && pytest`) |

Die Versionen in `requirements.txt` sind fest, weil die Modelle mit genau diesen
Versionen gespeichert wurden.

## Modelle aktualisieren (Betreuende)

Trainiert wird im vollständigen Paket (`Hannah_Spec/soilspec`,
`scripts/rebuild_models.py`). Danach die Modelle hierher kopieren:

```bash
rsync -a --delete --exclude 'dataset_*' ../Hannah_Spec/models/library ../Hannah_Spec/models/ossl models/
pytest
```

Bei Paket-Updates auch `soilspec/` abgleichen (gleiche Dateien wie im vollständigen
Paket) und die Versionen in `requirements.txt` prüfen.
