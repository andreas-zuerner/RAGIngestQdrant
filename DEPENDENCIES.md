# Dependencies & Usage Guide

This document summarizes the runtime dependencies of the project and shows how
to prepare a local environment for running the scanner and worker components.

## Python dependencies

Install the following packages into your virtual environment:

```
requests
chardet
beautifulsoup4
lxml
python-docx
PyPDF2
pdf2image  # optional, used when PDF OCR is enabled
Pillow      # required by pdf2image
docling     # zentrale Extraktion & Hybrid-Chunking Engine
```

You can install everything in one step with:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r <(printf "requests\nchardet\nbeautifulsoup4\nlxml\npython-docx\nPyPDF2\npdf2image\nPillow\ndocling\n")
```

> **Tip:** `pdf2image` requires the `poppler` utilities to be installed on the
> host system. OCR support additionally relies on `tesseract-ocr` being
> available on the `$PATH`.

## System packages for PDF/OCR support

The improved PDF pipeline combines multiple backends. To get consistent
results, make sure the following system tools are installed:

- `pdftotext` (part of the `poppler-utils` package on most Linux distributions,
  available via `brew install poppler` on macOS)
- `tesseract-ocr` plus language packs relevant to your documents (e.g.
  `tesseract-ocr-deu` on Debian/Ubuntu)
- `ghostscript` (indirect dependency of `pdf2image` on some platforms)

Example installation commands:

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-deu ghostscript

# macOS (Homebrew)
brew install poppler tesseract
```

If these tools are missing, the worker automatically skips the corresponding
fallbacks (e.g. OCR) instead of crashing, but scanned PDFs may stay empty.

To tweak performance you can set the following environment variables:

- `PDF_MAX_PAGES` &ndash; limit how many pages PyPDF2 parses (default `2000`)
- `PDF_OCR_MAX_PAGES` &ndash; cap the number of pages rendered for OCR (default
  `20`)
- `PDF_OCR_DPI` &ndash; DPI for OCR rendering (default `300`)
- `PDFTOTEXT_TIMEOUT_S` &ndash; timeout for the `pdftotext` subprocess (default
  `60` seconds)

Enable the OCR fallback by setting `ENABLE_OCR=1` in your environment or
`.env.local`. When `docling` is installed (see the upstream project at
https://docling-project.github.io/docling/), the ingest worker automatically
uses its unified parsing pipeline (including hybrid chunking, layout-aware
splitting and optional OCR support) instead of the custom heuristics that were
previously bundled with this repository.

## Project layout & test data

- `DocumentDatabase/` → location where the SQLite database (`state.db`)
  is created at runtime.
- `ct109-data/scan_root/` → sample `.txt` files for local testing of the
  scanner/worker integration.

## Configuration

Environment variables reside in `.env.local`. A full example (`.env.local.example`)
contains all keys (database path, scan roots, Brain/Ollama endpoints). Copy and
adjust it for your environment:

```bash
cp .env.local.example .env.local
# anschließend DB_PATH, BRAIN_API_KEY usw. konfigurieren
```

### Ingestion overview & key environment variables

* **Input discovery (Nextcloud):** `NEXTCLOUD_DOC_DIR` ist der Scan-Wurzelpfad,
  `EXCLUDE_GLOBS` blendet Muster aus. Der Scheduler legt pro Lauf maximal
  `MAX_JOBS_PER_PASS` Jobs an und schläft `SLEEP_SECS` Sekunden dazwischen.
* **docling-serve:** `DOCLING_SERVE_URL` (Sync), `DOCLING_SERVE_USE_ASYNC`
  aktiviert die Async-Pipeline, optional überschrieben durch
  `DOCLING_SERVE_ASYNC_URL` mit Timeouts über
  `DOCLING_SERVE_TIMEOUT`/`DOCLING_SERVE_ASYNC_TIMEOUT` und Poll-Intervallen
  via `DOCLING_SERVE_ASYNC_POLL_INTERVAL`. Die parallel laufenden
  docling-Aufträge werden per `MAX_WORKERS` (setzt intern `DOCLING_MAX_WORKERS`)
  begrenzt; die nachgelagerte Pipeline nutzt `PIPELINE_WORKERS`.
* **Dateirouting:** Erweiterte Dateitypen werden über `.env.local` gesteuert:

  | Kategorie | Erweiterbar über | Umschalter | Route |
  | --- | --- | --- | --- |
  | Standardformate | `FILE_TYPES_STANDARD` | – | Direkt zu docling-serve. |
  | Office/ODF | `FILE_TYPES_SOFFICE` | `ENABLE_SOFFICE_TYPES` | Vorab-Konvertierung mit `soffice`, anschließend docling-serve. |
  | Tabellen | `FILE_TYPES_TABLE` | – | Tabellen werden zusätzlich geparsed und in SQLite gespeichert; Relevanzprüfung wird übersprungen. |
  | MS-Extended | `FILE_TYPES_MS_EXTENDED` | `ENABLE_MS_EXTENDED_TYPES` | Gehen durch die Text-Fallback-Pipeline, falls docling sie nicht unterstützt. |
  | Audio/Video | `FILE_TYPES_AUDIO` | `ENABLE_AUDIO_TYPES` | Nicht für docling vorgesehen; best-effort Text-Fallback. |

* **Relevanz & Chunking:** Relevanz läuft über `OLLAMA_HOST` +
  `OLLAMA_MODEL_RELEVANCE` mit Schwelle `RELEVANCE_THRESHOLD`. Chunking nutzt
  `OLLAMA_MODEL_CHUNKING`, `BRAIN_CHUNK_TOKENS`, `OVERLAP`, `MAX_CHUNKS`.
  Kontextanreicherung erfolgt via `OLLAMA_MODEL_CONTEXT`.
* **Brain-Ingest:** `BRAIN_URL`, `BRAIN_API_KEY`, `BRAIN_COLLECTION`,
  Timeout `BRAIN_REQUEST_TIMEOUT`.
* **Persistenz & Storage:** `DB_PATH` zeigt auf die SQLite-Datenbank
  (`DocumentDatabase/state.db` per Default) mit den Tabellen `files` (alle
  gescannten Pfade), `jobs` (Queue), `decision_log` (Schrittprotokoll) sowie
  `table_registry`/`table_data` für extrahierte Tabellen. Vektorisierte Texte
  landen via Brain-Service in der Qdrant-Collection `BRAIN_COLLECTION`; Bilder
  werden in `NEXTCLOUD_IMAGE_DIR` abgelegt.

## Running the services

1. Activate the virtual environment and ensure dependencies are installed.
2. Copy and edit `.env.local` as described above.
3. Start scanner and worker together via the control script:

   ```bash
   ./brain_scan.sh start
   ```

   Der Scheduler legt Logs unter `logs/scan_scheduler.log` ab.

4. Optional können Sie den Status prüfen oder den Prozess stoppen:

   ```bash
   ./brain_scan.sh status
   ./brain_scan.sh stop
   ```

The scanner starts the worker automatically as soon as jobs appear in the queue.
The worker terminates itself when no documents remain.

## Inspecting the database with sqlite-web

To inspect `DocumentDatabase/state.db` via a browser, install `sqlite-web` on a
Debian/Ubuntu host and expose it on port 8081:

```bash
apt update
apt install -y python3-pip python3-venv
sqlite_web /srv/rag/RAGIngestQdrant/DocumentDatabase/state.db \
  -H 0.0.0.0 -p 8081
```

Afterwards the UI is available at `http://<host>:8081`.

## Working with OpenDocument files

OpenDocument- und klassische Office-Formate werden vor der Textextraktion mit
`soffice` konvertiert (`.ods` → `.xlsx`, andere → PDF) und dann an den
`docling-serve`-Endpoint geschickt. Installiere LibreOffice/OpenOffice, wenn du
die Konvertierung nutzen möchtest; ohne `soffice` schickt der Worker die
Originaldateien direkt an `docling-serve`.

## Optional components

- **Web GUI (`web_gui.py`):** install `flask` alongside `requests` to run the control panel.
- **Demo Brain service (`main_brain_not_part_of_this_folder.py`):** install `fastapi`, `uvicorn`,
  `httpx`, `pydantic`, `pydantic-settings`, and `pypdf` to start the example FastAPI backend.
