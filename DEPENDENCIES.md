# Dependencies & Usage Guide

This document summarizes the runtime dependencies of the BrainScanDocs project
and demonstrates how to prepare a local environment for running the scanner and
worker components.

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

Test fixtures live in `ct109-data/`, while the SQLite database is stored in a
separate, relative directory:

- `DocumentDatabase/` &rarr; location where the SQLite database (`state.db`)
  will be created at runtime.
Test fixtures live in `ct109-data/`:

- `ct109-data/scan_root/` &rarr; contains sample `.txt` files that help verify
  the scanner/worker integration during development.

## Configuration

Environment variables werden in `.env.local` abgelegt. Eine vollständige
Beispieldatei liegt unter `.env.local.example` bei und enthält sämtliche
Konfigurationsschlüssel (u. a. Datenbankpfad, Scan-Wurzelverzeichnisse, Brain- und
Ollama-Endpunkte). Kopieren Sie die Datei und passen Sie die Werte an Ihre
Umgebung an:

```bash
cp .env.local.example .env.local
# anschließend DB_PATH, ROOT_DIRS, BRAIN_API_KEY usw. konfigurieren
```

## Running the services

1. Activate the virtual environment and ensure dependencies are installed.
2. Kopieren und bearbeiten Sie `.env.local` wie oben beschrieben.
3. Starten Sie Scanner und Worker gemeinsam über das Steuerskript:

   ```bash
   ./brain_scan.sh start
   ```

   Der Scheduler legt Logs unter `logs/scan_scheduler.log` ab.

4. Optional können Sie den Status prüfen oder den Prozess stoppen:

   ```bash
   ./brain_scan.sh status
   ./brain_scan.sh stop
   ```

Der Scanner sorgt dafür, dass der Worker automatisch gestartet wird, sobald
Jobs in der Datenbank-Queue liegen. Der Worker beendet sich selbst, wenn keine
Dokumente mehr anstehen.

## Working with OpenDocument files

OpenDocument Text (`.odt`) and related formats can be placed in
`ct109-data/scan_root/`. The ingest worker now recognises these archives, pulls
text content from the `content.xml` payload and keeps the LLM/Brain interfaces
unchanged.
