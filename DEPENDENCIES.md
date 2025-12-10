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
# anschließend DB_PATH, ROOT_DIRS, BRAIN_API_KEY usw. konfigurieren
```

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

OpenDocument Text (`.odt`) and related formats can be placed in
`ct109-data/scan_root/`. The ingest worker now recognises these archives, pulls
text content from the `content.xml` payload and keeps the LLM/Brain interfaces
unchanged.
