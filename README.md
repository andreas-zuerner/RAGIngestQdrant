# RAGIngestQdrant

RAGIngestQdrant durchsucht konfigurierbare Verzeichnisse, legt gefundene Dokumente in
einer SQLite-Datenbank ab und lässt relevante Inhalte über einen externen
"Brain"-Dienst auswerten. Ein Ollama-Endpoint bewertet die Relevanz einzelner
Texte, bevor sie zur Persistierung weitergereicht werden.

## Quickstart

1. Abhängigkeiten installieren (siehe `DEPENDENCIES.md`) und ein virtuelles
   Environment vorbereiten:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r <(printf "requests\nchardet\nbeautifulsoup4\nlxml\npython-docx\nPyPDF2\npdf2image\nPillow\ndocling\n")
   ```

2. Beispiel-Konfiguration übernehmen und anpassen:

   ```bash
   cp .env.local.example .env.local
   # anschließend die Schlüssel wie BRAIN_API_KEY, OLLAMA_HOST usw. anpassen
   ```

   Relevante Parameter für den Brain-Dienst:

   * `BRAIN_CHUNK_TOKENS` – maximale Token-Anzahl pro Chunk (kleiner = kürzere Embedding-Laufzeiten).
   * `BRAIN_OVERLAP_TOKENS` – Überlappung zwischen zwei Chunks.
   * `BRAIN_REQUEST_TIMEOUT` – Timeout in Sekunden für `POST /ingest/text`.

3. Scanner + Worker mit dem neuen Steuerskript starten:

   ```bash
   ./brain_scan.sh start
   ```

   Der Scanner prüft periodisch auf neue Dateien und startet den Worker
   automatisch, sobald Jobs in der Queue liegen. Der Worker beendet sich selbst,
   wenn es keine offenen Dokumente mehr gibt. Bereits indizierte Dateien werden
   bei Neustarts nicht erneut eingeplant, solange ihr `next_review_at`-Termin noch
   in der Zukunft liegt. Erkennt der Scheduler, dass eine Datei wirklich
   verschwunden ist, markiert er sie als gelöscht; reine Verschiebungen werden
   über Datei-Metadaten erkannt und behalten ihren bisherigen Status. Der Worker
   verarbeitet sämtliche Dokumente inzwischen über `docling` (siehe
   https://docling-project.github.io/docling/) und profitiert dadurch von den
   eingebauten Extraktions-Pipelines inklusive Hybrid-Chunking sowie optionaler
   OCR-Unterstützung.

4. Status prüfen oder stoppen:

   ```bash
   ./brain_scan.sh status
   ./brain_scan.sh stop
   ```

   Die Konsolen-Ausgabe befindet sich in `logs/scan_scheduler.log`.

## Nextcloud-Einbindung und docling-serve

* Der Scheduler durchsucht standardmäßig zwei Nextcloud-Ordner per WebDAV: einen
  Dokumente-Ordner (`NEXTCLOUD_DOC_DIR`, Default `/RAGdocuments`) und
  einen Bilder-Ordner (`NEXTCLOUD_IMAGE_DIR`, Default
  `/Ragimages`). Beide Pfade lassen sich alternativ über
  `ROOT_DIRS` (kommagetrennt) überschreiben. Der Bilder-Ordner wird von der
  SQLite-Datenbank genauso überwacht wie der Dokument-Ordner und kann dadurch
  bequem mit einer Admin-Oberfläche wie phpLiteAdmin gepflegt werden.
* `scan_scheduler.py` initialisiert den WebDAV-Client über `env_client()` und
  läuft mit `client.walk(...)` rekursiv über die oben genannten Nextcloud-
  Ordner. So landen neu hochgeladene Dateien unmittelbar als Jobs in der
  lokalen Queue.
* `ingest_worker.py` nutzt ebenfalls denselben Nextcloud-Client: Fehlt eine
  Datei lokal, wird sie per `download_to_temp(...)` aus dem Dokumente-Ordner
  nachgeladen. Zusätzlich speichert der Docling-Ingestor erkannte Bilder via
  `upload_bytes(...)` in den dedizierten Nextcloud-Bilder-Ordner und fügt die
  Referenzen in den extrahierten Text ein.
* Die Nextcloud-Instanz ist über `http://192.168.177.133:8080` erreichbar. Der
  Standardbenutzer lautet `andreas`; der dazugehörige API-Token wird in
  `.env.local` unter `TOKEN` (optional auch `NEXTCLOUD_TOKEN`) hinterlegt und
  beim Starten des Scans automatisch als Umgebungsvariable geladen.
* Für die Textextraktion wird kein lokales `docling` mehr benötigt. Stattdessen
  spricht der Worker den bereitgestellten Webservice
  `docling-serve` unter `DOCLING_SERVE_URL` (Default
  `http://192.168.177.130:5001/v1/convert/file`). Der Dienst liefert Text und
  erkannte Bilder; letztere werden im Bilder-Ordner abgelegt und im Text mit
  eindeutigen Referenzen (`![…](<pfad>)`) vermerkt.

## Git-Befehle ausführen

* Alle Git-Kommandos (z. B. `git checkout -b <zweig>`, `git switch main`,
  `git pull`) führst du im Projektstamm aus: `/workspace/BrainScanDocs`.
* Von dort aus kannst du auch alternative Stände nutzen, ohne `main` zu ändern,
  z. B.:

  ```bash
  git checkout <commit-oder-branch>
  pip install .
  ```

  oder direkt aus einem Branch installieren:

  ```bash
  pip install "git+https://<repo-url>.git@<branch>"
  ```
