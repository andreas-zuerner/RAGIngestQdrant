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

3. Scanner + Worker mit dem Steuerskript starten:

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
   verarbeitet sämtliche Dokumente über `docling` (siehe
   https://docling-project.github.io/docling/) und profitiert dadurch von den
   eingebauten Extraktions-Pipelines inklusive Hybrid-Chunking sowie optionaler
   OCR-Unterstützung.

4. Status prüfen oder stoppen:

   ```bash
   ./brain_scan.sh status
   ./brain_scan.sh stop
   ```

   Die Konsolen-Ausgabe befindet sich in `logs/scan_scheduler.log`.

## LLM-Modellwahl pro Verarbeitungsschritt

Standardmäßig wird das Modell aus `OLLAMA_MODEL` für alle Schritte verwendet. Bei Bedarf
kannst du für einzelne Aufgaben spezifische Modelle hinterlegen (anderen Host musst du
dafür nicht setzen):

* `OLLAMA_MODEL_RELEVANCE` – Relevanzbewertung des extrahierten Volltexts.
* `OLLAMA_MODEL_CHUNKING` – LLM-gestütztes Chunking vor dem Ingest in den Brain-Dienst.
* `OLLAMA_MODEL_CONTEXT` – Kontextanreicherung der erzeugten Chunks.

Bleiben diese Variablen leer, greift automatisch das allgemeine Modell aus `OLLAMA_MODEL`.

## Nextcloud-Einbindung und docling-serve

* Der Scheduler durchsucht standardmäßig zwei Nextcloud-Ordner per WebDAV: einen
  Dokumente-Ordner (`NEXTCLOUD_DOC_DIR`, Default `/RAGdocuments`) und
  einen Bilder-Ordner (`NEXTCLOUD_IMAGE_DIR`, Default
  `/RAGimages`). Beide Pfade lassen sich alternativ über
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

## SQLite-Web: Datenbank prüfen

Um den aktuellen Zustand der SQLite-Datenbank (`DocumentDatabase/state.db`) zu
prüfen, kann `sqlite-web` eingesetzt werden. Installation und Start auf einer
Debian/Ubuntu-Maschine:

```bash
apt update
apt install -y python3-pip python3-venv
sqlite_web /srv/rag/RAGIngestQdrant/DocumentDatabase/state.db \
  -H 0.0.0.0 -p 8081
```

Der Dienst ist anschließend unter `http://<host>:8081` erreichbar und zeigt die
Tabellen, Einträge und Index-Informationen der Datenbank an.
