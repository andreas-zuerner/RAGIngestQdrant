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

## Repository overview

| File | Role | Required | Used by |
| brain_scan.sh | Start/stop/status wrapper for the ingestion pipeline (sets ENV, starts scheduler, logging) | Yes | Operator
| scan_scheduler.py | Scans Nextcloud, enqueues jobs, controls parallel workers | Yes | brain_scan.sh
| ingest_worker.py | Core ingestion worker: download, Docling extraction, relevance, chunking, context, image upload, SQL persistence | Yes | Scheduler
| text_extraction.py | Docling-Serve client (sync + async polling), document conversion helpers | Yes | Worker
| nextcloud_client.py | Nextcloud WebDAV abstraction (walk, download, upload, delete) | Yes | Scheduler, Worker, GUI
| chunking.py | LLM-based semantic chunking via Ollama | Yes | Worker
| add_context.py | Context enrichment for chunks before vector ingestion | Yes | Worker
| helpers.py | SQLite schema creation and shared DB helpers | Yes | Scheduler, Worker
| initENV.py | Central environment defaults and configuration | Yes | All runtime components
| prompt_store.py | Prompt loading and access abstraction | Yes | Worker, GUI
| prompts.json | Prompt definitions (relevance, chunking, context, etc.) | Yes | Prompt store
| web_gui.py | Optional web UI for controlling pipeline, prompts and reset operations | Optional | Operator
| gui.html | Frontend for web_gui.py | Optional | GUI
| test_nextcloud_client.py | Developer test for Nextcloud client | No (dev only) | Developers
| main_brain_not_part_of_this_folder.py | Example FastAPI “Brain” service (not part of ingestion pipeline) | No (example) | Demo
| schema.sql | Legacy SQL schema (superseded by helpers.ensure_db) | No (legacy) | —
| normalize_timesteps.sql | Legacy maintenance SQL for timestamp normalization | No (legacy) | —
Notes on database schema

The SQLite database schema is created programmatically via helpers.ensure_db() on startup.
The files schema.sql and normalize_timesteps.sql are legacy artifacts and are not required for normal operation.

## LLM-Modellwahl pro Verarbeitungsschritt

Standardmäßig wird das Modell aus `OLLAMA_MODEL` für alle Schritte verwendet. Bei Bedarf
kannst du für einzelne Aufgaben spezifische Modelle hinterlegen (anderen Host musst du
dafür nicht setzen):

* `OLLAMA_MODEL_RELEVANCE` – Relevanzbewertung des extrahierten Volltexts.
* `OLLAMA_MODEL_CHUNKING` – LLM-gestütztes Chunking vor dem Ingest in den Brain-Dienst.
* `OLLAMA_MODEL_CONTEXT` – Kontextanreicherung der erzeugten Chunks.

Bleiben diese Variablen leer, greift automatisch das allgemeine Modell aus `OLLAMA_MODEL`.

## Nextcloud-Einbindung und docling-serve

* Der Scheduler durchsucht per WebDAV den Dokumente-Ordner
  (`NEXTCLOUD_DOC_DIR`, Default `/RAGdocuments`; die Beispiel-`.env.local` setzt
  `/nextcloud/documents`). Der Bilder-Ordner (`NEXTCLOUD_IMAGE_DIR`, Default
  `/RAG-images`) wird nicht gescannt, sondern nur für Uploads der extrahierten
  Bilder genutzt.
* `scan_scheduler.py` initialisiert den WebDAV-Client über `env_client()` und
  läuft mit `client.walk(...)` rekursiv über die oben genannten Nextcloud-
  Ordner. So landen neu hochgeladene Dateien unmittelbar als Jobs in der
  lokalen Queue.
* `ingest_worker.py` nutzt ebenfalls denselben Nextcloud-Client: Fehlt eine
  Datei lokal, wird sie per `download_to_temp(...)` aus dem Dokumente-Ordner
  nachgeladen. Erkannte docling-Bilder werden erst nach der LLM-Relevanzprüfung
  (Resultat `TRUE`) in den dedizierten Nextcloud-Bilder-Ordner hochgeladen; sie
  landen dort in einem eigenen Unterordner pro Dokument (z. B.
  `/RAG-images/<slug>/...`) und die Referenzen werden dann im extrahierten Text
  aktualisiert. Bei aktivem `DEBUG=1` legt der Worker die von docling gelieferten
  Rohdaten zusätzlich in `logs/docling/<slug>/` ab (Text + PNG-Dateien),
  getrennt nach Dokumenten.
* Die Nextcloud-Instanz ist über `http://192.168.177.133:8080` erreichbar. Der
  Standardbenutzer lautet `andreas`; der dazugehörige API-Token wird in
  `.env.local` unter `NEXTCLOUD_TOKEN` hinterlegt und beim Starten des Scans
  automatisch als Umgebungsvariable geladen.
* Für die Textextraktion wird kein lokales `docling` mehr benötigt. Stattdessen
  spricht der Worker den bereitgestellten Webservice
  `docling-serve` unter `DOCLING_SERVE_URL` (Default
  `http://192.168.177.130:5001/v1/convert/file`). Der Dienst liefert Text und
  erkannte Bilder; letztere werden im Bilder-Ordner abgelegt und im Text mit
  eindeutigen Referenzen (`![…](<pfad>)`) vermerkt.

## Ingestion-Pipeline und Dateirouting

1. **Scan Scheduler (`scan_scheduler.py`)**
   * Läuft periodisch über `NEXTCLOUD_DOC_DIR` (Default `/RAGdocuments`).
   * Blendet Dateien aus, die auf `EXCLUDE_GLOBS` passen.
   * Enqueued pro Durchlauf höchstens `MAX_JOBS_PER_PASS` Einträge und schläft
     dazwischen `SLEEP_SECS` Sekunden. Sowohl die Docling-Extraktion (intern
     `DOCLING_MAX_WORKERS`, gesteuert per `MAX_WORKERS`) als auch die
     nachgelagerte Pipeline (`PIPELINE_WORKERS`) sind über Umgebungsvariablen
     begrenzt, sodass parallel nur die gewünschte Anzahl an Dokumenten
     verarbeitet wird.

2. **Download & Vorbereitung** im Worker (`ingest_worker.py`)
   * Falls die Datei lokal fehlt, lädt `NextcloudClient.download_to_temp(...)`
     sie aus Nextcloud nach (`NEXTCLOUD_BASE_URL`, `NEXTCLOUD_USER`,
     `NEXTCLOUD_TOKEN`). Erkannte Bilder landen im Upload-Ordner
     `NEXTCLOUD_IMAGE_DIR`.
   * Je nach Dateiendung wird entschieden, welcher Pfad genutzt wird:

     | Kategorie | Gesteuert über | Route |
     | --- | --- | --- |
     | Standardformate (`FILE_TYPES_STANDARD`) | aktiv | Direkt an `docling-serve`.
     | Office/ODF (`FILE_TYPES_SOFFICE`, nur wenn `ENABLE_SOFFICE_TYPES=1`) | aktiv | Vorab-Konvertierung mit `soffice` (`.ods` → `.xlsx`, Rest → PDF) und dann `docling-serve`.
     | Tabellen (`FILE_TYPES_TABLE`) | aktiv | Wie oben extrahiert, zusätzlich werden die Tabellen-Zeilen gelesen und in SQLite abgelegt. Relevanzprüfung wird übersprungen; es entsteht genau ein (ggf. gekürzter) Zusammenfassungs-Chunk.
     | Erweiterte MS-Formate (`FILE_TYPES_MS_EXTENDED`, `ENABLE_MS_EXTENDED_TYPES`) | optional | `docling-serve` wird übersprungen; Fallback liest die Binärdatei best-effort als Text.
     | Audio/Video (`FILE_TYPES_AUDIO`, `ENABLE_AUDIO_TYPES`) | optional | Nicht für die Docling-Pipeline vorgesehen; fällt auf best-effort-Text-Decode zurück.
     | Sonstige | – | werden zwar akzeptiert, landen aber direkt im generischen Text-Fallback.

   * `DOCLING_SERVE_USE_ASYNC=1` nutzt den asynchronen Endpoint (falls
     konfiguriert über `DOCLING_SERVE_ASYNC_URL`). Timeouts und Polling werden
     über `DOCLING_SERVE_TIMEOUT`, `DOCLING_SERVE_ASYNC_TIMEOUT` und
     `DOCLING_SERVE_ASYNC_POLL_INTERVAL` gesteuert.

3. **Relevanz & Chunking**
   * Tabellen-Dateien gelten automatisch als relevant.
   * Für alle anderen Formate bewertet Ollama (`OLLAMA_HOST`,
     `OLLAMA_MODEL_RELEVANCE`) die Relevanz. Die Verarbeitung endet, wenn die
     Konfidenz unter `RELEVANCE_THRESHOLD` liegt.
   * Chunking erfolgt per LLM (`OLLAMA_MODEL_CHUNKING`, Tokenlimit
     `BRAIN_CHUNK_TOKENS`, Overlap `OVERLAP`, Maximalanzahl `MAX_CHUNKS`).
     Fällt das LLM aus, werden Docling-Chunks als Fallback genutzt.

4. **Kontext & Embedding**
   * `add_context.py` reichert die Chunks mit Kontext an (`OLLAMA_MODEL_CONTEXT`).
   * Anschließend sendet der Worker die finalen Chunks an den Brain-Dienst
     (`BRAIN_URL`, `BRAIN_API_KEY`, `BRAIN_COLLECTION`) mit Timeout
     `BRAIN_REQUEST_TIMEOUT`. Bilder werden vorher in `NEXTCLOUD_IMAGE_DIR`
     hochgeladen und in den Text eingebettet.

5. **Persistenz**
   * Metadaten und Verarbeitungsstatus liegen in der SQLite-Datenbank unter
     `DB_PATH` (Default `DocumentDatabase/state.db`). Dort werden u. a. die
     gefundenen Dateien (`files`), die verplanten Jobs (`jobs`), Entscheidungen
     (`decision_log`) sowie extrahierte Tabellen (`table_registry` /
     `table_data`) gespeichert.
   * Vektorisierte Texte gehen per Brain-Endpoint in die konfigurierte
     Qdrant-Collection (`BRAIN_COLLECTION`) des Brain-Dienstes; die lokalen
     Tabellen-Referenzen bleiben parallel in SQLite erhalten.
   * Extrahierte Bilder werden in `NEXTCLOUD_IMAGE_DIR` abgelegt, referenziert
     im gespeicherten Text und bleiben damit unabhängig vom lokalen Dateisystem.

### Office-Dokumente (ODF/Office → PDF)

`text_extraction.py` wandelt klassische Office- und OpenDocument-Formate
(`.odf`, `.odp`, `.odt`, `.odg`, `.ods`, `.odm` sowie ältere `.doc`/`.xls`)
vor dem Senden an `docling-serve` in PDFs um. Dafür wird das CLI `soffice`
aus LibreOffice oder OpenOffice benötigt. Installationsbeispiele:

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y libreoffice libreoffice-writer libreoffice-calc

# macOS (Homebrew)
brew install --cask libreoffice
```

Das Binary `soffice` muss im `PATH` liegen; andernfalls überspringt der
Worker die Konvertierung und schickt die Originaldatei direkt an
`docling-serve`.

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

## Web GUI control

A lightweight Flask-based GUI is available to start/stop the scheduler, inspect logs, search Qdrant entries, manage environment variables (including prompt texts), delete individual file IDs, and perform full resets. Launch it with:

```bash
WEB_GUI_SECRET=change-me WEB_GUI_PORT=8088 WEB_GUI_HOST=0.0.0.0 \
python web_gui.py
```

The GUI reads and writes `.env.local` by default (controlled via `ENV_FILE`) and requires access to Qdrant, Nextcloud, and the local `DocumentDatabase/state.db`. The reset action drops the configured Qdrant collection, removes the state database, clears the `RAG-images` folder in Nextcloud, and can optionally restore `.env.local` from `.env.local.example`.

### LAN / reverse-proxy deployment (Apache or lighttpd)

The GUI must listen on the LAN for future Dockerization. Bind to all interfaces via `WEB_GUI_HOST=0.0.0.0` (default) and use a separate secret in production.

1. **Install prerequisites** on Debian/Ubuntu:

   ```bash
   apt update
   apt install -y python3-venv python3-pip git apache2 apache2-utils \
     libapache2-mod-proxy-html libxml2-dev
   # or, if you prefer lighttpd instead of Apache:
   # apt install -y lighttpd lighttpd-mod-proxy
   ```

2. **Place the application** under a dedicated path (example `/srv/rag/RAGIngestQdrant`):

   ```bash
   mkdir -p /srv/rag
   git clone <this-repo-url> /srv/rag/RAGIngestQdrant
   cd /srv/rag/RAGIngestQdrant
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r <(printf "flask\nrequests\n")
   chown -R www-data:www-data /srv/rag/RAGIngestQdrant
   chmod -R 750 /srv/rag/RAGIngestQdrant
   ```

   `www-data` (or your service user) needs read/write access to `.env.local`, `DocumentDatabase/state.db`, and the `logs/` directory to reflect scheduler state and log output.

3. **Run the app with a WSGI/ASGI server** (example: Gunicorn) so Apache/lighttpd can proxy it:

   ```bash
   source /srv/rag/RAGIngestQdrant/.venv/bin/activate
   WEB_GUI_SECRET=$(openssl rand -hex 32) \
   WEB_GUI_PORT=8088 WEB_GUI_HOST=0.0.0.0 \
   gunicorn --chdir /srv/rag/RAGIngestQdrant web_gui:app --bind 0.0.0.0:8088
   ```

   Create a systemd unit `/etc/systemd/system/rag-gui.service` for auto-start:

   ```ini
   [Unit]
   Description=RAG Control Panel
   After=network.target

   [Service]
   WorkingDirectory=/srv/rag/RAGIngestQdrant
   Environment="WEB_GUI_SECRET=/etc/rag_gui_secret"
   Environment="WEB_GUI_PORT=8088" "WEB_GUI_HOST=0.0.0.0"
   ExecStart=/srv/rag/RAGIngestQdrant/.venv/bin/gunicorn web_gui:app --bind 0.0.0.0:8088
   User=www-data
   Group=www-data
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

   Store the secret in `/etc/rag_gui_secret` with `chmod 640 /etc/rag_gui_secret` and `chown www-data:root /etc/rag_gui_secret`, then `systemctl daemon-reload && systemctl enable --now rag-gui`.

4. **Expose via Apache** (reverse proxy):

   ```bash
   a2enmod proxy proxy_http headers
   cat >/etc/apache2/sites-available/rag-gui.conf <<'EOF'
   <VirtualHost *:80>
     ServerName rag-gui.local
     ProxyPreserveHost On
     ProxyPass / http://127.0.0.1:8088/
     ProxyPassReverse / http://127.0.0.1:8088/
     RequestHeader set X-Forwarded-Proto http
     <Location />
       Require all granted
     </Location>
   </VirtualHost>
   EOF
   a2ensite rag-gui.conf
   systemctl reload apache2
   ```

5. **Expose via lighttpd** (reverse proxy):

   ```bash
   lighttpd-enable-mod proxy proxy_http
   cat >/etc/lighttpd/conf-available/30-rag-gui.conf <<'EOF'
   $HTTP["host"] == "rag-gui.local" {
     proxy.server = (
       "/" => ( ( "host" => "127.0.0.1", "port" => 8088 ) )
     )
   }
   EOF
   lighttpd-enable-mod rag-gui
   systemctl reload lighttpd
   ```

6. **File permissions for persistent state**: ensure the service user owns the runtime artifacts:

   ```bash
   chown www-data:www-data /srv/rag/RAGIngestQdrant/.env.local \
     /srv/rag/RAGIngestQdrant/DocumentDatabase /srv/rag/RAGIngestQdrant/logs
   chmod 640 /srv/rag/RAGIngestQdrant/.env.local
   chmod 750 /srv/rag/RAGIngestQdrant/DocumentDatabase /srv/rag/RAGIngestQdrant/logs
   ```

With `WEB_GUI_HOST=0.0.0.0`, the service is reachable from the local LAN (or Docker bridge when containerized); forward the port (e.g., `-p 8088:8088` with Docker) to reach the GUI externally.

### Optional dependencies for auxiliary components

- **Web GUI:** install `flask` (plus `requests`, already included above) to run `web_gui.py`.
- **Demo Brain service (`main_brain_not_part_of_this_folder.py`):** requires `fastapi`, `httpx`, `pydantic`, `pydantic-settings`, `pypdf`, and a server such as `uvicorn` to launch the app.
