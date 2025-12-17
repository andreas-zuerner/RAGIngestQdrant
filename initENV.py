import os
from pathlib import Path
from typing import Dict, Iterable, Set

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = Path(os.environ.get("ENV_FILE", PROJECT_ROOT / ".env.local"))
ENV_EXAMPLE = Path(os.environ.get("ENV_EXAMPLE", PROJECT_ROOT / ".env.local.example"))

_loaded_env: Dict[str, str] | None = None

def _load_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env

def load_env() -> Dict[str, str]:
    global _loaded_env
    if _loaded_env is not None:
        return _loaded_env
    _loaded_env = _load_env_file(ENV_FILE)
    for key, value in _loaded_env.items():
        os.environ.setdefault(key, value)
    return _loaded_env

def env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return str(val).lower() in {"1", "true", "yes"}

def env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except Exception:
        return default

def env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except Exception:
        return default

def env_list(key: str, default: Iterable[str] | None = None, *, sep: str = ",") -> list[str]:
    value = os.environ.get(key)
    if value is None:
        return list(default or [])
    return [item for item in value.split(sep) if item]


def _normalize_extensions(values: Iterable[str]) -> Set[str]:
    normalized: Set[str] = set()
    for raw in values:
        ext = (raw or "").strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


# Initialize environment on import
load_env()

DEBUG = env_bool("DEBUG", False)

DB_PATH = os.environ.get("DB_PATH", "DocumentDatabase/state.db")
BRAIN_URL = os.environ.get("BRAIN_URL", "http://192.168.177.151:8080").rstrip("/")
BRAIN_API_KEY = os.environ.get("BRAIN_API_KEY", "change-me")
BRAIN_COLLECTION = os.environ.get("BRAIN_COLLECTION", "documents")
BRAIN_CHUNK_TOKENS = env_int("BRAIN_CHUNK_TOKENS", 400)
BRAIN_REQUEST_TIMEOUT = env_float("BRAIN_REQUEST_TIMEOUT", 120)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.177.130:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral-small3.2:latest")
OLLAMA_MODEL_RELEVANCE = os.environ.get("OLLAMA_MODEL_RELEVANCE", OLLAMA_MODEL)
OLLAMA_MODEL_CHUNKING = os.environ.get("OLLAMA_MODEL_CHUNKING", OLLAMA_MODEL)
OLLAMA_MODEL_CONTEXT = os.environ.get("OLLAMA_MODEL_CONTEXT", OLLAMA_MODEL)

RELEVANCE_THRESHOLD = env_float("RELEVANCE_THRESHOLD", 0.55)
MIN_CHARS = env_int("MIN_CHARS", 200)
MAX_TEXT_CHARS = env_int("MAX_TEXT_CHARS", 100000)
MAX_CHARS = env_int("MAX_CHARS", 4000)
OVERLAP = env_int("OVERLAP", 400)
MAX_CHUNKS = env_int("MAX_CHUNKS", 200)
ENABLE_OCR = env_bool("ENABLE_OCR", False)

LOCK_TIMEOUT_S = env_int("LOCK_TIMEOUT_S", 600)
IDLE_SLEEP_S = env_float("IDLE_SLEEP_S", 1.0)
PDFTOTEXT_TIMEOUT_S = env_int("PDFTOTEXT_TIMEOUT_S", 60)
PDF_OCR_MAX_PAGES = env_int("PDF_OCR_MAX_PAGES", 20)
PDF_OCR_DPI = env_int("PDF_OCR_DPI", 300)

# File type configuration
FILE_TYPES_STANDARD = _normalize_extensions(
    env_list(
        "FILE_TYPES_STANDARD",
        default=[
            "docx",
            "pptx",
            "html",
            "htm",
            "xhtml",
            "pdf",
            "md",
            "csv",
            "xlsx",
            "xml",
            "json",
            "vtt",
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff",
            "bmp",
            "webp",
        ],
    )
)

FILE_TYPES_SOFFICE = _normalize_extensions(
    env_list(
        "FILE_TYPES_SOFFICE",
        default=[".xls", ".doc", ".odf", ".odp", ".odt", ".odg", ".ods", ".odm"],
    )
)

FILE_TYPES_MS_EXTENDED = _normalize_extensions(
    env_list(
        "FILE_TYPES_MS_EXTENDED",
        default=[
            ".dotx",
            ".docm",
            ".dotm",
            ".potx",
            ".ppsx",
            ".pptm",
            ".potm",
            ".ppsm",
            ".xlsm",
        ],
    )
)

FILE_TYPES_AUDIO = _normalize_extensions(
    env_list(
        "FILE_TYPES_AUDIO",
        default=[
            ".wav",
            ".mp3",
            ".m4a",
            ".aac",
            ".ogg",
            ".flac",
            ".mp4",
            ".avi",
            ".mov",
        ],
    )
)

FILE_TYPES_TABLE = _normalize_extensions(
    env_list(
        "FILE_TYPES_TABLE",
        default=[".csv", ".ods", ".xls", ".xlsx"],
    )
)

ENABLE_SOFFICE_TYPES = env_bool("ENABLE_SOFFICE_TYPES", True)
ENABLE_MS_EXTENDED_TYPES = env_bool("ENABLE_MS_EXTENDED_TYPES", False)
ENABLE_AUDIO_TYPES = env_bool("ENABLE_AUDIO_TYPES", False)

DOCLING_SERVE_URL = os.environ.get("DOCLING_SERVE_URL", "http://192.168.177.130:5001/v1/convert/file")
DOCLING_SERVE_TIMEOUT = env_float("DOCLING_SERVE_TIMEOUT", 300)
DOCLING_SERVE_USE_ASYNC = env_bool("DOCLING_SERVE_USE_ASYNC", True)
DOCLING_SERVE_ASYNC_URL = os.environ.get("DOCLING_SERVE_ASYNC_URL")
DOCLING_SERVE_ASYNC_TIMEOUT = env_float("DOCLING_SERVE_ASYNC_TIMEOUT", 900)
DOCLING_SERVE_ASYNC_POLL_INTERVAL = env_float("DOCLING_SERVE_ASYNC_POLL_INTERVAL", 5)

NEXTCLOUD_DOC_DIR = os.environ.get("NEXTCLOUD_DOC_DIR", "/RAGdocuments")
NEXTCLOUD_IMAGE_DIR = os.environ.get("NEXTCLOUD_IMAGE_DIR", "/RAG-images")
NEXTCLOUD_BASE_URL = os.environ.get("NEXTCLOUD_BASE_URL", "http://192.168.177.133:8080").rstrip("/")
NEXTCLOUD_USER = os.environ.get("NEXTCLOUD_USER", "andreas")
NEXTCLOUD_TOKEN = os.environ.get("NEXTCLOUD_TOKEN", "")

DECISION_LOG_ENABLED = env_bool("DECISION_LOG_ENABLED", True)
DECISION_LOG_MAX_PER_JOB = env_int("DECISION_LOG_MAX_PER_JOB", 50)

WEB_GUI_SECRET = os.environ.get("WEB_GUI_SECRET", "dev-secret")
WEB_GUI_PORT = env_int("WEB_GUI_PORT", 8088)
WEB_GUI_HOST = os.environ.get("WEB_GUI_HOST", "0.0.0.0")
SCHEDULER_LOG = os.environ.get("SCHEDULER_LOG", str(PROJECT_ROOT / "logs/scan_scheduler.log"))

SCAN_SCHEDULER_PID_FILE = os.environ.get("SCAN_SCHEDULER_PID_FILE")
EXCLUDE_GLOBS = env_list("EXCLUDE_GLOBS", default=[])
FOLLOW_SYMLINKS = env_bool("FOLLOW_SYMLINKS", False)
MAX_JOBS_PER_PASS = env_int("MAX_JOBS_PER_PASS", 5)
SLEEP_SECS = env_int("SLEEP_SECS", 10)
WORKER_STOP_TIMEOUT = env_int("WORKER_STOP_TIMEOUT", 20)
# Maximum number of concurrent docling-serve conversions
DOCLING_MAX_WORKERS = env_int("MAX_WORKERS", 1)
# Number of ingest_worker processes handling the post-extraction pipeline
PIPELINE_WORKERS = env_int("PIPELINE_WORKERS", 1)
