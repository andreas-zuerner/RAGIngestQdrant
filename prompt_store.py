import json
import os
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_PROMPTS: Dict[str, str] = {
    "relevance": (
        "You are a strict, privacy-aware classifier for a personal knowledge base. "
        "Return ONLY compact JSON with keys exactly: "
        '{"is_relevant": true|false, "confidence": number, "topics": [string], '""
        '"visibility": "public|private|confidential", "summary": "max 60 words"}.\n'""
        "Relevant = helpful for future Q&A of THIS user (operations/ERP/home-lab/finance), "
        "technical how-to, personal notes, key decisions, configs, logs WITH context.\n"
        "Irrelevant = binaries, trivial short logs, duplicates, cache noise, images without text.\n"
        "If content exposes secrets or identifiers -> visibility='confidential'.\n"
        "Output JSON ONLY. No prose."
    ),
    "chunking": (
        "You are a strict text-segmentation agent.\n"
        "Your ONLY task is to divide the provided text into coherent, semantically intact chunks.\n"
        "Use natural boundaries (paragraphs, headings, section breaks) whenever possible.\n\n"
        "MANDATORY RULES (no exceptions):\n"
        "1. Do NOT summarize any content.\n"
        "2. Do NOT rephrase any part of the text.\n"
        "3. Do NOT translate the text.\n"
        "4. Do NOT split inside a sentence.\n"
        "5. Do NOT add, remove, or alter any words, unless explicitly instructed.\n"
        "6. Produce the output as a verbatim excerpt of the original text, only cut at valid boundaries.\n\n"
        "Output requirement:\n"
        "Return only the first chunk."
    ),
    "context": (
        "You help improve retrieval for text chunks in a vector database.\n\n"
        "Given:\n"
        "- The full source document.\n"
        "- One specific chunk taken from that document.\n\n"
        "Task:\n"
        "- Write a short context paragraph of about 50–500 words.\n"
        "- The context should describe where this chunk belongs in the overall document (topic, section, purpose, important entities, abbreviations, etc.).\n"
        "- Do NOT rewrite or quote the chunk itself.\n"
        "- Do NOT mention \"chunk\" or \"document\" explicitly.\n"
        "- Just output the context paragraph, nothing else."
    ),
}

_ENV_PROMPT_KEYS: Dict[str, str] = {
    "relevance": "PROMPT_RELEVANCE",
    "chunking": "PROMPT_CHUNKING",
    "context": "PROMPT_CONTEXT",
}


def prompt_file_path() -> Path:
    return Path(os.environ.get("PROMPTS_FILE", PROJECT_ROOT / "prompts.json"))


def load_prompts() -> Dict[str, str]:
    prompts = dict(_DEFAULT_PROMPTS)

    path = prompt_file_path()
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key, value in loaded.items():
                    if isinstance(value, str) and key in prompts:
                        prompts[key] = value
        except Exception:
            pass

    for name, env_key in _ENV_PROMPT_KEYS.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            prompts[name] = env_val

    return prompts


def save_prompts(prompts: Dict[str, str]):
    path = prompt_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = {k: v for k, v in prompts.items() if k in _DEFAULT_PROMPTS}
    path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")


def get_prompt(name: str) -> str:
    prompts = load_prompts()
    return prompts.get(name, _DEFAULT_PROMPTS.get(name, ""))

