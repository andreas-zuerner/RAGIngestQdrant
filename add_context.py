import json
import textwrap
from typing import List

import requests


def _context_prompt(document: str, chunks: List[str]) -> str:
    joined = "\n---\n".join(chunks)
    return textwrap.dedent(
        """
        You enrich pre-defined document chunks with minimal context so each chunk is self contained.
        For every chunk, add 1-2 concise sentences that provide missing context (speaker, location, section purpose).
        Keep the original chunk text intact and append the context inline. Do NOT summarize or remove details.

        Return JSON array of strings in the same order, one enriched chunk per input chunk.

        Full document for reference is enclosed below. Use it only for context completion.
        """
    ).strip() + "\n<document>\n" + document + "\n</document>\n<chunks>\n" + joined + "\n</chunks>"


def enrich_chunks_with_context(
    *,
    document: str,
    chunks: List[str],
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
    debug: bool = False,
) -> List[str]:
    if not chunks:
        return []

    prompt = _context_prompt(document, chunks)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192, "top_p": 1.0},
    }
    url = f"{ollama_host.rstrip('/')}/api/generate"
    response = requests.post(url, json=payload, timeout=timeout)
    if debug:
        try:
            from pathlib import Path
            dbg_dir = Path("./debug/add_context")
            dbg_dir.mkdir(parents=True, exist_ok=True)
            (dbg_dir / "prompt.txt").write_text(prompt[:5000], encoding="utf-8")
            (dbg_dir / "response.txt").write_text(response.text[:5000], encoding="utf-8")
        except Exception:
            pass

    response.raise_for_status()
    raw = (response.json().get("response") or "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return chunks
    if isinstance(parsed, list) and len(parsed) == len(chunks):
        enriched: List[str] = []
        for original, candidate in zip(chunks, parsed):
            if isinstance(candidate, str) and candidate.strip():
                enriched.append(candidate.strip())
            else:
                enriched.append(original)
        return enriched
    return chunks
