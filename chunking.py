import json
import textwrap
from typing import List

import requests


class ChunkingError(RuntimeError):
    """Raised when LLM-based chunking fails."""


def _approx_chars(token_count: int) -> int:
    return max(token_count * 4, 1)


def _default_options():
    return {
        "temperature": 0,
        "top_p": 1.0,
        "num_ctx": 8192,
    }


def _extract_response(json_body: dict) -> str:
    raw = (json_body or {}).get("response") or ""
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _parse_chunk_payload(raw: str) -> List[str]:
    try:
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise ChunkingError(f"could not parse chunk JSON: {exc}") from exc

    if isinstance(parsed, list):
        texts: List[str] = []
        for item in parsed:
            if isinstance(item, str):
                candidate = item.strip()
            elif isinstance(item, dict):
                candidate = str(item.get("text", "")).strip()
            else:
                continue
            if candidate:
                texts.append(candidate)
        if texts:
            return texts
    raise ChunkingError("chunk response did not contain any text chunks")


def _chunk_prompt(text: str, *, target_tokens: int, max_chunks: int) -> str:
    target_chars = _approx_chars(target_tokens)
    return textwrap.dedent(
        f"""
        You are a meticulous document segmenter.
        Split the provided document into coherent, non-overlapping chunks that preserve context.
        Aim for chunks close to {target_tokens} tokens (~{target_chars} characters) without exceeding natural boundaries.
        Never create more than {max_chunks} chunks. Prefer paragraph or section borders instead of arbitrary cuts.

        Return ONLY compact JSON: [{{"text": "chunk text", "reason": "why you cut here"}}, ...].
        Keep the original wording. Do not summarize or paraphrase. Maintain the original order.

        Document:
        """
    ).strip() + "\n<document>\n" + text + "\n</document>"


def llm_chunk_document(
    text: str,
    *,
    ollama_host: str,
    model: str,
    target_tokens: int = 800,
    max_chunks: int = 20,
    timeout: float = 120.0,
    debug: bool = False,
) -> List[str]:
    if not text or not text.strip():
        raise ChunkingError("empty text supplied")

    prompt = _chunk_prompt(text, target_tokens=target_tokens, max_chunks=max_chunks)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": _default_options(),
    }
    url = f"{ollama_host.rstrip('/')}/api/generate"
    response = requests.post(url, json=payload, timeout=timeout)
    if debug:
        try:
            from pathlib import Path

            dbg_dir = Path("./debug/chunking")
            dbg_dir.mkdir(parents=True, exist_ok=True)
            (dbg_dir / "prompt.txt").write_text(prompt[:5000], encoding="utf-8")
            (dbg_dir / "response.txt").write_text(response.text[:5000], encoding="utf-8")
        except Exception:
            pass

    response.raise_for_status()
    raw = _extract_response(response.json())
    return _parse_chunk_payload(raw)


def fallback_chunk_document(
    text: str,
    *,
    chunk_size_chars: int,
    overlap_chars: int,
    max_chunks: int,
) -> List[str]:
    if not text or not text.strip():
        return []
    step = max(chunk_size_chars - overlap_chars, 1)
    chunks: List[str] = []
    idx = 0
    while idx < len(text):
        if max_chunks and len(chunks) >= max_chunks:
            break
        end = min(idx + chunk_size_chars, len(text))
        chunk_text = text[idx:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        idx += step
    if not chunks:
        chunks = [text.strip()]
    return chunks


def chunk_document_with_llm_fallback(
    text: str,
    *,
    ollama_host: str,
    model: str,
    target_tokens: int,
    overlap_chars: int,
    max_chunks: int,
    timeout: float,
    debug: bool = False,
) -> List[str]:
    try:
        return llm_chunk_document(
            text,
            ollama_host=ollama_host,
            model=model,
            target_tokens=target_tokens,
            max_chunks=max_chunks,
            timeout=timeout,
            debug=debug,
        )
    except Exception:
        chunk_chars = _approx_chars(target_tokens)
        return fallback_chunk_document(
            text,
            chunk_size_chars=chunk_chars,
            overlap_chars=overlap_chars,
            max_chunks=max_chunks,
        )
