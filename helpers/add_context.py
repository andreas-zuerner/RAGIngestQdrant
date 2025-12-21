import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .prompt_store import get_prompt

CONTEXT_MIN_WORDS = 50
CONTEXT_MAX_WORDS = 500

PROMPT_TEMPLATE = """{system}\n\n---\n\nFULL DOCUMENT:\n{document}\n\n---\n\nFOCUS CHUNK:\n{chunk}\n\n---\n\nCONTEXT ({min_words}-{max_words} words):"""


@dataclass
class ChunkItem:
    chunk_id: int
    start: Optional[int]
    end: Optional[int]
    content: str
    fullText: Optional[str] = None
    fileName: Optional[str] = None
    filePath: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


@dataclass
class ContextChunk:
    text: str
    meta: Dict[str, Any]
    error: Optional[str] = None


def _extract_image_refs(text: str) -> List[str]:
    """Parse the chunk text and return unique inline image markers like `[IMAGE:name]`."""
    if not text or not isinstance(text, str):
        return []
    regex = re.compile(r"\[IMAGE:([^\]]+)\]")
    images: List[str] = []
    for match in regex.finditer(text):
        name = match.group(1).strip()
        if name and name not in images:
            images.append(name)
    return images


def _extract_table_refs(text: str) -> List[str]:
    """Parse chunk text for table references like `[TABLE:uuid]`."""
    if not text or not isinstance(text, str):
        return []
    regex = re.compile(r"\[TABLE:([^\]]+)\]")
    refs: List[str] = []
    for match in regex.finditer(text):
        tid = match.group(1).strip()
        if tid and tid not in refs:
            refs.append(tid)
    return refs


EXTRACTED_IMAGES_SECTION = re.compile(
    r"\n## Extracted images\s*\n(?:!\[[^\]]*\]\([^\)]+\)\s*\n?)+\s*$",
    re.IGNORECASE,
)


def _strip_extracted_images_section(text: str) -> str:
    """Remove trailing image listing sections from docling outputs."""
    if not isinstance(text, str):
        return "" if text is None else str(text)
    cleaned = EXTRACTED_IMAGES_SECTION.sub("", text)
    return cleaned.rstrip()


def _reconstruct_full_text(chunks: List[ChunkItem]) -> str:
    """Rebuild the best-effort full document text from individual chunks."""
    ordered = sorted(
        chunks,
        key=lambda c: (
            c.start if isinstance(c.start, int) else 10**12,
            c.chunk_id if isinstance(c.chunk_id, int) else 0,
        ),
    )
    # Join with newlines to avoid accidentally gluing words together when chunk
    # boundaries are based on offsets without explicit separators.
    return "\n\n".join(c.content or "" for c in ordered)


def _call_llm(full_doc: str, chunk_text: str, *, ollama_host: str, model: str, timeout: float) -> str:
    """Send the document + chunk to the LLM and return the generated context."""
    prompt = PROMPT_TEMPLATE.format(
        system=get_prompt("context"),
        document=full_doc,
        chunk=chunk_text,
        min_words=CONTEXT_MIN_WORDS,
        max_words=CONTEXT_MAX_WORDS,
    )
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 32768,
            "num_predict": 200,
        },
    }
    url = f"{ollama_host.rstrip('/')}/api/generate"
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response") if isinstance(data, dict) else ""
    context = str(raw or "").strip()

    words = context.split()
    if len(words) > CONTEXT_MAX_WORDS * 2:
        context = " ".join(words[: CONTEXT_MAX_WORDS * 2])
    return context


def _sort_chunks(items: List[ChunkItem]) -> List[ChunkItem]:
    """Sort chunks by their start offset and chunk id for deterministic processing."""
    return sorted(
        items,
        key=lambda c: (
            c.start if isinstance(c.start, int) else 10**12,
            c.chunk_id if isinstance(c.chunk_id, int) else 0,
        ),
    )


def add_context_to_chunks(
    items: List[Dict[str, Any]],
    *,
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
    debug: Optional[Any] = None,
    table_lookup: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Annotate chunks with LLM-generated context paragraphs and rich metadata."""
    chunks = [ChunkItem(chunk_id=i.get("chunk_id", idx + 1), start=i.get("start"), end=i.get("end"), content=i.get("content") or i.get("text") or "", fullText=i.get("fullText"), fileName=i.get("fileName") or i.get("filename"), filePath=i.get("filePath") or i.get("filepath"), debug=i.get("debug", debug)) for idx, i in enumerate(items)]

    if not chunks:
        return []

    first = chunks[0]
    document_name = first.fileName
    document_path = first.filePath

    full_doc_candidates = [c.fullText for c in chunks if c.fullText]
    full_doc = full_doc_candidates[0] if full_doc_candidates else _reconstruct_full_text(chunks)
    if not isinstance(full_doc, str):
        full_doc = str(full_doc or "")
    full_doc = _strip_extracted_images_section(full_doc)

    sorted_chunks = _sort_chunks(chunks)
    total_chunks = len(sorted_chunks)

    results: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(sorted_chunks):
        chunk_text = _strip_extracted_images_section(chunk.content or "")
        image_refs = _extract_image_refs(chunk_text)
        table_refs = _extract_table_refs(chunk_text)

        # IMPORTANT: Do not inline raw table rows into the chunk that will later be sent to the
        # vector store (brain). This can easily explode the payload size. Instead, we only
        # enrich the *FULL DOCUMENT* with a (small) sampled table excerpt, so the LLM can
        # generate a compact context paragraph.
        chunk_tables = {}
        chunk_text_with_tables = chunk_text
        if table_refs and table_lookup:
            table_summaries = []
            for table_id in table_refs:
                rows = table_lookup.get(table_id)
                if rows:
                    table_summaries.append(f"[TABLE_DATA {table_id}]\n{json.dumps(rows, ensure_ascii=False)}")
            full_doc_with_tables = f"{full_doc}\n\n" + "\n\n".join(table_summaries) if table_summaries else full_doc
        else:
            full_doc_with_tables = full_doc

        if not chunk_text.strip():
            results.append(
                {
                    "text": chunk_text,
                    "meta": {
                        "source": "RAGIngestQdrant",
                        "document_name": document_name,
                        "document_path": document_path,
                        "chunk_id": chunk.chunk_id or (idx + 1),
                        "chunk_count": total_chunks,
                        "start": chunk.start,
                        "end": chunk.end,
                        "images": image_refs,
                        "note": "empty_chunk_no_context_generated",
                    },
                }
            )
            continue

        if table_refs and table_lookup:
            table_summaries = []
            for table_id in table_refs:
                rows = table_lookup.get(table_id)
                if rows:
                    table_summaries.append(f"[TABLE_DATA {table_id}]\n{json.dumps(rows, ensure_ascii=False)}")
            if table_summaries:
                full_doc_with_tables = f"{full_doc}\n\n" + "\n\n".join(table_summaries)
            else:
                full_doc_with_tables = full_doc
        else:
            full_doc_with_tables = full_doc

        try:
            context = _call_llm(full_doc_with_tables, chunk_text_with_tables, ollama_host=ollama_host, model=model, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "text": chunk_text,
                    "meta": {
                        "source": "RAGIngestQdrant",
                        "document_name": document_name,
                        "document_path": document_path,
                        "chunk_id": chunk.chunk_id or (idx + 1),
                        "chunk_count": total_chunks,
                        "start": chunk.start,
                        "end": chunk.end,
                        "images": image_refs,
                        "tables": table_refs,
                        "context_error": str(exc),
                    },
                }
            )
            continue

        combined_text = f"{context}\n\n{chunk_text_with_tables}" if context else chunk_text_with_tables
        results.append(
            {
                "text": combined_text,
                "meta": {
                    "source": "RAGIngestQdrant",
                    "document_name": document_name,
                    "document_path": document_path,
                    "chunk_id": chunk.chunk_id or (idx + 1),
                    "chunk_count": total_chunks,
                    "start": chunk.start,
                    "end": chunk.end,
                    "images": image_refs,
                    "tables": table_refs,
                    "original_debug": chunk.debug,
                },
            }
        )

    return results


def enrich_chunks_with_context(
    *,
    document: str,
    chunks: List[str],
    ollama_host: str,
    model: str,
    timeout: float = 120.0,
    debug: Optional[Any] = None,
    table_lookup: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[ContextChunk]:
    """Convenience wrapper to enrich plain chunk strings with generated context."""
    items = [
        {
            "chunk_id": idx + 1,
            "start": None,
            "end": None,
            "content": chunk,
            "fullText": document,
            "debug": debug,
        }
        for idx, chunk in enumerate(chunks)
    ]

    enriched_items = add_context_to_chunks(
        items,
        ollama_host=ollama_host,
        model=model,
        timeout=timeout,
        debug=debug,
        table_lookup=table_lookup,
    )
    context_chunks: List[ContextChunk] = []
    for item in enriched_items:
        meta = item.get("meta") or {}
        context_chunks.append(
            ContextChunk(
                text=item.get("text", ""),
                meta=meta,
                error=meta.get("context_error"),
            )
        )

    return context_chunks
