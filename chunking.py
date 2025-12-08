import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

SYSTEM_PROMPT = """
You are a strict text-segmentation agent.
Your ONLY task is to divide the provided text into coherent, semantically intact chunks.
Use natural boundaries (paragraphs, headings, section breaks) whenever possible.

MANDATORY RULES (no exceptions):
1. Do NOT summarize any content.
2. Do NOT rephrase any part of the text.
3. Do NOT translate the text.
4. Do NOT split inside a sentence.
5. Do NOT add, remove, or alter any words, unless explicitly instructed.
6. Produce the output as a verbatim excerpt of the original text, only cut at valid boundaries.

Output requirement:
Return only the first chunk.
""".strip()

PROMPT_TEMPLATE = """{system}\n\n---\n\nTEXT SEGMENT:\n{segment}"""

WINDOW_SIZE_CHARS = 5000
MAX_ITERATIONS = 500
MIN_CHUNK_LEN = 500
MATCH_KEY_DEFAULT_LEN = 100
MATCH_KEY_MIN_LEN = 30
LONG_CHUNK_RATIO = 0.9
OVERLAP_RATIO = 0.2
LONG_CHUNK_NEXT_OVERLAP_RATIO = 0.05


@dataclass
class ChunkDebug:
    iteration: int
    offset_before: int
    offset_after: int
    windowStart: int
    windowEnd: int
    windowText: str
    chunkText: str
    matchKey: Optional[str]
    idxInFullText: Optional[int]
    usedFallback: Optional[bool]
    originalWindowSize: int
    originalChunkLen: int
    originalRatio: float
    halfWindowRetry: bool
    halfWindowSize: int
    secondChunkLen: Optional[int]
    secondRatio: Optional[float]
    overlapMode: bool
    overlapLen: int
    longChunkRatio: float


@dataclass
class Chunk:
    chunk_id: int
    start: int
    end: int
    content: str
    fullText: str
    fileName: Optional[str] = None
    filePath: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)


def _clean_text_for_match(text: str) -> str:
    """Normalize text for deterministic matching while keeping length stable."""
    return text.replace("\r", " ").replace("\n", " ")


def _build_prompt(segment: str) -> str:
    return PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, segment=segment)


def _call_llm(window_text: str, *, ollama_host: str, model: str, timeout: float) -> str:
    body = {
        "model": model,
        "prompt": _build_prompt(window_text),
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 40960,
            "num_predict": 1000,
        },
    }
    url = f"{ollama_host.rstrip('/')}/api/generate"
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response") if isinstance(data, dict) else ""
    return str(raw or "").strip()


def _find_chunk_end(
    prepared_full_text: str, offset: int, chunk_text: str
) -> Optional[Tuple[int, str, int, bool]]:
    if not chunk_text or not chunk_text.strip():
        return None

    cleaned = _clean_text_for_match(chunk_text).strip()
    if not cleaned:
        return None

    match_key = cleaned[-MATCH_KEY_DEFAULT_LEN:].strip()
    if len(match_key) < MATCH_KEY_MIN_LEN and len(cleaned) > MATCH_KEY_MIN_LEN:
        match_key = cleaned[-2 * MATCH_KEY_DEFAULT_LEN :].strip()

    idx = -1
    if match_key:
        idx = prepared_full_text.find(match_key, offset)

    used_fallback = False
    if idx != -1:
        end_pos = idx + len(match_key)
    else:
        used_fallback = True
        end_pos = offset + max(len(cleaned), MIN_CHUNK_LEN)

    return end_pos, match_key, idx, used_fallback


def _chunk_full_text(
    full_text: str,
    *,
    ollama_host: str,
    model: str,
    window_size: int = WINDOW_SIZE_CHARS,
    timeout: float = 120.0,
) -> List[Chunk]:
    if not full_text or not full_text.strip():
        return []

    prepared_full_text = _clean_text_for_match(full_text)

    offset = 0
    chunk_index = 1
    iterations = 0
    chunks: List[Chunk] = []

    while offset < len(full_text) and iterations < MAX_ITERATIONS:
        iterations += 1
        offset_before = offset

        window_start1 = offset_before
        window_end1 = min(offset_before + window_size, len(full_text))
        window_text1 = full_text[window_start1:window_end1]
        window_size1 = window_end1 - window_start1

        if not window_text1.strip():
            break

        try:
            chunk_text1 = _call_llm(window_text1, ollama_host=ollama_host, model=model, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            debug = ChunkDebug(
                iteration=iterations,
                offset_before=offset_before,
                offset_after=offset_before,
                windowStart=window_start1,
                windowEnd=window_end1,
                windowText=window_text1,
                chunkText="",
                matchKey=None,
                idxInFullText=None,
                usedFallback=None,
                originalWindowSize=window_size1,
                originalChunkLen=0,
                originalRatio=0.0,
                halfWindowRetry=False,
                halfWindowSize=0,
                secondChunkLen=None,
                secondRatio=None,
                overlapMode=False,
                overlapLen=0,
                longChunkRatio=LONG_CHUNK_RATIO,
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_index,
                    start=offset_before,
                    end=offset_before,
                    content="",
                    fullText=full_text,
                    debug={"error": str(exc), "detail": dataclasses.asdict(debug)},
                )
            )
            break

        if not chunk_text1.strip():
            debug = ChunkDebug(
                iteration=iterations,
                offset_before=offset_before,
                offset_after=offset_before,
                windowStart=window_start1,
                windowEnd=window_end1,
                windowText=window_text1,
                chunkText=chunk_text1,
                matchKey=None,
                idxInFullText=None,
                usedFallback=None,
                originalWindowSize=window_size1,
                originalChunkLen=0,
                originalRatio=0.0,
                halfWindowRetry=False,
                halfWindowSize=0,
                secondChunkLen=None,
                secondRatio=None,
                overlapMode=False,
                overlapLen=0,
                longChunkRatio=LONG_CHUNK_RATIO,
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_index,
                    start=offset_before,
                    end=offset_before,
                    content="",
                    fullText=full_text,
                    debug={"error": "Empty LLM response", "detail": dataclasses.asdict(debug)},
                )
            )
            break

        chunk_len1 = len(chunk_text1)
        ratio1 = (chunk_len1 / window_size1) if window_size1 else 0.0

        effective_window_start = window_start1
        effective_window_end = window_end1
        effective_window_text = window_text1
        effective_chunk_text = chunk_text1
        effective_window_size = window_size1
        half_window_retry = False
        overlap_mode = False
        overlap_len = 0
        applied_overlap_mode = False
        applied_overlap_len = 0
        manual_end_pos: Optional[int] = None
        chunk_len2: Optional[int] = None
        ratio2: Optional[float] = None

        if ratio1 > LONG_CHUNK_RATIO and window_size1 > MIN_CHUNK_LEN * 2:
            half_window_retry = True
            half_window_size = int(window_size1 * 0.5)
            window_start2 = offset_before
            window_end2 = min(offset_before + half_window_size, len(full_text))
            window_text2 = full_text[window_start2:window_end2]
            actual_half_size = window_end2 - window_start2

            try:
                chunk_text2 = _call_llm(
                    window_text2, ollama_host=ollama_host, model=model, timeout=timeout
                )
            except Exception:
                chunk_text2 = ""

            if chunk_text2 and chunk_text2.strip():
                chunk_len2 = len(chunk_text2)
                ratio2 = (chunk_len2 / actual_half_size) if actual_half_size else 0.0

                if ratio2 > LONG_CHUNK_RATIO:
                    overlap_mode = True
                    overlap_len = int(window_size * OVERLAP_RATIO)
                    manual_end_pos = offset_before + actual_half_size + overlap_len
                    manual_end_pos = min(manual_end_pos, len(full_text))
                    effective_window_start = window_start2
                    effective_window_end = window_end2
                    effective_window_text = window_text2
                    effective_chunk_text = chunk_text2
                    effective_window_size = actual_half_size
                else:
                    effective_window_start = window_start2
                    effective_window_end = window_end2
                    effective_window_text = window_text2
                    effective_chunk_text = chunk_text2
                    effective_window_size = actual_half_size
            # if second try fails, fall back to first attempt

        end_pos: int
        match_key: Optional[str]
        idx: int
        used_fallback_match: bool

        if not overlap_mode:
            match = _find_chunk_end(prepared_full_text, offset_before, effective_chunk_text)
            if match:
                end_pos, match_key, idx, used_fallback_match = match
            else:
                used_fallback_match = True
                match_key = None
                idx = -1
                end_pos = offset_before + min(
                    effective_window_size, max(len(effective_chunk_text), MIN_CHUNK_LEN)
                )
        else:
            end_pos = manual_end_pos if manual_end_pos is not None else offset_before + effective_window_size
            match_key = None
            idx = -1
            used_fallback_match = True

        if end_pos <= offset_before:
            used_fallback_match = True
            end_pos = min(offset_before + MIN_CHUNK_LEN, len(full_text))
        if end_pos > len(full_text):
            used_fallback_match = True
            end_pos = len(full_text)

        offset_after = end_pos
        content = full_text[offset_before:end_pos]

        if not overlap_mode:
            effective_ratio = (
                (len(content) / effective_window_size) if effective_window_size else 0.0
            )
            if effective_ratio >= LONG_CHUNK_RATIO:
                proposed_overlap = int(effective_window_size * LONG_CHUNK_NEXT_OVERLAP_RATIO)
                if proposed_overlap > 0:
                    applied_overlap_mode = True
                    applied_overlap_len = proposed_overlap
        else:
            applied_overlap_mode = overlap_mode and overlap_len > 0
            applied_overlap_len = overlap_len if overlap_len > 0 else 0

        debug = ChunkDebug(
            iteration=iterations,
            offset_before=offset_before,
            offset_after=offset_after,
            windowStart=effective_window_start,
            windowEnd=effective_window_end,
            windowText=effective_window_text,
            chunkText=effective_chunk_text,
            matchKey=match_key,
            idxInFullText=idx,
            usedFallback=used_fallback_match,
            originalWindowSize=window_size1,
            originalChunkLen=chunk_len1,
            originalRatio=ratio1,
            halfWindowRetry=half_window_retry,
            halfWindowSize=effective_window_size,
            secondChunkLen=chunk_len2,
            secondRatio=ratio2,
            overlapMode=applied_overlap_mode,
            overlapLen=applied_overlap_len,
            longChunkRatio=LONG_CHUNK_RATIO,
        )

        chunks.append(
            Chunk(
                chunk_id=chunk_index,
                start=offset_before,
                end=end_pos,
                content=content,
                fullText=full_text,
                debug=dataclasses.asdict(debug),
            )
        )

        chunk_index += 1
        if applied_overlap_mode and applied_overlap_len > 0:
            next_offset = end_pos - applied_overlap_len
            offset = end_pos if next_offset <= offset_before else next_offset
        else:
            offset = end_pos

    if offset < len(full_text):
        tail_start = offset
        tail_end = len(full_text)
        chunks.append(
            Chunk(
                chunk_id=chunk_index,
                start=tail_start,
                end=tail_end,
                content=full_text[tail_start:tail_end],
                fullText=full_text,
                debug={"tail": True, "tailStart": tail_start, "tailEnd": tail_end},
            )
        )

    return chunks


def chunk_document_with_llm_fallback(
    text: str,
    *,
    ollama_host: str,
    model: str,
    target_tokens: int = 1000,
    overlap_chars: int = 0,
    max_chunks: Optional[int] = None,
    timeout: float = 120.0,
    debug: bool = False,
    return_debug: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    window_size = max(int(target_tokens * 4), WINDOW_SIZE_CHARS)
    chunks = _chunk_full_text(text, ollama_host=ollama_host, model=model, window_size=window_size, timeout=timeout)
    if max_chunks:
        chunks = chunks[:max_chunks]
    chunk_texts = [c.content for c in chunks]
    metadata: Dict[str, Any] = {
        "chunk_source": "llm",
        "llm_window_size": window_size,
        "chunks": len(chunk_texts),
    }
    if debug:
        metadata["raw_chunks"] = [dataclasses.asdict(c) for c in chunks]
    if return_debug:
        return chunk_texts, metadata
    return chunk_texts, metadata
