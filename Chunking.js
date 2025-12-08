// n8n Code-Node (JavaScript)
// Erwartet im Eingang: $json.fullText ODER $json.text

return (async () => {
  // ================== Konfiguration ==================
  const OLLAMA_URL = 'http://192.168.177.130:11434/api/generate'; // ggf. anpassen
  const MODEL = 'llama3.2:latest';                                // z. B. 'llama3.2:3b'
  const WINDOW_SIZE = 5000;                                       // Zeichen, die ans LLM gehen
  const MAX_ITERATIONS = 500;                                     // Sicherheitslimit
  const MIN_CHUNK_LEN = 500;                                      // Fallback-Mindestlänge in Zeichen
  const MATCH_KEY_DEFAULT_LEN = 100;                              // Länge für das Matching-Ende
  const MATCH_KEY_MIN_LEN = 30;                                   // Mindestlänge, bevor wir verlängern
  const LONG_CHUNK_RATIO = 0.9;                                   // 90 % der Window Size
  const OVERLAP_RATIO = 0.2;                                      // 20 % Overlap von WINDOW_SIZE

  // ================== Eingang laden ==================
  const fullText = $json.fullText || $json.text || '';
  const fileName = $json.filename || '';
  const filePath = $json.filepath || '';
  if (!fullText || typeof fullText !== 'string') {
    return [{
      json: {
        error: 'No valid fullText or text on input.',
      },
    }];
  }

  // ================== Hilfsfunktionen ==================

  // LLM aufrufen: liefert den ersten Textabschnitt (Chunk) für ein Fenster
  async function callLLM(windowText) {
    const systemPrompt = `
You are a pure text splitter. Divide the given text into coherent, semantically intact chunks.

STRICT RULES:
- You MUST NOT summarize the text.
- You MUST NOT rephrase the text.
- You MUST NOT translate the text.
- You MUST NOT add or remove any words from the text, unless I explicitly instruct you to do so.

Return only the FIRST chunk.
    `.trim();

    const prompt = `${systemPrompt}

---

TEXT SEGMENT:
${windowText}`;

    const body = {
      model: MODEL,
      prompt,
      stream: false,
      options: {
        temperature: 0.0,
        num_ctx: 40960,
        // Wichtig: num_predict statt max_tokens, sonst wird Option von Ollama ignoriert
        num_predict: 1000,
      },
    };

    const resp = await this.helpers.httpRequest({
      method: 'POST',
      url: OLLAMA_URL,
      body,
      json: true,
    });

    // /api/generate gibt standardmäßig { response: "..." } zurück
    const out = (resp && typeof resp.response === 'string')
      ? resp.response
      : String(resp || '');
    return out.trim();
  }

  // Chunk-Ende finden: nutze die letzten ~100 Zeichen des chunkText als "Match Key"
  function findChunkEnd(fullText, offset, chunkText) {
    if (!chunkText || typeof chunkText !== 'string') {
      return null;
    }

    let cleaned = chunkText.trim();
    if (!cleaned) {
      return null;
    }

    // Standard: letzte MATCH_KEY_DEFAULT_LEN Zeichen
    let matchKey = cleaned.slice(-MATCH_KEY_DEFAULT_LEN).trim();

    // Falls der Key zu kurz ist, versuche etwas mehr Kontext aus dem Ende
    if (matchKey.length < MATCH_KEY_MIN_LEN && cleaned.length > MATCH_KEY_MIN_LEN) {
      matchKey = cleaned.slice(-2 * MATCH_KEY_DEFAULT_LEN).trim();
    }

    // Im Originaltext ab offset suchen
    let idx = -1;
    if (matchKey.length > 0) {
      idx = fullText.indexOf(matchKey, offset);
    }

    let endPos;
    let usedFallback = false;

    if (idx !== -1) {
      // Erfolg: Ende des Chunks = Ende des Match Keys im Originaltext
      endPos = idx + matchKey.length;
    } else {
      // Fallback: schneide nach der Länge der LLM-Antwort,
      // mindestens aber MIN_CHUNK_LEN Zeichen
      usedFallback = true;
      endPos = offset + Math.max(cleaned.length, MIN_CHUNK_LEN);
    }

    return {
      endPos,
      matchKey,
      idx,
      usedFallback,
    };
  }

  // ================== Hauptloop ==================
  let offset = 0;
  let chunkIndex = 1;
  let iterations = 0;
  const chunks = [];

  while (offset < fullText.length && iterations < MAX_ITERATIONS) {
    iterations++;

    const offsetBefore = offset;

    // Basisfenster (volle WINDOW_SIZE)
    const windowStart1 = offsetBefore;
    const windowEnd1 = Math.min(offsetBefore + WINDOW_SIZE, fullText.length);
    const windowText1 = fullText.slice(windowStart1, windowEnd1);
    const windowSize1 = windowEnd1 - windowStart1;

    if (!windowText1.trim()) {
      // nichts Sinnvolles mehr im Fenster → abbrechen
      break;
    }

    // LLM: erster Versuch mit voller Window Size
    let chunkText1 = '';
    let llmError = null;

    try {
      chunkText1 = await callLLM.call(this, windowText1);
    } catch (e) {
      llmError = e;
    }

    if (llmError) {
      // Abbruch bei LLM-Fehler, inkl. Debug
      chunks.push({
        chunk_id: chunkIndex,
        start: offsetBefore,
        end: offsetBefore,
        content: '',
        error: `LLM error: ${llmError.message}`,
        debug: {
          iteration: iterations,
          offset_before: offsetBefore,
          offset_after: offsetBefore,
          windowStart: windowStart1,
          windowEnd: windowEnd1,
          windowText: windowText1,
          chunkText: chunkText1,
          matchKey: null,
          idxInFullText: null,
          usedFallback: null,
          note: 'Error during LLM call (first try)',
        },
      });
      break;
    }

    if (!chunkText1 || !chunkText1.trim()) {
      // Leere LLM-Antwort → abbrechen mit Debug
      chunks.push({
        chunk_id: chunkIndex,
        start: offsetBefore,
        end: offsetBefore,
        content: '',
        error: 'Empty LLM response (first try)',
        debug: {
          iteration: iterations,
          offset_before: offsetBefore,
          offset_after: offsetBefore,
          windowStart: windowStart1,
          windowEnd: windowEnd1,
          windowText: windowText1,
          chunkText: chunkText1,
          matchKey: null,
          idxInFullText: null,
          usedFallback: null,
          note: 'LLM returned empty output (first try)',
        },
      });
      break;
    }

    // Analyse des ersten Versuchs
    const chunkLen1 = chunkText1.length;
    const ratio1 = windowSize1 > 0 ? (chunkLen1 / windowSize1) : 0;

    // Variablen für den "effektiven" Chunk, der weiterverarbeitet wird
    let effectiveWindowStart = windowStart1;
    let effectiveWindowEnd = windowEnd1;
    let effectiveWindowText = windowText1;
    let effectiveChunkText = chunkText1;
    let effectiveWindowSize = windowSize1;
    let halfWindowRetry = false;
    let overlapMode = false;
    let overlapLen = 0;
    let manualEndPos = null;
    let chunkLen2 = null;
    let ratio2 = null;

    // Fallback-Strategie: wenn Chunk > 90 % der Window Size → halbe Window Size probieren
    if (ratio1 > LONG_CHUNK_RATIO && windowSize1 > MIN_CHUNK_LEN * 2) {
      halfWindowRetry = true;

      const halfWindowSize = Math.floor(windowSize1 * 0.5);
      const windowStart2 = offsetBefore;
      const windowEnd2 = Math.min(offsetBefore + halfWindowSize, fullText.length);
      const windowText2 = fullText.slice(windowStart2, windowEnd2);
      const actualHalfSize = windowEnd2 - windowStart2;

      let chunkText2 = '';
      let llmError2 = null;

      try {
        chunkText2 = await callLLM.call(this, windowText2);
      } catch (e2) {
        llmError2 = e2;
      }

      if (!llmError2 && chunkText2 && chunkText2.trim()) {
        chunkLen2 = chunkText2.length;
        ratio2 = actualHalfSize > 0 ? (chunkLen2 / actualHalfSize) : 0;

        if (ratio2 > LONG_CHUNK_RATIO) {
          // Zweiter Versuch auch zu lang → Overlap-Modus
          overlapMode = true;
          overlapLen = Math.floor(WINDOW_SIZE * OVERLAP_RATIO);

          // manueller Chunk-Endpunkt: halfWindow + Overlap
          manualEndPos = offsetBefore + actualHalfSize + overlapLen;
          if (manualEndPos > fullText.length) {
            manualEndPos = fullText.length;
          }

          // Für Debug: effektive Parameter auf den zweiten Versuch setzen
          effectiveWindowStart = windowStart2;
          effectiveWindowEnd = windowEnd2;
          effectiveWindowText = windowText2;
          effectiveChunkText = chunkText2;
          effectiveWindowSize = actualHalfSize;
        } else {
          // Zweiter Versuch ist okay → diesen als effektiven Chunk verwenden
          effectiveWindowStart = windowStart2;
          effectiveWindowEnd = windowEnd2;
          effectiveWindowText = windowText2;
          effectiveChunkText = chunkText2;
          effectiveWindowSize = actualHalfSize;
        }
      } else {
        // Wenn zweiter Versuch fehlschlägt, bleiben wir beim ersten Chunk
        // (kein Overlap-Modus, aber halfWindowRetry = true fürs Debug)
      }
    }

    // Chunk-Ende im Originaltext bestimmen
    let endPos;
    let matchKey = null;
    let idx = -1;
    let usedFallbackMatch = false;

    if (!overlapMode) {
      const match = findChunkEnd(fullText, offsetBefore, effectiveChunkText);

      if (match) {
        endPos = match.endPos;
        matchKey = match.matchKey;
        idx = match.idx;
        usedFallbackMatch = match.usedFallback;
      } else {
        // Sicherheits-Fallback, falls findChunkEnd nichts liefert
        usedFallbackMatch = true;
        endPos = offsetBefore + Math.min(
          effectiveWindowSize,
          Math.max(effectiveChunkText.length, MIN_CHUNK_LEN),
        );
      }
    } else {
      // Overlap-Modus: manueller Endpunkt
      endPos = manualEndPos;
      matchKey = null;
      idx = -1;
      usedFallbackMatch = true;
    }

    // Grenzen absichern
    if (endPos <= offsetBefore) {
      usedFallbackMatch = true;
      endPos = Math.min(offsetBefore + MIN_CHUNK_LEN, fullText.length);
    }
    if (endPos > fullText.length) {
      usedFallbackMatch = true;
      endPos = fullText.length;
    }

    const offsetAfter = endPos;
    const content = fullText.slice(offsetBefore, endPos);

    // Chunk inkl. Debug-Infos speichern
    chunks.push({
      chunk_id: chunkIndex,
      start: offsetBefore,
      end: endPos,
      content,
      fullText,
      fileName,
      filePath, 
      debug: {
        iteration: iterations,
        offset_before: offsetBefore,
        offset_after: offsetAfter,
        // Effektives Fenster, das für den finalen Chunk relevant war
        windowStart: effectiveWindowStart,
        windowEnd: effectiveWindowEnd,
        windowText: effectiveWindowText,
        chunkText: effectiveChunkText,
        matchKey,
        idxInFullText: idx,
        usedFallbackMatch,
        // Zusatzinfos zur Fallback-Strategie
        originalWindowSize: windowSize1,
        originalChunkLen: chunkLen1,
        originalRatio: ratio1,
        halfWindowRetry,
        halfWindowSize: effectiveWindowSize,
        secondChunkLen: chunkLen2,
        secondRatio: ratio2,
        overlapMode,
        overlapLen,
        longChunkRatio: LONG_CHUNK_RATIO,
      },
    });

    chunkIndex++;

    // Nächste offset-Position
    if (overlapMode && overlapLen > 0) {
      // Overlap: wir springen nicht ganz bis endPos, sondern 20 % WINDOW_SIZE zurück
      const nextOffset = endPos - overlapLen;
      // Sicherheitscheck: wir müssen Fortschritt machen
      if (nextOffset <= offsetBefore) {
        offset = endPos; // notfalls ohne Overlap weitermachen
      } else {
        offset = nextOffset;
      }
    } else {
      // normaler Modus: direkt ans Ende des aktuellen Chunks
      offset = endPos;
    }
  }

  // Falls der Loop nicht bis zum Textende kam, letzten Rest anhängen
  if (offset < fullText.length) {
    const tailStart = offset;
    const tailEnd = fullText.length;
    chunks.push({
      chunk_id: chunkIndex,
      start: tailStart,
      end: tailEnd,
      content: fullText.slice(tailStart, tailEnd),
      note: 'tail_chunk_added_without_llm',
      debug: {
        tail: true,
        tailStart,
        tailEnd,
      },
    });
  }

  // n8n erwartet eine Item-Liste
  return chunks.map(c => ({ json: c }));
})();

