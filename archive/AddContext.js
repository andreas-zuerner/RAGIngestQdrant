// n8n Code-Node (JavaScript)
// Mode: "Run Once for All Items"
// Eingang: Chunk-Items vom Chunking-Node
// Erwartetes Schema pro Item:
// {
//   chunk_id: number,
//   start: number,
//   end: number,
//   content: string,
//   debug: { ... },
//   // optional: fullText / text / documentName / documentPath
// }

return (async () => {
  // ================== Konfiguration ==================
  const OLLAMA_URL = 'http://192.168.177.130:11434/api/generate'; // anpassen falls nötig
  const MODEL = 'llama3.2:latest';                                // Kontext-Modell
  const CONTEXT_MIN_WORDS = 50;
  const CONTEXT_MAX_WORDS = 100;

  const itemsIn = items;
  if (!itemsIn || itemsIn.length === 0) {
    return [{ json: { error: 'No chunk items on input.' } }];
  }

  // ================== Hilfsfunktionen ==================

  // Volltext rekonstruieren, falls er nicht als fullText/text mitkommt
  function stripExtractedImagesSection(text) {
    if (!text || typeof text !== 'string') return '';
    const pattern = /\n## Extracted images\s*\n(?:!\[[^\]]*\]\([^\)]+\)\s*\n?)+\s*$/i;
    return text.replace(pattern, '').trimEnd();
  }

  function reconstructFullText(chunks) {
    const sorted = [...chunks].sort((a, b) => {
      const sa = (typeof a.start === 'number') ? a.start : Number.MAX_SAFE_INTEGER;
      const sb = (typeof b.start === 'number') ? b.start : Number.MAX_SAFE_INTEGER;
      if (sa !== sb) return sa - sb;
      const ca = (typeof a.chunk_id === 'number') ? a.chunk_id : 0;
      const cb = (typeof b.chunk_id === 'number') ? b.chunk_id : 0;
      return ca - cb;
    });
    return sorted.map(c => c.content || '').join('');
  }

  // Bildreferenzen aus dem Chunktext extrahieren: [IMAGE:foo]
  function extractImageRefsFromChunk(text) {
    if (!text || typeof text !== 'string') return [];
    const regex = /\[IMAGE:([^\]]+)\]/g;
    const images = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
      const name = match[1].trim();
      if (name && !images.includes(name)) {
        images.push(name);
      }
    }
    return images;
  }

  // LLM-Aufruf, um Kontext für einen Chunk zu erzeugen
  async function generateContext(fullDoc, chunkText) {
    const systemPrompt = `
You help improve retrieval for text chunks in a vector database.

Given:
- The full source document.
- One specific chunk taken from that document.

Task:
- Write a short context paragraph of about ${CONTEXT_MIN_WORDS}–${CONTEXT_MAX_WORDS} words.
- The context should describe where this chunk belongs in the overall document (topic, section, purpose, important entities, abbreviations, etc.).
- Do NOT rewrite or quote the chunk itself.
- Do NOT mention "chunk" or "document" explicitly.
- Just output the context paragraph, nothing else.
    `.trim();

    const prompt = `${systemPrompt}

---

FULL DOCUMENT:
${fullDoc}

---

FOCUS CHUNK:
${chunkText}

---

CONTEXT (${CONTEXT_MIN_WORDS}-${CONTEXT_MAX_WORDS} words):`;

    const body = {
      model: MODEL,
      prompt,
      stream: false,
      options: {
        temperature: 0.2,
        num_ctx: 32768,
        num_predict: 200, // reicht für ~50–100 Wörter
      },
    };

    const resp = await this.helpers.httpRequest({
      method: 'POST',
      url: OLLAMA_URL,
      body,
      json: true,
    });

    const raw = (resp && typeof resp.response === 'string')
      ? resp.response
      : String(resp || '');

    let context = raw.trim();

    // Sicherheitsnetz: zu lange Antworten grob kappen
    const words = context.split(/\s+/);
    if (words.length > CONTEXT_MAX_WORDS * 2) {
      context = words.slice(0, CONTEXT_MAX_WORDS * 2).join(' ');
    }

    return context;
  }

  // ================== Volltext & Metadaten bestimmen ==================

  const chunks = itemsIn.map(i => i.json);

  const first = itemsIn[0].json;
  const documentName =
    first.documentName ||
    first.docName ||
    first.fileName ||
    first.filename ||
    null;

  const documentPath =
    first.documentPath ||
    first.docPath ||
    first.filePath ||
    null;

  let fullDoc =
    first.fullText ||
    first.text ||
    reconstructFullText(chunks);

  if (typeof fullDoc !== 'string') {
    fullDoc = String(fullDoc || '');
  }

  fullDoc = stripExtractedImagesSection(fullDoc);

  const totalChunks = chunks.length;

  // ================== Hauptlogik: pro Chunk Kontext erzeugen ==================

  const resultItems = [];

  const sortedChunks = [...chunks].sort((a, b) => {
    const sa = (typeof a.start === 'number') ? a.start : Number.MAX_SAFE_INTEGER;
    const sb = (typeof b.start === 'number') ? b.start : Number.MAX_SAFE_INTEGER;
    if (sa !== sb) return sa - sb;
    const ca = (typeof a.chunk_id === 'number') ? a.chunk_id : 0;
    const cb = (typeof b.chunk_id === 'number') ? b.chunk_id : 0;
    return ca - cb;
  });

  for (let i = 0; i < sortedChunks.length; i++) {
    const c = sortedChunks[i];
    const chunkText = stripExtractedImagesSection(c.content || '');

    // Bildreferenzen aus dem Chunktext extrahieren
    const imageRefs = extractImageRefsFromChunk(chunkText);

    if (!chunkText.trim()) {
      resultItems.push({
        json: {
          text: chunkText,
          meta: {
            source: 'n8n',
            document_name: documentName,
            document_path: documentPath,
            chunk_id: c.chunk_id ?? (i + 1),
            //chunk_index: i,
            chunk_count: totalChunks,
            start: c.start,
            end: c.end,
            images: imageRefs,
            note: 'empty_chunk_no_context_generated',
          },
        },
      });
      continue;
    }

    let context = '';
    try {
      context = await generateContext(fullDoc, chunkText);
    } catch (e) {
      resultItems.push({
        json: {
          text: chunkText,
          meta: {
            source: 'n8n',
            document_name: documentName,
            document_path: documentPath,
            chunk_id: c.chunk_id ?? (i + 1),
            //chunk_index: i,
            chunk_count: totalChunks,
            start: c.start,
            end: c.end,
            images: imageRefs,
            context_error: e.message,
          },
        },
      });
      continue;
    }

    const combinedText = context
      ? `${context}\n\n${chunkText}`
      : chunkText;

    const meta = {
      source: 'n8n',
      document_name: documentName,
      document_path: documentPath,
      chunk_id: c.chunk_id ?? (i + 1),
      //chunk_index: i,
      chunk_count: totalChunks,
      start: c.start,
      end: c.end,
      images: imageRefs,
      original_debug: c.debug || undefined,
    };

    resultItems.push({
      json: {
        text: combinedText,
        meta,
      },
    });
  }

  return resultItems;
})();

