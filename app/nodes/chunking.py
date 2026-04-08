"""
chunker.py
----------
Splits transcripts into semantically meaningful chunks for sentiment analysis.

Key improvements over the original:
- Respects sentence boundaries (no mid-sentence cuts)
- Tags each chunk with its source quality tier
- Skips 'metadata_only' and 'failed' transcripts for chunking
  (they are analysed whole, not in pieces)
- Adds chunk index and position fraction for time-based sentiment mapping
"""

import re
from typing import List


CHUNK_WORD_TARGET = 150   # aim for ~150 words per chunk
CHUNK_WORD_MIN    = 40    # discard chunks shorter than this


def _split_into_sentences(text: str) -> List[str]:
    """Simple sentence splitter that handles common medical abbreviations."""
    # Protect common abbreviations so they don't get split
    abbrevs = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "vs.", "etc.", "e.g.", "i.e.", "approx."]
    for abbr in abbrevs:
        text = text.replace(abbr, abbr.replace(".", "<!DOT!>"))
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.replace("<!DOT!>", ".").strip() for s in sentences if s.strip()]


def _build_chunks(sentences: List[str], target_words: int = CHUNK_WORD_TARGET) -> List[str]:
    """Greedily group sentences into chunks near the target word count."""
    chunks, current, current_words = [], [], 0
    for sent in sentences:
        word_count = len(sent.split())
        if current_words + word_count > target_words and current:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sent)
        current_words += word_count
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_transcripts(state: dict) -> dict:
    """
    LangGraph node: converts transcripts → chunks.

    Output chunk shape:
      {
        "video_id": str,
        "chunk_index": int,
        "total_chunks": int,
        "position": float,   # 0.0 (start) → 1.0 (end)
        "text": str,
        "word_count": int,
        "source": str,       # inherited from transcript
        "usable": bool,
      }
    """
    transcripts = state.get("transcripts", [])
    all_chunks = []

    for transcript in transcripts:
        vid = transcript["video_id"]
        source = transcript.get("source", "failed")
        text = transcript.get("text", "").strip()
        usable = transcript.get("usable", False)

        if not text:
            continue

        # Metadata-only and very short transcripts: treat as a single chunk
        if source in ("metadata_only", "failed") or not usable:
            all_chunks.append({
                "video_id": vid,
                "chunk_index": 0,
                "total_chunks": 1,
                "position": 0.0,
                "text": text,
                "word_count": len(text.split()),
                "source": source,
                "usable": usable,
            })
            continue

        sentences = _split_into_sentences(text)
        raw_chunks = _build_chunks(sentences)

        # Filter out very short chunks
        raw_chunks = [c for c in raw_chunks if len(c.split()) >= CHUNK_WORD_MIN]

        total = max(len(raw_chunks), 1)
        for i, chunk_text in enumerate(raw_chunks):
            all_chunks.append({
                "video_id": vid,
                "chunk_index": i,
                "total_chunks": total,
                "position": round(i / total, 3),
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "source": source,
                "usable": True,
            })

    return {**state, "chunks": all_chunks}
