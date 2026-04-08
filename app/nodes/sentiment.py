"""
sentiment_analyzer.py  (also save as app/nodes/sentiment.py if that's your filename)
---------------------
Structured sentiment analysis on transcript chunks via OpenAI.
"""

import json
import os
from typing import List
from openai import OpenAI


SYSTEM_PROMPT = """You are a healthcare sentiment analyst specialising in patient experience videos about Mayo Clinic.

Analyse the transcript chunks provided and return a JSON object in EXACTLY this format:
{
  "results": [
    {
      "sentiment": "positive" | "negative" | "neutral" | "mixed",
      "score": <float -1.0 to 1.0>,
      "confidence": <float 0.0 to 1.0>,
      "emotions": ["<emotion1>", "<emotion2>"],
      "themes": ["<theme1>", "<theme2>"],
      "key_phrases": ["<phrase1>", "<phrase2>"],
      "notable_quote": "<most sentiment-rich sentence from chunk, max 30 words>"
    }
  ]
}

Rules:
- The "results" array MUST have exactly one entry per chunk, in the same order.
- emotions: pick from [hope, gratitude, relief, anxiety, frustration, trust, fear, joy, sadness, anger, surprise, acceptance]
- themes: pick from [diagnosis, treatment, surgery, recovery, staff_quality, communication, wait_times, facility, cost, emotional_support, outcomes, second_opinion, cancer_care, cardiac_care, neurology, pediatrics, other]
- score: +1.0 = extremely positive, 0.0 = neutral, -1.0 = extremely negative
- Return ONLY valid JSON. No markdown fences, no explanation."""


def _extract_results_list(raw: str) -> List[dict]:
    """
    Robustly extract the list of sentiment dicts from the model response.
    Handles all known wrapping patterns:
      - {"results": [...]}
      - {"sentiments": [...]}
      - {"analyses": [...]}
      - {"chunks": [...]}
      - bare list (rare with json_object mode but handled)
    """
    raw = raw.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[Sentiment] JSON parse error: {e}\nRaw response: {raw[:300]}")
        return []

    # Already a list
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    # It's a dict — find the first value that is a non-empty list of dicts
    if isinstance(parsed, dict):
        # Try the key we asked for first
        if "results" in parsed and isinstance(parsed["results"], list):
            return [item for item in parsed["results"] if isinstance(item, dict)]
        # Fall back to any list value
        for v in parsed.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v

    print(f"[Sentiment] Unexpected response structure: {str(parsed)[:300]}")
    return []


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


SOURCE_CONFIDENCE_MODIFIER = {
    "cc_manual":      1.0,
    "cc_auto":        0.85,
    "cc_translated":  0.70,
    "whisper":        0.80,
    "metadata_only":  0.40,
    "failed":         0.20,
}

BATCH_SIZE = 5  # chunks per API call


def _call_openai(client: OpenAI, chunks: List[dict]) -> List[dict]:
    numbered = "\n\n".join(
        f"[CHUNK {i+1}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyse these {len(chunks)} chunk(s):\n\n{numbered}"},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    results = _extract_results_list(raw)
    print(f"[Sentiment] API returned {len(results)} result(s) for {len(chunks)} chunk(s)")
    return results


def _blank_sentiment(chunk: dict) -> dict:
    return {
        **chunk,
        "sentiment": "unknown",
        "score": 0.0,
        "confidence": 0.0,
        "emotions": [],
        "themes": [],
        "key_phrases": [],
        "notable_quote": "",
    }


def analyze_sentiment(state: dict) -> dict:
    """LangGraph node — adds sentiment fields to each chunk."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    chunks = state.get("chunks", [])
    results = []

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        try:
            sentiment_results = _call_openai(client, batch)
        except Exception as e:
            print(f"[Sentiment] Batch {batch_start}–{batch_start + BATCH_SIZE} failed: {type(e).__name__}: {e}")
            sentiment_results = []

        for i, chunk in enumerate(batch):
            if i < len(sentiment_results):
                sr = sentiment_results[i]
                # Validate sr is actually a dict before calling .get()
                if not isinstance(sr, dict):
                    print(f"[Sentiment] Non-dict result at index {i}: {type(sr)} — {sr!r:.100}")
                    results.append(_blank_sentiment(chunk))
                    continue

                modifier = SOURCE_CONFIDENCE_MODIFIER.get(chunk.get("source", ""), 0.60)
                raw_conf = _safe_float(sr.get("confidence", 0.5), 0.5)
                sr["confidence"] = round(raw_conf * modifier, 3)
                # Ensure score is a float too
                sr["score"] = _safe_float(sr.get("score", 0.0), 0.0)
                results.append({**chunk, **sr})
            else:
                print(f"[Sentiment] Missing result for chunk {batch_start + i} — using blank")
                results.append(_blank_sentiment(chunk))

    return {**state, "sentiments": results}