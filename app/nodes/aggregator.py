"""
aggregator.py
-----------------
Aggregates per-chunk sentiment into a structured report.
"""

from collections import defaultdict, Counter
from typing import List, Dict, Any


def _weighted_score(chunks: List[dict]) -> float:
    total_weight = sum(c.get("confidence", 0) for c in chunks)
    if total_weight == 0:
        return 0.0
    return round(
        sum(c.get("score", 0) * c.get("confidence", 0) for c in chunks) / total_weight, 3
    )

def _label(score: float) -> str:
    if score >= 0.35:  return "positive"
    if score <= -0.35: return "negative"
    if abs(score) < 0.15: return "neutral"
    return "mixed"

def _arc(chunks: List[dict]) -> str:
    if len(chunks) < 3:
        return "insufficient_data"
    scores = [c.get("score", 0) for c in chunks]
    n = max(len(scores) // 3, 1)
    diff = sum(scores[-n:]) / n - sum(scores[:n]) / n
    if diff > 0.2:  return "improves"
    if diff < -0.2: return "worsens"
    return "stable"

def _top(counter: Counter, n: int = 5) -> List[str]:
    return [item for item, _ in counter.most_common(n)]


def build_report(state: dict) -> dict:
    sentiments: List[dict] = state.get("sentiments", [])
    transcripts: List[dict] = state.get("transcripts", [])

    # ── Transcript quality summary ─────────────────────────────────────────────
    source_counter: Counter = Counter()
    source_by_vid: Dict[str, str] = {}
    failed_vids = []
    for t in transcripts:
        src = t.get("source", "failed")
        source_counter[src] += 1
        source_by_vid[t["video_id"]] = src
        if src == "failed" or not t.get("usable"):
            failed_vids.append(t["video_id"])

    # ── Meaningful failure message ─────────────────────────────────────────────
    if not sentiments:
        reasons = []
        if failed_vids:
            reasons.append(
                f"No transcript could be retrieved for: {', '.join(failed_vids)}. "
                "This usually means: (a) the video has no CC and Whisper/ffmpeg is not installed, "
                "or (b) yt-dlp could not download the audio (age-restricted or private video). "
                "Check the terminal/logs for [CC], [Whisper], [Metadata] lines to see exactly which tier failed."
            )
        else:
            reasons.append("Transcripts were fetched but chunking/sentiment produced no output. Check logs.")
        return {**state, "report": {"error": " | ".join(reasons), "transcript_quality": dict(source_counter)}}

    # ── Global aggregation ─────────────────────────────────────────────────────
    overall_score = _weighted_score(sentiments)
    overall_label = _label(overall_score)
    avg_conf = round(sum(c.get("confidence", 0) for c in sentiments) / len(sentiments), 3)

    pos_themes: Counter = Counter()
    neg_themes: Counter = Counter()
    all_emotions: Counter = Counter()

    for chunk in sentiments:
        score = chunk.get("score", 0)
        for t in chunk.get("themes", []):
            if score > 0.1:   pos_themes[t] += 1
            elif score < -0.1: neg_themes[t] += 1
        for e in chunk.get("emotions", []):
            all_emotions[e] += 1

    # Notable quotes
    quotes = sorted(
        [
            {
                "quote": c.get("notable_quote", ""),
                "sentiment": c.get("sentiment", ""),
                "theme": (c.get("themes") or ["other"])[0],
                "_score": abs(c.get("score", 0)),
            }
            for c in sentiments
            if c.get("notable_quote") and c.get("confidence", 0) > 0.4
        ],
        key=lambda x: x["_score"],
        reverse=True,
    )[:5]
    for q in quotes:
        del q["_score"]

    # ── Per-video breakdown ────────────────────────────────────────────────────
    video_chunks: Dict[str, List[dict]] = defaultdict(list)
    for chunk in sentiments:
        video_chunks[chunk["video_id"]].append(chunk)

    video_insights: Dict[str, Any] = {}
    for vid, chunks in video_chunks.items():
        sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
        v_score = _weighted_score(sorted_chunks)
        v_themes: Counter = Counter()
        v_emotions: Counter = Counter()
        for c in sorted_chunks:
            for t in c.get("themes", []): v_themes[t] += 1
            for e in c.get("emotions", []): v_emotions[e] += 1

        video_insights[vid] = {
            "sentiment": _label(v_score),
            "score": v_score,
            "arc": _arc(sorted_chunks),
            "top_themes": _top(v_themes, 4),
            "top_emotions": _top(v_emotions, 3),
            "transcript_source": source_by_vid.get(vid, "unknown"),
            "chunk_count": len(sorted_chunks),
            "avg_confidence": round(
                sum(c.get("confidence", 0) for c in sorted_chunks) / len(sorted_chunks), 3
            ),
        }

    # ── Confidence tier ────────────────────────────────────────────────────────
    good = source_counter.get("cc_manual", 0) + source_counter.get("cc_auto", 0) + source_counter.get("whisper", 0)
    total = len(transcripts) or 1
    if avg_conf >= 0.70 and good / total >= 0.8:
        conf_tier = "high"
    elif avg_conf >= 0.45 and good / total >= 0.5:
        conf_tier = "medium"
    else:
        conf_tier = "low"

    # ── Insights ───────────────────────────────────────────────────────────────
    insights = []
    if "staff_quality" in pos_themes:
        insights.append("Staff quality is a recurring positive theme — patients frequently praise care team interactions.")
    if "wait_times" in neg_themes:
        insights.append("Wait times appear as a pain point in negative-sentiment segments.")
    if "outcomes" in pos_themes:
        insights.append("Patients frequently highlight positive treatment outcomes and recovery experiences.")
    if source_counter.get("metadata_only", 0):
        insights.append(
            f"{source_counter['metadata_only']} video(s) used metadata only (no audio transcript). "
            "Add a YouTube Data API key or ensure ffmpeg is installed for better coverage."
        )
    if source_counter.get("whisper", 0):
        insights.append(
            f"{source_counter['whisper']} video(s) were transcribed via Whisper (no CC available)."
        )
    if failed_vids:
        insights.append(
            f"⚠️ {len(failed_vids)} video(s) had no retrievable transcript: {', '.join(failed_vids)}. "
            "Check the terminal logs for the specific failure reason."
        )

    summary = (
        f"Analysed {len(transcripts)} video(s) across {len(sentiments)} chunks. "
        f"Overall sentiment: {overall_label} (score {overall_score:+.2f}). "
        f"Confidence: {conf_tier}."
    )

    report = {
        "overall_sentiment": overall_label,
        "overall_score": overall_score,
        "overall_confidence": avg_conf,
        "transcript_quality": dict(source_counter),
        "positive_themes": _top(pos_themes, 6),
        "negative_themes": _top(neg_themes, 6),
        "top_emotions": _top(all_emotions, 5),
        "notable_quotes": quotes,
        "video_insights": video_insights,
        "summary": summary,
        "insights": insights,
        "confidence": conf_tier,
    }

    return {**state, "report": report}