"""
builder.py
----------
Builds the LangGraph pipeline connecting all nodes in sequence:
  fetch_transcripts → chunk_transcripts → analyze_sentiment → build_report
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any

from app.nodes.transcript import fetch_transcripts
from app.nodes.chunking import chunk_transcripts
from app.nodes.sentiment import analyze_sentiment
from app.nodes.aggregator import build_report


# ── State schema ──────────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    videos: List[dict]         # [{"video_id": "..."}]
    transcripts: List[dict]    # output of fetch_transcripts
    chunks: List[dict]         # output of chunk_transcripts
    sentiments: List[dict]     # output of analyze_sentiment
    report: dict               # output of build_report
    query: str                 # optional — not used by current nodes


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("fetch_transcripts", fetch_transcripts)
    g.add_node("chunk_transcripts", chunk_transcripts)
    g.add_node("analyze_sentiment", analyze_sentiment)
    g.add_node("build_report",      build_report)

    g.set_entry_point("fetch_transcripts")
    g.add_edge("fetch_transcripts", "chunk_transcripts")
    g.add_edge("chunk_transcripts",  "analyze_sentiment")
    g.add_edge("analyze_sentiment",  "build_report")
    g.add_edge("build_report",       END)

    return g.compile()
