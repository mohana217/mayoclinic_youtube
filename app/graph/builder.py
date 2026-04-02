from langgraph.graph import StateGraph, END
from app.graph.state import PipelineState

from app.nodes.search import search_node
from app.nodes.transcript import transcript_node
from app.nodes.chunking import chunk_node
from app.nodes.sentiment import sentiment_node
from app.nodes.comments import comments_node
from app.nodes.aggregator import aggregator_node

def build_graph():
    builder = StateGraph(PipelineState)

    builder.add_node("search", search_node)
    builder.add_node("transcript", transcript_node)
    builder.add_node("chunk", chunk_node)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("comments", comments_node)
    builder.add_node("aggregate", aggregator_node)

    builder.set_entry_point("transcript")

    builder.add_edge("search", "transcript")
    builder.add_edge("transcript", "chunk")
    builder.add_edge("chunk", "sentiment")
    builder.add_edge("sentiment", "comments")
    builder.add_edge("comments", "aggregate")
    builder.add_edge("aggregate", END)

    return builder.compile()