from typing import TypedDict, List, Dict

class PipelineState(TypedDict):
    query: str
    videos: List[Dict]
    transcripts: List[Dict]
    chunks: List[Dict]
    sentiments: List[Dict]
    comments: List[Dict]
    report: Dict