from app.services.youtube_client import search_videos

def search_node(state):
    videos = search_videos(state["query"])
    return {"videos": videos}