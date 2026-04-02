from app.services.youtube_client import get_comments

def comments_node(state):
    data = []

    for v in state["videos"]:
        comments = get_comments(v["video_id"])

        data.append({
            "video_id": v["video_id"],
            "comments": comments
        })

    return {"comments": data}