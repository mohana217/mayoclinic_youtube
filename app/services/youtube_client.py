from googleapiclient.discovery import build
from app.config.settings import YOUTUBE_API_KEY

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def search_videos(query: str, max_results=5):
    response = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    ).execute()

    return [
        {
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"]
        }
        for item in response["items"]
    ]


def get_comments(video_id: str):
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=20
        ).execute()

        return [
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in response["items"]
        ]
    except:
        return []