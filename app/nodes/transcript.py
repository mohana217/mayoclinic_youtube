from youtube_transcript_api import YouTubeTranscriptApi

# make sure this exists
from app.services.youtube_client import get_comments


def transcript_node(state):
    transcripts = []
    api = YouTubeTranscriptApi()

    for v in state["videos"]:
        video_id = v["video_id"]
        print(f"[Transcript] Fetching transcript for {video_id}")

        try:
            # ✅ Try transcript with multiple languages
            t = api.fetch(video_id, languages=["en", "en-US", "hi"])
            text = " ".join([x.text for x in t])

            transcripts.append({
                "video_id": video_id,
                "text": text
            })

        except Exception as e:
            print(f"[ERROR] Transcript failed for {video_id}: {e}")

            # ✅ Fallback to comments
            print(f"[Fallback] Using comments for {video_id}")

            try:
                comments = get_comments(video_id)

                if comments:
                    transcripts.append({
                        "video_id": video_id,
                        "text": " ".join(comments)
                    })
                else:
                    print(f"[WARNING] No comments available for {video_id}")

            except Exception as e2:
                print(f"[ERROR] Comments also failed for {video_id}: {e2}")

    print("\n[DEBUG] transcripts:\n", transcripts)

    return {"transcripts": transcripts}