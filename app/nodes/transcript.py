from youtube_transcript_api import YouTubeTranscriptApi

def transcript_node(state):
    transcripts = []

    api = YouTubeTranscriptApi()

    for v in state["videos"]:
        video_id = v["video_id"]
        print(f"[Transcript] Fetching transcript for {video_id}")

        try:
            t = api.fetch(video_id)

            text = " ".join([x.text for x in t])

            transcripts.append({
                "video_id": video_id,
                "text": text
            })

        except Exception as e:
            print(f"[ERROR] Transcript failed for {video_id}: {e}")

    print("\n[DEBUG] transcripts:\n", transcripts)

    return {"transcripts": transcripts}