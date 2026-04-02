from app.services.llm_client import get_llm
import os


llm = get_llm()

def sentiment_node(state):
    results = []

    for c in state["chunks"]:
        print(f"[Sentiment] Processing chunk for video {c['video_id']}")

        prompt = f"""
Analyze this patient experience.

Return JSON:
{{
  "sentiment": "positive / negative / neutral",
  "emotion": "",
  "topics": []
}}

Text:
{c["chunk"]}
"""

        try:
            response = llm.invoke(prompt)

            results.append({
                "video_id": c["video_id"],
                "analysis": response.content
            })

        except Exception as e:
            print("[ERROR] sentiment failed:", e)

    print("\n[DEBUG] sentiment results:\n", results)

    return {"sentiments": results}