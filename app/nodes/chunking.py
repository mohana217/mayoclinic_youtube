def chunk_node(state):
    chunks = []

    for t in state["transcripts"]:
        words = t["text"].split()
        size = 200

        for i in range(0, len(words), size):
            chunks.append({
                "video_id": t["video_id"],
                "chunk": " ".join(words[i:i+size])
            })
    print("\n[DEBUG] chunks:\n", chunks)
        
    return {"chunks": chunks}