from app.graph.builder import build_graph

def run(videos_override=None):
    graph = build_graph()

    if videos_override:
        return graph.invoke({
            "videos": videos_override,
            "transcripts": [],
            "chunks": [],
            "sentiments": [],
            "report": {}
        })

    return graph.invoke({
        "query": "",
        "videos": videos_override,
        "transcripts": [],
        "chunks": [],
        "sentiments": [],
        "report": {}
    })

if __name__ == "__main__":
    result = run()

    print("\nFINAL REPORT:\n")
    print(result["report"])