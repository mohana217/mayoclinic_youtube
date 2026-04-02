from app.services.llm_client import get_llm
import json
import re




def clean_json_output(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def aggregator_node(state):
    llm = get_llm()

    # Step 1: sentiment counts
    positive = 0
    negative = 0

    print("\n[DEBUG] sentiments data:\n", state["sentiments"])

    positive = 0
    negative = 0

    for s in state["sentiments"]:
        text = s["analysis"].lower()

        if '"sentiment": "negative"' in text:
            negative += 1
        elif '"sentiment": "positive"' in text:
            positive += 1
      
    

    total_chunks = len(state["sentiments"])
    neutral = total_chunks - (positive + negative)

    overall_hint = f"positive_count={positive}, negative_count={negative}"

    print(f"\n[DEBUG] total_chunks={total_chunks}, positive={positive}, negative={negative}, neutral={neutral}")
    print(total_chunks, positive, negative)
    summary_hint = f"""
total_chunks={total_chunks},
positive={positive},
negative={negative},
neutral={neutral}
"""

    # Step 2: detect real negative signal
    negative_keywords = ["wait", "delay", "cost", "expensive", "billing", "bad", "poor"]

    has_negative_signal = any(
        any(word in str(s["analysis"]).lower() for word in negative_keywords)
        for s in state["sentiments"]
    )
    print(state["sentiments"])
    # Step 3: prompt (FIXED JSON ESCAPING)
    prompt = f"""
You are analyzing real patient experiences of Mayo Clinic.

RULES:
- Extract ONLY what is present in data
- DO NOT invent issues
- If no negative themes exist, return []
- If negative_signal_present is False, negative_themes MUST be []

Context:
{overall_hint}
negative_signal_present: {has_negative_signal}

Data:
sentiments: {state["sentiments"]}
comments: {state["comments"]}

Also calculate:
- total_chunks from sentiments
- number of positive, negative, and neutral chunks

For each video_id:
- determine overall sentiment
- identify the main theme

Generate:
- realistic themes
- emotional patterns
- meaningful insights

Insights must:
- explain WHY sentiment is positive/negative
- highlight patterns across patients
- not be empty

Themes should be specific and concrete.
Avoid generic phrases like "good care" or "satisfaction".
Instead use detailed themes like:
- "timely diagnosis"
- "doctor communication"
- "treatment effectiveness"

Themes must include frequency counts based on occurrences in data.

Summary hint:
{summary_hint}

Return ONLY valid JSON:
{{
  "overall_sentiment": "",

  "summary": {{
    "total_chunks": 0,
    "positive": 0,
    "negative": 0,
    "neutral": 0
  }},

  "positive_themes": [
    {{"theme": "", "count": 0}}
  ],

  "negative_themes": [
    {{"theme": "", "count": 0}}
  ],

  "emotions": [],

  "video_insights": [
    {{
      "video_id": "",
      "sentiment": "",
      "main_theme": ""
    }}
  ],

  "insights": [],

  "confidence": ""
}}
"""

    # Step 4: call LLM
    response = llm.invoke(prompt)

    print("\n[Aggregator] Raw LLM output:\n", response.content)

    # Step 5: clean + parse

    try:
      cleaned = clean_json_output(response.content)
      report = json.loads(cleaned)

      report["summary"] = {
          "total_chunks": total_chunks,
          "positive": positive,
          "negative": negative,
          "neutral": neutral
      }

    except Exception as e:
      report = {
          "error": "Invalid JSON from LLM",
          "exception": str(e),
          "raw": response.content
      }
    
    
    # Step 6: final structured output print
    print("\n[Aggregator] Parsed Report:\n", json.dumps(report, indent=2))

    return {"report": report}