"""
ui.py  —  Streamlit frontend for Mayo Clinic YouTube Sentiment Analyzer
"""

import streamlit as st
import re
import os

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="Mayo Clinic — YouTube Sentiment Analyzer",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar: API keys ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔐 API Keys")
    st.caption("Keys are stored in session memory only and never saved.")

    openai_key = st.text_input("OpenAI API Key", type="password", key="oai_key")
    yt_key = st.text_input(
        "YouTube Data API Key (optional)",
        type="password",
        key="yt_key",
        help="Enables title/description fallback for videos with no transcript. "
             "Get one free from console.cloud.google.com",
    )
    st.divider()
    whisper_model = st.selectbox(
        "Whisper model size",
        ["base", "small", "medium"],
        index=0,
        help="Larger = better accuracy, slower. 'base' is fine for most videos.",
    )
    st.caption(
        "**Transcript strategy** (automatic):\n"
        "1. YouTube CC / auto-captions\n"
        "2. Whisper (audio download) if CC absent\n"
        "3. Video title + description (metadata)"
    )

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🏥 Mayo Clinic — YouTube Sentiment Analyzer")
st.write("Paste YouTube video URLs below to analyse patient experience sentiment.")

# ── Input ─────────────────────────────────────────────────────────────────────
video_links = st.text_area(
    "Enter YouTube URLs (one per line)",
    height=140,
    placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    url = url.strip()
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url


SOURCE_BADGE = {
    "cc_manual":     ("✅ Manual CC",    "#2ecc71"),
    "cc_auto":       ("🤖 Auto CC",      "#27ae60"),
    "cc_translated": ("🌐 Translated CC","#f39c12"),
    "whisper":       ("🎙️ Whisper",      "#8e44ad"),
    "metadata_only": ("📄 Metadata only","#e67e22"),
    "failed":        ("❌ No transcript","#e74c3c"),
}

SENTIMENT_EMOJI = {
    "positive": "😊 Positive",
    "negative": "😟 Negative",
    "neutral":  "😐 Neutral",
    "mixed":    "🔄 Mixed",
    "unknown":  "❓ Unknown",
}

ARC_EMOJI = {
    "improves":          "📈 Improves",
    "worsens":           "📉 Worsens",
    "stable":            "➡️ Stable",
    "insufficient_data": "⚠️ Too short",
}

CONFIDENCE_COLORS = {"high": "green", "medium": "orange", "low": "red"}


def sentiment_color(label: str) -> str:
    return {"positive": "#27ae60", "negative": "#c0392b", "neutral": "#7f8c8d", "mixed": "#8e44ad"}.get(label, "#7f8c8d")


def score_bar(score: float) -> str:
    """Render a simple coloured score bar using HTML."""
    pct = int((score + 1) / 2 * 100)  # –1→0%, +1→100%
    color = "#27ae60" if score >= 0 else "#c0392b"
    return f"""
    <div style="background:#eee;border-radius:6px;height:10px;width:100%;margin:4px 0 8px">
      <div style="background:{color};width:{pct}%;height:10px;border-radius:6px;"></div>
    </div>
    <small style="color:#888">{score:+.2f}</small>
    """


# ── Analyse button ────────────────────────────────────────────────────────────

if not openai_key:
    st.warning("⚠️ Enter your OpenAI API key in the sidebar to continue.")
    st.stop()

if st.button("🔍 Analyse Videos", type="primary", use_container_width=True):
    if not video_links.strip():
        st.warning("Please enter at least one video URL.")
        st.stop()

    # Set environment variables for this session
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["WHISPER_MODEL_SIZE"] = whisper_model
    if yt_key:
        os.environ["YOUTUBE_API_KEY"] = yt_key

    urls = [u.strip() for u in video_links.strip().split("\n") if u.strip()]
    videos = [{"video_id": extract_video_id(url)} for url in urls]

    with st.spinner(f"Fetching transcripts & running sentiment analysis on {len(videos)} video(s)…"):
        from main import run
        result = run(videos_override=videos)

    report = result.get("report", {})

    if "error" in report:
        st.error(f"Analysis failed: {report['error']}")
        st.stop()

    st.success("✅ Analysis complete!")
    st.divider()

    # ── 1. Overall summary banner ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    overall = report.get("overall_sentiment", "unknown")
    score   = report.get("overall_score", 0.0)
    conf    = report.get("confidence", "low")
    n_vids  = len(report.get("video_insights", {}))

    col1.metric("Overall Sentiment", SENTIMENT_EMOJI.get(overall, overall))
    col2.metric("Sentiment Score",   f"{score:+.2f}")
    col3.metric("Videos Analysed",   n_vids)
    col4.metric(
        "Confidence",
        conf.capitalize(),
        help="Based on transcript quality and model confidence scores.",
    )

    st.markdown(f"> {report.get('summary', '')}")
    st.divider()

    # ── 2. Transcript quality overview ───────────────────────────────────────
    quality = report.get("transcript_quality", {})
    if quality:
        st.subheader("📡 Transcript Source Quality")
        cols = st.columns(len(SOURCE_BADGE))
        i = 0
        for src_key, (label, color) in SOURCE_BADGE.items():
            count = quality.get(src_key, 0)
            if count > 0:
                cols[i].markdown(
                    f"<div style='background:{color}22;border-left:4px solid {color};"
                    f"padding:8px 12px;border-radius:6px;'>"
                    f"<b style='color:{color}'>{label}</b><br>{count} video(s)</div>",
                    unsafe_allow_html=True,
                )
                i += 1
        st.divider()

    # ── 3. Themes & emotions ──────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("✅ Positive Themes")
        for t in report.get("positive_themes", []):
            st.markdown(f"• {t.replace('_', ' ').title()}")

    with col_b:
        st.subheader("❌ Negative Themes")
        for t in report.get("negative_themes", []):
            st.markdown(f"• {t.replace('_', ' ').title()}")

    with col_c:
        st.subheader("💬 Top Emotions")
        for e in report.get("top_emotions", []):
            st.markdown(f"• {e.replace('_', ' ').title()}")

    st.divider()

    # ── 4. Notable quotes ─────────────────────────────────────────────────────
    quotes = report.get("notable_quotes", [])
    if quotes:
        st.subheader("💬 Notable Patient Quotes")
        for q in quotes:
            sentiment_label = q.get("sentiment", "neutral")
            color = sentiment_color(sentiment_label)
            theme = q.get("theme", "").replace("_", " ").title()
            st.markdown(
                f"<blockquote style='border-left:4px solid {color};padding:8px 16px;"
                f"background:{color}11;border-radius:4px;'>"
                f"<em>\"{q.get('quote', '')}\"</em><br>"
                f"<small style='color:#888'>{SENTIMENT_EMOJI.get(sentiment_label, sentiment_label)} · {theme}</small>"
                f"</blockquote>",
                unsafe_allow_html=True,
            )
        st.divider()

    # ── 5. Per-video breakdown ────────────────────────────────────────────────
    video_insights = report.get("video_insights", {})
    if video_insights:
        st.subheader("🎥 Per-Video Breakdown")
        for vid, info in video_insights.items():
            v_sent   = info.get("sentiment", "unknown")
            v_score  = info.get("score", 0.0)
            v_arc    = info.get("arc", "stable")
            v_source = info.get("transcript_source", "unknown")
            v_conf   = info.get("avg_confidence", 0.0)
            src_label, src_color = SOURCE_BADGE.get(v_source, ("Unknown", "#95a5a6"))

            with st.expander(
                f"📺 {vid}  —  {SENTIMENT_EMOJI.get(v_sent, v_sent)}  |  {ARC_EMOJI.get(v_arc, v_arc)}"
            ):
                c1, c2, c3 = st.columns(3)
                c1.markdown(
                    f"<span style='background:{src_color}22;color:{src_color};"
                    f"padding:2px 8px;border-radius:4px;font-size:0.85em'>{src_label}</span>",
                    unsafe_allow_html=True,
                )
                c2.markdown(f"**Chunks analysed:** {info.get('chunk_count', 0)}")
                c3.markdown(f"**Avg confidence:** {v_conf:.0%}")

                st.markdown(f"**Score:** {score_bar(v_score)}", unsafe_allow_html=True)
                st.markdown(f"**Top themes:** {', '.join(t.replace('_', ' ').title() for t in info.get('top_themes', []))}")
                st.markdown(f"**Top emotions:** {', '.join(e.title() for e in info.get('top_emotions', []))}")
                st.markdown(
                    f"[▶️ Open on YouTube](https://www.youtube.com/watch?v={vid})",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── 6. Insights ───────────────────────────────────────────────────────────
    insights = report.get("insights", [])
    if insights:
        st.subheader("💡 Key Insights")
        for i in insights:
            st.info(i)

    # ── 7. Raw report (collapsed) ─────────────────────────────────────────────
    with st.expander("🔧 Raw report JSON (for debugging)"):
        st.json(report)
