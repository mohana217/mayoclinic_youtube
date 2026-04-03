import streamlit as st
from main import run   
import re
import os



st.sidebar.title("🔐 Configuration")
st.sidebar.info("Your API key is not stored and is used only for this session.")

user_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password"
)

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")

st.title("🎥 YouTube Video Sentiment Analyzer")
st.write("Paste YouTube video links to analyze patient experiences")

# Input box
video_links = st.text_area(
    "Enter YouTube URLs (one per line)",
    height=150
)


def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1]
    return url.strip()

if not user_api_key:
    st.error("Please enter your OpenAI API key")
    st.stop()

# Set API key for this session
os.environ["OPENAI_API_KEY"] = user_api_key


if st.button("Analyze Videos"):

    if not video_links.strip():
        st.warning("Please enter at least one video URL")
    else:
        with st.spinner("Running analysis..."):

            urls = video_links.strip().split("\n")
            videos = [{"video_id": extract_video_id(url)} for url in urls]

            import os

            if not user_api_key:
                st.error("Please enter your OpenAI API key")
                st.stop()

            os.environ["OPENAI_API_KEY"] = user_api_key

            # 👇 Call your existing pipeline WITHOUT modifying it
            result = run(videos_override=videos)

            report = result["report"]

        st.success("Analysis Complete!")

        # ---- Display Results ---- #

        st.subheader("📊 Overall Sentiment")
        st.write(report.get("overall_sentiment"))

        st.subheader("📈 Summary")
        st.json(report.get("summary"))

        st.subheader("✅ Positive Themes")
        st.json(report.get("positive_themes"))

        st.subheader("❌ Negative Themes")
        st.json(report.get("negative_themes"))

        st.subheader("😊 Emotions")
        st.write(report.get("emotions"))

        st.subheader("🎥 Video Insights")
        st.json(report.get("video_insights"))

        st.subheader("💡 Insights")
        for i in report.get("insights", []):
            st.write(f"- {i}")

        st.subheader("🔍 Confidence")
        st.write(report.get("confidence"))