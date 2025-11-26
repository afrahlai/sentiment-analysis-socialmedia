import streamlit as st
from transformers import pipeline
import yt_dlp
import pandas as pd
from langdetect import detect, LangDetectException

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Comment Sentiment Analyzer",
    page_icon="🎥",
    layout="centered"
)

# --- App Title ---
st.title("🎥 YouTube Comment Sentiment Analyzer")
st.write("Paste a YouTube video link below to fetch comments, filter English ones, and analyze their sentiment.")

# --- Load Sentiment Model (Cached) ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

model = load_model()

# --- Function to Extract Comments using yt-dlp ---
def fetch_youtube_comments(video_url, limit=50):
    """
    Extracts comments from a YouTube video using yt-dlp (no API key required).
    """
    try:
        ydl_opts = {
            "skip_download": True,
            "extract_flat": True,
            "quiet": True,
            "getcomments": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        comments = []
        if "comments" in info:
            for c in info["comments"][:limit]:
                if "text" in c:
                    comments.append(c["text"])

        return comments

    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []


# --- Function to Filter Only English Comments ---
def filter_english_comments(comments):
    """
    Keeps only comments detected as English using langdetect.
    """
    english_comments = []
    for comment in comments:
        try:
            if detect(comment) == "en":
                english_comments.append(comment)
        except LangDetectException:
            continue  # skip unrecognized language or too-short text
    return english_comments


# --- Streamlit Input UI ---
video_url = st.text_input("Enter a YouTube video URL:")
limit = st.slider("Number of comments to fetch", 10, 200, 50)

if st.button("Analyze Comments"):
    if video_url.strip():
        with st.spinner("Fetching comments..."):
            comments = fetch_youtube_comments(video_url, limit=limit)

        if comments:
            st.info(f"Fetched {len(comments)} total comments. Filtering for English...")
            english_comments = filter_english_comments(comments)

            if not english_comments:
                st.warning("No English comments found.")
            else:
                st.success(f"{len(english_comments)} English comments detected. Analyzing sentiments...")

                # Run sentiment analysis
                results = model(english_comments)

                # Create DataFrame
                df = pd.DataFrame([
                    {"Comment": c, "Sentiment": r["label"], "Confidence": round(r["score"], 2)}
                    for c, r in zip(english_comments, results)
                ])

                # Display results
                st.dataframe(df, use_container_width=True)

                # Show summary
                summary = df["Sentiment"].value_counts().reset_index()
                summary.columns = ["Sentiment", "Count"]
                st.bar_chart(summary.set_index("Sentiment"))
        else:
            st.warning("No comments found or unable to fetch comments.")
    else:
        st.warning("Please enter a valid YouTube video URL.")
