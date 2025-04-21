"""Outlier Radar – Streamlit app for spotting over‑performing YouTube videos
and generating title/thumbnail optimisation ideas.

Author: Toby (Katlein Media)
"""

from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from typing import List, Dict

# ---------------------- CONFIG ----------------------
VIEW_MULTIPLIER = 2.5  # × median views to count as outlier

# ---------------------- HELPERS ---------------------
def get_youtube_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)

def fetch_channel_uploads(youtube, channel_id: str, max_results: int = 50) -> List[Dict]:
    resp = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    uploads_id = resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    videos, next_page = [], None
    while len(videos) < max_results:
        pl_resp = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_id,
            maxResults=min(50, max_results - len(videos)),
            pageToken=next_page,
        ).execute()
        videos.extend(pl_resp["items"])
        next_page = pl_resp.get("nextPageToken")
        if not next_page:
            break
    return videos

def video_stats_batch(youtube, ids: List[str]):
    stats = []
    for i in range(0, len(ids), 50):
        resp = youtube.videos().list(part="statistics,snippet", id=",".join(ids[i:i+50])).execute()
        stats.extend(resp["items"])
    return stats

def build_dataframe(items: List[Dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        sid = it["id"]
        stats = it["statistics"]
        snippet = it["snippet"]
        rows.append({
            "video_id": sid,
            "title": snippet["title"],
            "published": snippet["publishedAt"],
            "views": int(stats.get("viewCount", 0)),
        })
    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["age_hours"] = (pd.Timestamp.utcnow() - df["published"]).dt.total_seconds() / 3600
    return df

def compute_outliers(df: pd.DataFrame) -> pd.DataFrame:
    recent = df[df["age_hours"] <= 24]
    median_views = recent["views"].median() or 1
    recent["outlier_score"] = recent["views"] / median_views
    return recent[recent["outlier_score"] >= VIEW_MULTIPLIER].copy()

def rewrite_title(openai_key: str, title: str) -> str:
    if not openai_key:
        return "(OpenAI key missing – can't rewrite)"
    import openai
    openai.api_key = openai_key
    prompt = (
        "Rewrite this YouTube title in 12 words or fewer, keep the core hook, "
        "avoid ALL CAPS, make it feel curiosity-driven but not clickbaity:\n" + title
    )
    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=24,
        temperature=0.7,
    )
    return resp.choices[0].text.strip()

# ---------------------- UI --------------------------
st.set_page_config(page_title="Outlier Radar", layout="centered")
st.title("📈 Outlier Radar – YouTube Growth Dashboard")

# Secure key handling
api_key_input = st.sidebar.text_input("YouTube API Key (optional override)", type="password")
openai_key_input = st.sidebar.text_input("OpenAI API Key (optional override)", type="password")

api_key = api_key_input or st.secrets.get("YT_API_KEY", "")
openai_key = openai_key_input or st.secrets.get("OPENAI_API_KEY", "")

channel_id = st.sidebar.text_input("Channel ID (UC…)")
if st.sidebar.button("Fetch Data") and api_key and channel_id:
    yt = get_youtube_client(api_key)
    with st.spinner("Fetching latest videos…"):
        uploads = fetch_channel_uploads(yt, channel_id, max_results=100)
        stats = video_stats_batch(yt, [v["contentDetails"]["videoId"] for v in uploads])
        df = build_dataframe(stats)

    st.subheader("Latest Videos")
    st.dataframe(df[["title", "views", "age_hours"]].head(20))

    outliers = compute_outliers(df)
    if outliers.empty:
        st.success("No outliers in the last 24 hours 🚀")
    else:
        st.subheader("🚀 Potential Outliers")
        for _, row in outliers.iterrows():
            st.markdown(f"### [{row['title']}](https://youtu.be/{row['video_id']})")
            st.write(f"**Views**: {row['views']:,} | **Outlier Score**: {row['outlier_score']:.2f}×")
            suggested = rewrite_title(openai_key, row["title"])
            st.write("**AI Suggestion:**", suggested)
            st.image(f"https://i.ytimg.com/vi/{row['video_id']}/hqdefault.jpg", width=320)
            st.markdown("---")

st.caption("© 2025 Katlein Media – quietly crafted")
