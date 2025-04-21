"""Outlier Radar 2.1 – Deep‑dive YouTube idea hunter
Multi‑channel scan (last 50 videos each) • Flags lifetime outliers
Adds engagement ratios + thumbnail hue insights.
Author: Toby (Katlein Media)
"""

from __future__ import annotations
import os, io, requests
from typing import List, Dict
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from PIL import Image

# ------------------ CONFIG ------------------
LAST_N_VIDEOS = 50          # videos per channel
OUTLIER_MULTIPLIER = 2.0    # views / median
STOPWORDS = {
    'the','a','and','of','to','in','my','your','our','why','how','what','on','for','with','at','is','it','this','that','an','i','you','we'
}
HUE_NAMES = [
    (0, 15, 'red'), (15, 40, 'orange'), (40, 65, 'yellow'), (65, 150, 'green'),
    (150, 255, 'cyan'), (255, 300, 'blue'), (300, 345, 'magenta'), (345, 360, 'red')
]

# --------------- YT HELPERS -----------------

def yt_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def fetch_video_ids(youtube, channel_id: str, max_results: int = LAST_N_VIDEOS) -> List[str]:
    ids: List[str] = []
    next_page = None
    while len(ids) < max_results:
        resp = youtube.search().list(
            part="id",
            channelId=channel_id,
            maxResults=min(50, max_results - len(ids)),
            order="date",
            type="video",
            pageToken=next_page,
        ).execute()
        ids.extend([item["id"]["videoId"] for item in resp["items"]])
        next_page = resp.get("nextPageToken")
        if not next_page:
            break
    return ids


def fetch_stats_batch(youtube, vid_ids: List[str]):
    stats = []
    for i in range(0, len(vid_ids), 50):
        chunk = vid_ids[i : i + 50]
        resp = youtube.videos().list(
            part="statistics,snippet",
            id=",".join(chunk),
            maxResults=len(chunk),
        ).execute()
        stats.extend(resp["items"])
    return stats

# ------------- ANALYSIS HELPERS -------------

def build_df(channel_id: str, items: List[Dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        snip = it["snippet"]
        stats = it["statistics"]
        views = int(stats.get("viewCount", 0)) or 1  # avoid div/0
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))
        rows.append(
            {
                "channel_id": channel_id,
                "video_id": it["id"],
                "title": snip["title"],
                "published": snip["publishedAt"],
                "views": views,
                "likes": likes,
                "comments": comments,
                "like_ratio": likes / views,
                "comment_ratio": comments / views,
            }
        )
    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    out_frames = []
    for cid, grp in df.groupby("channel_id"):
        median_views = grp["views"].median() or 1
        grp = grp.assign(outlier_score=grp["views"] / median_views)
        out_frames.append(grp[grp["outlier_score"] >= OUTLIER_MULTIPLIER])
    return pd.concat(out_frames) if out_frames else pd.DataFrame()


def rewrite_title(openai_key: str, title: str) -> str:
    if not openai_key:
        return "(Add OpenAI key to get AI suggestions)"
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


def top_title_words(df: pd.DataFrame, top_k: int = 5) -> List[str]:
    text = " ".join(df["title"].str.lower().tolist())
    words = [w.strip("()[]{}:,.!?\"'") for w in text.split()]
    counts: Dict[str,int] = {}
    for w in words:
        if w and w not in STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    return sorted(counts, key=counts.get, reverse=True)[:top_k]


def hue_name_from_rgb(rgb):
    # convert to HSV hue 0‑360
    import colorsys
    r, g, b = [x / 255 for x in rgb]
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    deg = h * 360
    for low, high, name in HUE_NAMES:
        if low <= deg < high:
            return name
    return "unknown"


def dominant_hue(video_id: str) -> str:
    url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    try:
        img_bytes = requests.get(url, timeout=5).content
        img = Image.open(io.BytesIO(img_bytes)).resize((50, 50))
        # get most common pixel
        pixels = list(img.getdata())
        rgb = max(set(pixels), key=pixels.count)
        return hue_name_from_rgb(rgb)
    except Exception:
        return "error"

# ------------------- UI ---------------------
st.set_page_config(page_title="Outlier Radar 2.1", layout="wide")
st.title("🎯 Outlier Radar 2.1 – Deep Insights")

api_key_input = st.sidebar.text_input("YouTube API Key (optional override)", type="password")
openai_key_input = st.sidebar.text_input("OpenAI API Key (optional override)", type="password")
api_key = api_key_input or st.secrets.get("YT_API_KEY", "")
openai_key = openai_key_input or st.secrets.get("OPENAI_API_KEY", "")

channels_text = st.sidebar.text_area("Channel IDs (one per line)")
if st.sidebar.button("Run Analysis") and api_key and channels_text.strip():
    yt = yt_client(api_key)
    channel_ids = [c.strip() for c in channels_text.splitlines() if c.strip()]
    bar = st.progress(0.0)
    all_df = pd.DataFrame()
    for idx, cid in enumerate(channel_ids, start=1):
        bar.progress(idx / len(channel_ids), text=f"Fetching {cid}…")
        try:
            vids = fetch_video_ids(yt, cid)
            stats = fetch_stats_batch(yt, vids)
            df_chan = build_df(cid, stats)
            all_df = pd.concat([all_df, df_chan])
        except Exception as e:
            st.error(f"Failed {cid}: {e}")
    bar.empty()
    if all_df.empty:
        st.warning("No data loaded.")
        st.stop()

    out_df = detect_outliers(all_df)
    if out_df.empty():
        st.info("No outliers under current settings.")
    else:
        st.subheader("🚀 Lifetime Outliers")
        for _, row in out_df.sort_values("outlier_score", ascending=False).iterrows():
            st.markdown(f"### [{row['title']}](https://youtu.be/{row['video_id']})")
            st.write(
                f"Channel: `{row['channel_id']}` | Views: {row['views']:,} | "
                f"Score: {row['outlier_score']:.2f}× | "
                f"👍 {row['like_ratio']*100:.1f}% | 💬 {row['comment_ratio']*100:.2f}%"
            )
            suggestion = rewrite_title(openai_key, row["title"])
            st.write("**AI Title Idea:**", suggestion)
            hue = dominant_hue(row["video_id"])
            st.write(f"Thumbnail hue: **{hue}**")
            st.image(f"https://i.ytimg.com/vi/{row['video_id']}/hqdefault.jpg", width=320)
            st.markdown("---")

        st.subheader("📊 Engagement Averages by Channel")
        agg = all_df.groupby("channel_id").agg(
            median_like_ratio=("like_ratio", "median"),
            median_comment_ratio=("comment_ratio", "median"),
        )
        st.dataframe(agg.style.format({
            "median_like_ratio": "{:.2%}",
            "median_comment_ratio": "{:.2%}"
        }))

        st.subheader("🎨 Dominant Thumbnail Hues (Outliers)")
        hue_counts = out_df.assign(hue=out_df["video_id"].apply(dominant_hue)).groupby("hue").size().sort_values(ascending=False)
        st.write(hue_counts.to_frame("count"))

        st.subheader("🔎 Title Word Patterns in Outliers")
        for cid, grp in out_df.groupby("channel_id"):
            words = ", ".join(top_title_words(grp)) or "(no data)"
            st.write(f"**{cid}** – common winning words: {words}")

st.caption("© 2025 Katlein Media – quietly crafted")
