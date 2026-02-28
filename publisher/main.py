"""
Instagram Publisher with AI-generated captions via Reka.

Pipeline:
  1. Upload the clip to Reka Vision Agent
  2. Ask Reka to generate a social media caption based on the video
  3. Post the clip to Instagram Reels with the AI caption

Required .env variables:
  - REKA_API_KEY: API key from https://platform.reka.ai/
  - IG_USERNAME / IG_PASSWORD: Instagram credentials
"""

import json
import logging
import os
import sys
import time

import requests
from dotenv import load_dotenv
from instagrapi import Client

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reka Vision Agent (REST API)
# ---------------------------------------------------------------------------

REKA_BASE_URL = "https://vision-agent.api.reka.ai"


def _get_reka_headers() -> dict:
    """Build headers for the Reka Vision Agent API."""
    api_key = os.environ.get("REKA_API_KEY")
    if not api_key:
        raise ValueError("REKA_API_KEY is not set in .env")
    return {
        "X-Api-Key": api_key,
    }


def _upload_video_to_reka(video_path: str) -> str:
    """Upload a local video file to Reka Vision Agent and return the video_id."""
    url = f"{REKA_BASE_URL}/videos/upload"
    video_name = os.path.basename(video_path)

    data = {
        "index": True,
        "enable_thumbnails": False,
        "video_name": video_name,
    }

    logger.info("[reka] Uploading video to Reka: %s", video_name)
    with open(video_path, "rb") as f:
        files = {"file": (video_name, f, "video/mp4")}
        response = requests.post(
            url, headers=_get_reka_headers(), data=data, files=files, timeout=120,
        )

    if response.status_code != 200:
        raise RuntimeError(f"Reka upload failed ({response.status_code}): {response.text}")

    result = response.json()
    video_id = result.get("video_id", "")
    logger.info("[reka] Video uploaded: video_id=%s", video_id)
    return video_id


def _wait_for_indexing(video_id: str, max_wait: int = 120) -> None:
    """Poll Reka until the video is indexed and ready for Q&A."""
    url = f"{REKA_BASE_URL}/videos/{video_id}"
    headers = _get_reka_headers()

    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                status = resp.json().get("status", "")
                if status in ("indexed", "ready", "completed"):
                    logger.info("[reka] Video indexed and ready")
                    return
                logger.info("[reka] Video status: %s, waiting...", status)
        except requests.RequestException:
            pass
        time.sleep(5)

    logger.warning("[reka] Timed out waiting for indexing, proceeding anyway")


def _ask_reka_for_caption(video_id: str, emotion_context: str = "") -> str:
    """Ask Reka Vision Agent to generate a social media caption for the clip."""
    url = f"{REKA_BASE_URL}/qa/chat"

    prompt = (
        "You are a social media manager for a Twitch streamer. "
        "Watch this clip and write a short, engaging Instagram Reels caption for it. "
        "The caption should be 1-2 sentences max, include 2-3 relevant emojis, "
        "and end with 3-5 hashtags. Make it feel authentic and hype, not corporate. "
        "Do NOT include quotes around the caption."
    )

    if emotion_context:
        prompt += f"\n\nContext: The clip's dominant emotions are: {emotion_context}"

    payload = {
        "video_id": video_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    headers = {**_get_reka_headers(), "Content-Type": "application/json"}

    logger.info("[reka] Asking Reka for a caption...")
    response = requests.post(url, json=payload, headers=headers, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"Reka Q&A failed ({response.status_code}): {response.text}")

    result = response.json()
    # The response structure from /qa/chat contains the assistant message
    caption = result.get("message", result.get("answer", ""))

    # If it's nested inside a messages array
    if not caption and "messages" in result:
        msgs = result["messages"]
        for msg in reversed(msgs):
            if msg.get("role") == "assistant":
                caption = msg.get("content", "")
                break

    if not caption:
        logger.warning("[reka] Empty caption from Reka, using raw response: %s", result)
        caption = "Stream highlight 🔥 #twitch #gaming #clips"

    logger.info("[reka] Generated caption: %s", caption)
    return caption.strip()


def generate_caption(video_path: str, emotion_report_path: str | None = None) -> str:
    """
    Generate an AI caption for a video clip using Reka.

    Args:
        video_path: Path to the video file (.mp4)
        emotion_report_path: Optional path to the emotion_report.json for context

    Returns:
        The generated caption string.
    """
    # Load emotion context if available
    emotion_context = ""
    if emotion_report_path and os.path.exists(emotion_report_path):
        try:
            with open(emotion_report_path, encoding="utf-8") as f:
                report = json.load(f)
            emotions = report.get("emotions", {})
            top_emotions = sorted(
                [(e, c) for e, c in emotions.items() if c > 0],
                key=lambda x: x[1], reverse=True,
            )[:3]
            if top_emotions:
                emotion_context = ", ".join(f"{e} ({c}x)" for e, c in top_emotions)
        except Exception as exc:
            logger.warning("[reka] Could not load emotion report: %s", exc)

    # Upload video to Reka
    video_id = _upload_video_to_reka(video_path)

    # Wait for it to be indexed
    _wait_for_indexing(video_id)

    # Ask Reka for a caption
    caption = _ask_reka_for_caption(video_id, emotion_context)

    return caption


# ---------------------------------------------------------------------------
# Instagram publishing
# ---------------------------------------------------------------------------

def login_instagram() -> Client:
    """Authenticate with Instagram using credentials from .env."""
    username = os.environ.get("IG_USERNAME")
    password = os.environ.get("IG_PASSWORD")

    if not username or not password:
        logger.error("IG_USERNAME and IG_PASSWORD must be set in .env")
        sys.exit(1)

    logger.info("Logging into Instagram as %s...", username)
    cl = Client()

    try:
        cl.login(username, password)
        logger.info("Instagram login successful!")
        return cl
    except Exception as e:
        logger.error("Instagram login failed: %s", e)
        sys.exit(1)


def post_clip(cl: Client, video_path: str, caption: str = "") -> None:
    """Post a local video to Instagram as a Reel."""
    logger.info("Publishing clip as Reel: %s", video_path)
    if not os.path.exists(video_path):
        logger.error("File does not exist: %s", video_path)
        return

    try:
        media = cl.clip_upload(video_path, caption)
        logger.info("Clip posted successfully! Media ID: %s", media.pk)
    except Exception as e:
        logger.error("Failed to post clip: %s", e)


def publish_highlight(
    video_path: str,
    emotion_report_path: str | None = None,
    custom_caption: str | None = None,
) -> None:
    """
    Full pipeline: generate AI caption → upload to Instagram Reels.

    Args:
        video_path: Path to the highlight video (.mp4)
        emotion_report_path: Optional emotion_report.json for better captions
        custom_caption: If provided, skip Reka and use this caption instead
    """
    if custom_caption:
        caption = custom_caption
    else:
        try:
            caption = generate_caption(video_path, emotion_report_path)
        except Exception as exc:
            logger.exception("[publisher] Reka caption generation failed, using default")
            caption = "Stream highlights 🔥 #twitch #gaming #clips #streamer"

    logger.info("[publisher] Caption: %s", caption)

    ig_client = login_instagram()
    post_clip(ig_client, video_path, caption)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <path_to_video> [path_to_emotion_report.json] [custom_caption]")
        print()
        print("Examples:")
        print('  python main.py clips/clip_1/highlight.mp4')
        print('  python main.py clips/clip_1/highlight.mp4 clips/clip_1/emotion_report.json')
        print('  python main.py clips/clip_1/highlight.mp4 - "My custom caption 🎮"')
        sys.exit(1)

    video = sys.argv[1]
    emotion_json = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    caption_override = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else None

    publish_highlight(video, emotion_json, caption_override)
