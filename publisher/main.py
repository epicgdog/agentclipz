"""
Instagram Publisher with AI-generated captions via Reka Chat API.

Pipeline:
  1. Read the emotion_report.json (transcript + emotions from Modulate)
  2. Send the transcript context to Reka Chat to generate a social media caption
  3. Post the clip to Instagram Reels with the AI caption

Required .env variables:
  - REKA_API_KEY: API key from https://platform.reka.ai/
  - IG_USERNAME / IG_PASSWORD: Instagram credentials
"""

import json
import logging
import os
import sys

from dotenv import load_dotenv
from reka import ChatMessage
from reka.client import Reka
from instagrapi import Client

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reka Chat API — caption generation from transcript
# ---------------------------------------------------------------------------

def generate_caption(
    emotion_report_path: str | None = None,
    streamer_name: str | None = None,
) -> str:
    """
    Generate an Instagram caption using Reka Chat API.

    Uses the transcript and emotion data from emotion_report.json
    to write a caption — no video upload needed.
    """
    api_key = os.environ.get("REKA_API_KEY")
    if not api_key:
        raise ValueError("REKA_API_KEY is not set in .env")

    # Build context from the emotion report
    context_parts = []

    if emotion_report_path and os.path.exists(emotion_report_path):
        with open(emotion_report_path, encoding="utf-8") as f:
            report = json.load(f)

        # Get top emotions
        emotions = report.get("emotions", {})
        top_emotions = sorted(
            [(e, c) for e, c in emotions.items() if c > 0],
            key=lambda x: x[1], reverse=True,
        )[:5]
        if top_emotions:
            context_parts.append(
                "Dominant emotions: " + ", ".join(f"{e} ({c}x)" for e, c in top_emotions)
            )

        # Get transcript text
        utterances = report.get("utterances", [])
        if utterances:
            transcript = " ".join(u.get("text", "") for u in utterances)
            context_parts.append(f"Transcript: \"{transcript}\"")

    streamer = streamer_name or os.environ.get("TARGET_CHANNEL", "the streamer")
    context = "\n".join(context_parts) if context_parts else "A fun Twitch stream clip"

    prompt = (
        f"You are a social media manager for a Twitch streamer named {streamer}. "
        f"Based on the following clip data, write a short, engaging Instagram Reels caption.\n\n"
        f"{context}\n\n"
        f"Rules:\n"
        f"- 1-2 sentences max\n"
        f"- Include 2-3 relevant emojis\n"
        f"- End with 3-5 hashtags\n"
        f"- Make it feel authentic and hype, not corporate\n"
        f"- Do NOT wrap the caption in quotes\n"
        f"- Write ONLY the caption, nothing else"
    )

    logger.info("[reka] Generating caption via Reka Chat...")
    client = Reka(api_key=api_key)

    response = client.chat.create(
        messages=[
            ChatMessage(
                content=[{"type": "text", "text": prompt}],
                role="user",
            )
        ],
        model=os.environ.get("REKA_MODEL", "reka-core"),
    )

    caption = response.responses[0].message.content.strip()
    logger.info("[reka] Generated caption: %s", caption)
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
        video_path: Path to the video file (.mp4)
        emotion_report_path: Optional emotion_report.json for better captions
        custom_caption: If provided, skip Reka and use this caption instead
    """
    if custom_caption:
        caption = custom_caption
    else:
        try:
            caption = generate_caption(emotion_report_path)
        except Exception:
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
        print('  python main.py clips/raw_clip_1_subtitled.mp4')
        print('  python main.py clips/raw_clip_1_subtitled.mp4 clips/clip_1/emotion_report.json')
        print('  python main.py clips/raw_clip_1_subtitled.mp4 - "My custom caption 🎮"')
        sys.exit(1)

    video = sys.argv[1]
    emotion_json = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    caption_override = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else None

    publish_highlight(video, emotion_json, caption_override)
