"""Quick test: generate a caption using Reka Chat API from emotion_report.json."""

import os
from dotenv import load_dotenv

load_dotenv()

# Test just the caption generation (no Instagram login needed)
from publisher.main import generate_caption

EMOTION_REPORT = "clips/clip_1/emotion_report.json"

if __name__ == "__main__":
    api_key = os.environ.get("REKA_API_KEY")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Emotion report: {EMOTION_REPORT}")
    print(f"Exists: {os.path.exists(EMOTION_REPORT)}")
    print("=" * 50)

    caption = generate_caption(EMOTION_REPORT)
    print(f"\nGenerated caption:\n{caption}")
