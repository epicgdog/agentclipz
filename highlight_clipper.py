"""
Highlight Clipping Agent
========================
Finds the peak moment (chat intensity + emotion confidence) in a 2-minute video,
aligns to the nearest sentence start, and exports a 30-second clip.
Reads inputs from the clip-worthy/ folder.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

CLIP_WORTHY_DIR = Path(__file__).resolve().parent / "clip-worthy"
CLIP_DURATION_SEC = 30
SOURCE_DURATION_SEC = 120  # 2 minutes
SENTENCE_WINDOW_BEFORE_PEAK_SEC = 5


def _load_data(clip_worthy_dir: Path) -> tuple[Path, Path, dict]:
    """Find video and JSON in clip-worthy, load JSON."""
    video_extensions = (".mp4", ".webm", ".mov")
    videos = [f for f in clip_worthy_dir.iterdir() if f.suffix.lower() in video_extensions]
    jsons = [f for f in clip_worthy_dir.iterdir() if f.suffix.lower() == ".json"]

    if not videos:
        raise FileNotFoundError(f"No video file (.mp4/.webm/.mov) found in {clip_worthy_dir}")
    if not jsons:
        raise FileNotFoundError(f"No JSON file found in {clip_worthy_dir}")

    video_path = videos[0]
    json_path = jsons[0]

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    return video_path, json_path, data


def _find_t_peak(data: dict) -> float:
    """Find the timestamp where combined score (chat intensity + emotion confidence) is highest."""
    chat_df = pd.DataFrame(data["chat_activity"])
    emotions_df = pd.DataFrame(data["streamer_emotions"])

    # Build a dense time grid (every 0.5s) to evaluate combined score
    t_max = max(
        chat_df["timestamp"].max() if len(chat_df) else 0,
        emotions_df["timestamp"].max() if len(emotions_df) else 0,
    )
    t_max = min(t_max, SOURCE_DURATION_SEC)
    grid = pd.DataFrame({"timestamp": np.arange(0, t_max + 0.5, 0.5)})

    # Forward-fill then interpolate chat intensity
    chat_merged = grid.merge(chat_df, on="timestamp", how="left")
    chat_merged["intensity"] = chat_merged["intensity"].interpolate(method="linear").fillna(0)
    chat_merged = chat_merged[["timestamp", "intensity"]]

    # Same for emotion confidence
    emotions_merged = grid.merge(emotions_df, on="timestamp", how="left")
    emotions_merged["confidence"] = emotions_merged["confidence"].interpolate(method="linear").fillna(0)
    emotions_merged = emotions_merged[["timestamp", "confidence"]]

    merged = chat_merged.merge(emotions_merged, on="timestamp")
    merged["combined_score"] = merged["intensity"] + merged["confidence"]

    best_row = merged.loc[merged["combined_score"].idxmax()]
    return float(best_row["timestamp"])


def _find_aligned_start(data: dict, t_peak: float) -> float:
    """Find nearest sentence start within 5 seconds before T_peak."""
    transcript = data.get("transcript") or []
    window_start = max(0, t_peak - SENTENCE_WINDOW_BEFORE_PEAK_SEC)
    window_end = t_peak

    sentence_starts = [
        seg["start"]
        for seg in transcript
        if seg.get("is_sentence_start") and window_start <= seg["start"] <= window_end
    ]

    if not sentence_starts:
        return t_peak  # fallback to peak itself
    return max(sentence_starts)  # nearest = most recent before peak


def _compute_clip_bounds(aligned_start: float) -> tuple[float, float]:
    """Compute start/end for 30s clip, constrained to [0, SOURCE_DURATION_SEC]."""
    clip_start = aligned_start
    clip_end = clip_start + CLIP_DURATION_SEC

    if clip_start < 0:
        clip_start = 0
        clip_end = CLIP_DURATION_SEC
    elif clip_end > SOURCE_DURATION_SEC:
        clip_end = SOURCE_DURATION_SEC
        clip_start = clip_end - CLIP_DURATION_SEC

    return clip_start, clip_end


def get_best_highlight(
    clip_worthy_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> str:
    """
    Find the best 30-second highlight from the video in clip-worthy and export to MP4.

    Returns:
        Path to the exported MP4 file.
    """
    dir_path = Path(clip_worthy_dir) if clip_worthy_dir else CLIP_WORTHY_DIR
    video_path, json_path, data = _load_data(dir_path)

    t_peak = _find_t_peak(data)
    aligned_start = _find_aligned_start(data, t_peak)
    clip_start, clip_end = _compute_clip_bounds(aligned_start)

    if output_path is None:
        output_path = dir_path / "highlight.mp4"
    output_path = Path(output_path)

    with VideoFileClip(str(video_path)) as clip:
        subclip = clip.subclip(clip_start, clip_end)
        subclip.write_videofile(str(output_path), codec="libx264", audio_codec="aac", logger=None)

    return str(output_path)


if __name__ == "__main__":
    result = get_best_highlight()
    print(f"Exported highlight to: {result}")
