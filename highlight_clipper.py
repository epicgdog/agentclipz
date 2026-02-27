"""
Highlight Clipping Agent
========================
Finds the peak moment (chat intensity + emotion confidence) in a 2-minute video,
aligns to the nearest sentence start, and exports a 30-second clip.
Reads inputs from the clip-worthy/ folder or from clip-transcript JSON + video.
Supports both legacy (transcript + chat_activity + streamer_emotions) and
clip-transcript format (utterances with emotion, from ClipperModulate).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

CLIP_WORTHY_DIR = Path(__file__).resolve().parent / "clip-worthy"
CLIP_DURATION_SEC = 30
SOURCE_DURATION_SEC = 120  # 2 minutes (legacy / full source)
SENTENCE_WINDOW_BEFORE_PEAK_SEC = 5

# Emotion weights for clip-transcript format (confidence proxy; align with ClipperModulate)
EMOTION_WEIGHTS = {
    "Excited": 1.0, "Angry": 0.95, "Surprised": 0.9, "Afraid": 0.85, "Frustrated": 0.85,
    "Contemptuous": 0.8, "Stressed": 0.8, "Anxious": 0.75, "Disgusted": 0.75, "Amused": 0.7,
    "Happy": 0.6, "Proud": 0.6, "Confident": 0.55, "Concerned": 0.5, "Hopeful": 0.5,
    "Affectionate": 0.45, "Interested": 0.45, "Confused": 0.45, "Disappointed": 0.4,
    "Sad": 0.4, "Ashamed": 0.4, "Relieved": 0.35,
    "Calm": 0.15, "Neutral": 0.1, "Bored": 0.05, "Tired": 0.05,
}


def _is_clip_transcript_format(data: dict) -> bool:
    """True if JSON is clip-transcript format (utterances + start_ms/end_ms at top level)."""
    return (
        isinstance(data.get("utterances"), list)
        and "start_ms" in data
        and "end_ms" in data
        and "chat_activity" not in data
    )


def _normalize_clip_transcript(data: dict) -> tuple[dict, float]:
    """
    Convert clip-transcript JSON to the internal format. Uses clip-relative time.
    Returns (normalized_data, source_duration_sec).
    """
    duration_ms = data["duration_ms"]
    duration_sec = duration_ms / 1000.0
    utterances = data.get("utterances") or []

    transcript = []
    streamer_emotions = []
    for u in utterances:
        rel_start_s = u["relative_start_ms"] / 1000.0
        rel_end_s = u["relative_end_ms"] / 1000.0
        # Only include segments that overlap the clip [0, duration_sec]
        if rel_end_s <= 0 or rel_start_s >= duration_sec:
            continue
        start_s = max(0.0, rel_start_s)
        end_s = min(duration_sec, rel_end_s)
        transcript.append({
            "text": u.get("text", ""),
            "start": start_s,
            "end": end_s,
            "is_sentence_start": True,  # treat each utterance start as sentence start
        })
        emotion = u.get("emotion")
        if emotion:
            confidence = EMOTION_WEIGHTS.get(emotion, 0.5)
            streamer_emotions.append({
                "timestamp": start_s,
                "label": emotion,
                "confidence": confidence,
            })

    normalized = {
        "transcript": transcript,
        "chat_activity": [],  # no chat in this format; peak uses emotion only
        "streamer_emotions": streamer_emotions,
    }
    return normalized, duration_sec


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


def _find_t_peak(data: dict, source_duration_sec: float | None = None) -> float:
    """Find the timestamp where combined score (chat intensity + emotion confidence) is highest."""
    cap = source_duration_sec if source_duration_sec is not None else SOURCE_DURATION_SEC
    chat_df = pd.DataFrame(data.get("chat_activity") or [])
    emotions_df = pd.DataFrame(data.get("streamer_emotions") or [])

    # Build a dense time grid (every 0.5s) to evaluate combined score
    t_max = 0.0
    if len(chat_df) and "timestamp" in chat_df.columns:
        t_max = max(t_max, float(chat_df["timestamp"].max()))
    if len(emotions_df) and "timestamp" in emotions_df.columns:
        t_max = max(t_max, float(emotions_df["timestamp"].max()))
    t_max = min(t_max, cap)
    if t_max <= 0:
        return 0.0
    grid = pd.DataFrame({"timestamp": np.arange(0, t_max + 0.5, 0.5)})

    # Forward-fill then interpolate chat intensity (use 0 if no chat data)
    if len(chat_df) and "timestamp" in chat_df.columns and "intensity" in chat_df.columns:
        chat_merged = grid.merge(chat_df, on="timestamp", how="left")
        chat_merged["intensity"] = chat_merged["intensity"].interpolate(method="linear").fillna(0)
        chat_merged = chat_merged[["timestamp", "intensity"]]
    else:
        chat_merged = grid.copy()
        chat_merged["intensity"] = 0.0

    # Same for emotion confidence (use 0 if no emotion data)
    if len(emotions_df) and "timestamp" in emotions_df.columns and "confidence" in emotions_df.columns:
        emotions_merged = grid.merge(emotions_df, on="timestamp", how="left")
        emotions_merged["confidence"] = emotions_merged["confidence"].interpolate(method="linear").fillna(0)
        emotions_merged = emotions_merged[["timestamp", "confidence"]]
    else:
        emotions_merged = grid.copy()
        emotions_merged["confidence"] = 0.0

    merged = chat_merged.merge(emotions_merged, on="timestamp")
    merged["combined_score"] = merged["intensity"] + merged["confidence"]

    if len(merged) == 0:
        return 0.0
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


def _compute_clip_bounds(
    aligned_start: float,
    source_duration_sec: float | None = None,
) -> tuple[float, float]:
    """Compute start/end for 30s clip, constrained to [0, source_duration_sec]."""
    cap = source_duration_sec if source_duration_sec is not None else SOURCE_DURATION_SEC
    clip_start = aligned_start
    clip_end = clip_start + CLIP_DURATION_SEC

    if clip_start < 0:
        clip_start = 0
        clip_end = min(CLIP_DURATION_SEC, cap)
    elif clip_end > cap:
        clip_end = cap
        clip_start = max(0, clip_end - CLIP_DURATION_SEC)

    return clip_start, clip_end


def _resolve_video_from_transcript_path(json_path: Path) -> Path:
    """Infer video path from clip-transcript path (e.g. clips/transcripts/clip_1_transcript.json -> clips/clip_1.mp4)."""
    stem = json_path.stem.replace("_transcript", "")
    # Prefer same parent as transcript dir (e.g. clips/transcripts -> clips)
    video_dir = json_path.parent.parent if json_path.parent.name == "transcripts" else json_path.parent
    for ext in (".mp4", ".webm", ".mov"):
        candidate = video_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return video_dir / f"{stem}.mp4"  # default extension for error message if not found


def get_best_highlight(
    clip_worthy_dir: Path | str | None = None,
    video_path: Path | str | None = None,
    json_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> str:
    """
    Find the best 30-second highlight and export to MP4.

    Input (one of):
      - clip_worthy_dir: directory containing a video and legacy JSON (transcript + chat_activity + streamer_emotions).
      - video_path + json_path: specific files. If json_path is clip-transcript format (utterances + emotion),
        it is normalized and the clip duration is used; video_path can be omitted and inferred from json_path.

    Returns:
        Path to the exported MP4 file.
    """
    source_duration_sec: float | None = None
    if json_path is not None:
        json_path = Path(json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if _is_clip_transcript_format(data):
            data, source_duration_sec = _normalize_clip_transcript(data)
            video_path = Path(video_path) if video_path is not None else _resolve_video_from_transcript_path(json_path)
        else:
            video_path = Path(video_path) if video_path is not None else None
            if video_path is None or not video_path.exists():
                raise FileNotFoundError("Legacy format requires video_path; for clip-transcript format video is inferred from json_path.")
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        out_dir = Path(output_path).parent if output_path is not None else (json_path.parent.parent if json_path.parent.name == "transcripts" else json_path.parent)
    else:
        dir_path = Path(clip_worthy_dir) if clip_worthy_dir else CLIP_WORTHY_DIR
        video_path, json_path, data = _load_data(dir_path)
        out_dir = dir_path

    t_peak = _find_t_peak(data, source_duration_sec)
    aligned_start = _find_aligned_start(data, t_peak)
    clip_start, clip_end = _compute_clip_bounds(aligned_start, source_duration_sec)

    if output_path is None:
        output_path = Path(out_dir) / "highlight.mp4"
    output_path = Path(output_path)

    with VideoFileClip(str(video_path)) as clip:
        subclip = clip.subclip(clip_start, clip_end)
        subclip.write_videofile(str(output_path), codec="libx264", audio_codec="aac", logger=None)

    return str(output_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        # python highlight_clipper.py <json_path> [output_path]
        # or: python highlight_clipper.py <video_path> <json_path> [output_path]
        # For clip-transcript JSON, video is inferred from json_path if not given.
        if len(sys.argv) >= 3 and Path(sys.argv[1]).suffix.lower() in (".mp4", ".webm", ".mov"):
            result = get_best_highlight(
                video_path=sys.argv[1],
                json_path=sys.argv[2],
                output_path=sys.argv[3] if len(sys.argv) > 3 else None,
            )
        else:
            result = get_best_highlight(
                json_path=sys.argv[1],
                output_path=sys.argv[2] if len(sys.argv) > 2 else None,
            )
    else:
        result = get_best_highlight()
    print(f"Exported highlight to: {result}")
