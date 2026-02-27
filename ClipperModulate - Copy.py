"""
Modulate Emotion Clipping Engine
=================================
Extracts audio from video, sends to Modulate API for emotion detection,
applies per-emotion and overall thresholding to trigger clips,
and generates emotion distribution graphs for optimal head/tail trimming.
For testing on general videos before hooking into live Twitch streams.
"""
import os
import json
import asyncio
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import requests
import aiohttp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# ============================================================
# CONFIG
# ============================================================
MODULATE_API_KEY = os.environ.get("MODULATE_API_KEY", "97fa1c48-12c7-4264-8ab3-fbe307a9f189")
BATCH_URL = "https://modulate-developer-apis.com/api/velma-2-stt-batch"
STREAM_URL = "wss://modulate-developer-apis.com/api/velma-2-stt-streaming"
# All 26 emotions Modulate can return
ALL_EMOTIONS = [
    "Neutral", "Calm", "Happy", "Amused", "Excited", "Proud",
    "Affectionate", "Interested", "Hopeful", "Frustrated", "Angry",
    "Contemptuous", "Concerned", "Afraid", "Sad", "Ashamed",
    "Bored", "Tired", "Surprised", "Anxious", "Stressed",
    "Disgusted", "Disappointed", "Confused", "Relieved", "Confident",
]
# Emotions that signal HIGH engagement / clip-worthy moments
HIGH_ENERGY_EMOTIONS = {
    "Excited", "Angry", "Frustrated", "Surprised", "Afraid",
    "Amused", "Happy", "Stressed", "Anxious", "Contemptuous",
    "Proud", "Disgusted",
}
# Emotions that signal LOW engagement (anti-clip)
LOW_ENERGY_EMOTIONS = {
    "Neutral", "Calm", "Bored", "Tired",
}
# ============================================================
# EMOTION SCORING - Convert labels to numeric scores
# ============================================================
# Since Modulate returns a single label (not numeric scores),
# we assign intensity weights to each emotion for thresholding.
EMOTION_WEIGHTS = {
    # High clip-worthy emotions (0.7 - 1.0)
    "Excited":      1.0,
    "Angry":        0.95,
    "Surprised":    0.9,
    "Afraid":       0.85,
    "Frustrated":   0.85,
    "Contemptuous": 0.8,
    "Stressed":     0.8,
    "Anxious":      0.75,
    "Disgusted":    0.75,
    "Amused":       0.7,
    # Medium emotions (0.4 - 0.6)
    "Happy":        0.6,
    "Proud":        0.6,
    "Confident":    0.55,
    "Concerned":    0.5,
    "Hopeful":      0.5,
    "Affectionate": 0.45,
    "Interested":   0.45,
    "Confused":     0.45,
    "Disappointed": 0.4,
    "Sad":          0.4,
    "Ashamed":      0.4,
    "Relieved":     0.35,
    # Low energy / anti-clip emotions (0.0 - 0.2)
    "Calm":         0.15,
    "Neutral":      0.1,
    "Bored":        0.05,
    "Tired":        0.05,
}
# ============================================================
# THRESHOLDS
# ============================================================
@dataclass
class ClipThresholds:
    """Configurable thresholds for triggering clips."""
    # --- Per-emotion thresholding ---
    # If any single high-energy emotion appears X times in a sliding window, trigger
    per_emotion_count_threshold: int = 3  # e.g., 3 "Excited" utterances in window
    # --- Overall emotion score thresholding ---
    # Sum all emotion weights in a sliding window; if above this, trigger
    overall_score_threshold: float = 5.0
    # --- Sliding window ---
    window_size_ms: int = 30_000  # 30-second sliding window
    # --- Clip parameters ---
    buffer_before_ms: int = 15_000   # buffer 15s before the trigger point
    buffer_after_ms: int = 10_000    # buffer 10s after the trigger point
    min_clip_duration_ms: int = 15_000  # minimum 15s clip
    max_clip_duration_ms: int = 90_000  # maximum 90s clip
    merge_gap_ms: int = 10_000  # merge clips that are within 10s of each other
# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class Utterance:
    """Single utterance from Modulate."""
    uuid: str
    text: str
    start_ms: int
    duration_ms: int
    speaker: int
    language: str
    emotion: Optional[str]
    accent: Optional[str]
    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms
    @property
    def emotion_score(self) -> float:
        return EMOTION_WEIGHTS.get(self.emotion, 0.0) if self.emotion else 0.0
@dataclass
class ClipCandidate:
    """A detected clip-worthy segment."""
    start_ms: int
    end_ms: int
    trigger_emotion: str
    trigger_type: str  # "per_emotion" or "overall_score"
    peak_score: float
    utterances: list = field(default_factory=list)
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def save_clip_transcript(
    clip: ClipCandidate,
    utterances: list,
    clip_index: int,
    output_dir: str,
):
    """
    Save the transcript for a clip as a JSON file.
    Creates a 'transcripts' subdirectory if it doesn't exist.
    """
    transcripts_dir = os.path.join(output_dir, "transcripts")
    os.makedirs(transcripts_dir, exist_ok=True)
    
    # Filter utterances within this clip
    clip_utts = [u for u in utterances if u.start_ms < clip.end_ms and u.end_ms > clip.start_ms]
    
    transcript_data = {
        "clip_index": clip_index,
        "start_ms": clip.start_ms,
        "end_ms": clip.end_ms,
        "duration_ms": clip.duration_ms,
        "trigger_emotion": clip.trigger_emotion,
        "trigger_type": clip.trigger_type,
        "peak_score": clip.peak_score,
        "utterances": [
            {
                "text": u.text,
                "start_ms": u.start_ms,
                "end_ms": u.end_ms,
                "duration_ms": u.duration_ms,
                "emotion": u.emotion,
                "speaker": u.speaker,
                "relative_start_ms": u.start_ms - clip.start_ms,
                "relative_end_ms": u.end_ms - clip.start_ms,
            }
            for u in clip_utts
        ],
        "full_transcript": " ".join(u.text for u in clip_utts if u.text),
    }
    
    json_path = os.path.join(transcripts_dir, f"clip_{clip_index}_transcript.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    print(f"Transcript saved: {json_path}")

# ============================================================
# STEP 1: EXTRACT AUDIO FROM VIDEO (for batch processing)
# ============================================================
def extract_audio_from_video(video_path: str, output_format: str = "mp3") -> str:
    """
    Extract audio from a video file using ffmpeg.
    Returns the path to the extracted audio file.
    Modulate supports MP4/MOV directly, but extracting audio first
    reduces file size and upload time.
    """
    output_path = tempfile.mktemp(suffix=f".{output_format}")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",  # no video
        "-acodec", "libmp3lame" if output_format == "mp3" else "pcm_s16le",
        "-ar", "16000",  # 16kHz sample rate (good for speech)
        "-ac", "1",  # mono
        "-y",  # overwrite
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    print(f"Extracted audio: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
    return output_path


def get_audio_duration_ms(audio_path: str) -> int:
    """Get duration of audio file in milliseconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return int(float(result.stdout.strip()) * 1000)


def split_audio_into_chunks(
    audio_path: str,
    chunk_duration_ms: int = 300_000,  # 5 minutes default
) -> list[tuple[str, int]]:
    """
    Split audio file into smaller chunks for API processing.
    Returns list of (chunk_path, offset_ms) tuples.
    """
    total_duration_ms = get_audio_duration_ms(audio_path)
    chunks = []
    
    for offset_ms in range(0, total_duration_ms, chunk_duration_ms):
        chunk_path = tempfile.mktemp(suffix=".mp3")
        start_s = offset_ms / 1000
        duration_s = min(chunk_duration_ms, total_duration_ms - offset_ms) / 1000
        
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-ss", str(start_s),
            "-t", str(duration_s),
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            chunk_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg chunk split failed: {result.stderr}")
        
        chunks.append((chunk_path, offset_ms))
        print(f"  Created chunk: {start_s:.0f}s - {start_s + duration_s:.0f}s")
    
    return chunks


# ============================================================
# STEP 2A: BATCH TRANSCRIPTION WITH EMOTION (pre-recorded videos)
# ============================================================
def transcribe_batch_with_emotion(audio_path: str, time_offset_ms: int = 0, max_retries: int = 3) -> list[Utterance]:
    """
    Send audio file to Modulate Batch API with emotion detection enabled.
    Returns list of Utterance objects with emotion labels.
    time_offset_ms: Added to all timestamps (for chunked processing)
    """
    print(f"Sending {audio_path} to Modulate Batch API...")
    
    last_error = None
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as f:
                response = requests.post(
                    BATCH_URL,
                    headers={"X-API-Key": MODULATE_API_KEY},
                    files={"upload_file": f},
                    data={
                        "speaker_diarization": "true",
                        "emotion_signal": "true",     # <-- ENABLE EMOTION
                        "accent_signal": "false",
                        "pii_phi_tagging": "false",
                    },
                    timeout=300,  # 5 minute timeout per chunk
                )
            response.raise_for_status()
            break  # Success
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise last_error
    
    result = response.json()
    utterances = []
    for u in result["utterances"]:
        utterances.append(Utterance(
            uuid=u["utterance_uuid"],
            text=u["text"],
            start_ms=u["start_ms"] + time_offset_ms,  # Apply offset
            duration_ms=u["duration_ms"],
            speaker=u["speaker"],
            language=u["language"],
            emotion=u["emotion"],
            accent=u["accent"],
        ))
    print(f"Got {len(utterances)} utterances, duration: {result['duration_ms']}ms")
    return utterances


def transcribe_chunked(audio_path: str, chunk_duration_ms: int = 300_000, max_chunks: int = None) -> list[Utterance]:
    """
    Split audio into chunks and transcribe each, combining results.
    Handles long videos that would timeout the API.
    max_chunks: Limit number of chunks to process (for testing)
    """
    total_duration_ms = get_audio_duration_ms(audio_path)
    
    # If short enough, process directly
    if total_duration_ms <= chunk_duration_ms:
        print("Audio is short enough, processing directly...")
        return transcribe_batch_with_emotion(audio_path)
    
    # Calculate how many chunks we'll actually process
    total_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms
    chunks_to_process = min(total_chunks, max_chunks) if max_chunks else total_chunks
    
    print(f"Audio is {total_duration_ms/1000/60:.1f} minutes ({total_chunks} chunks)")
    if max_chunks and max_chunks < total_chunks:
        print(f"  (Testing mode: processing only first {chunks_to_process} chunks)")
    
    all_utterances = []
    for i in range(chunks_to_process):
        offset_ms = i * chunk_duration_ms
        chunk_path = tempfile.mktemp(suffix=".mp3")
        start_s = offset_ms / 1000
        duration_s = min(chunk_duration_ms, total_duration_ms - offset_ms) / 1000
        
        # Create only this chunk
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-ss", str(start_s),
            "-t", str(duration_s),
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            chunk_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"\n  Processing chunk {i+1}/{chunks_to_process} ({start_s:.0f}s - {start_s + duration_s:.0f}s)...")
        try:
            chunk_utterances = transcribe_batch_with_emotion(chunk_path, time_offset_ms=offset_ms)
            all_utterances.extend(chunk_utterances)
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
    
    # Print combined emotion distribution
    emotion_counts = defaultdict(int)
    for u in all_utterances:
        if u.emotion:
            emotion_counts[u.emotion] += 1
    print("\nCombined emotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count} utterances")
    
    return all_utterances
# ============================================================
# STEP 2B: STREAMING TRANSCRIPTION WITH EMOTION (live/real-time)
# ============================================================
async def transcribe_streaming_with_emotion(
    audio_path: str,
    on_utterance=None,
    chunk_size: int = 8192,
) -> list[Utterance]:
    """
    Stream audio file to Modulate Streaming API with emotion detection.
    Calls on_utterance callback for each utterance (for real-time processing).
    Returns list of all Utterance objects.
    """
    url = (
        f"{STREAM_URL}"
        f"?api_key={MODULATE_API_KEY}"
        f"&speaker_diarization=true"
        f"&emotion_signal=true"    # <-- ENABLE EMOTION
        f"&accent_signal=false"
        f"&pii_phi_tagging=false"
    )
    utterances = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            async def send_audio():
                with open(audio_path, "rb") as f:
                    while chunk := f.read(chunk_size):
                        await ws.send_bytes(chunk)
                        await asyncio.sleep(chunk_size / 4000)  # pace near real-time
                await ws.send_str("")  # signal end of audio
            send_task = asyncio.create_task(send_audio())
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data["type"] == "utterance":
                            u_data = data["utterance"]
                            u = Utterance(
                                uuid=u_data["utterance_uuid"],
                                text=u_data["text"],
                                start_ms=u_data["start_ms"],
                                duration_ms=u_data["duration_ms"],
                                speaker=u_data["speaker"],
                                language=u_data["language"],
                                emotion=u_data["emotion"],
                                accent=u_data["accent"],
                            )
                            utterances.append(u)
                            if on_utterance:
                                on_utterance(u)
                        elif data["type"] == "done":
                            print(f"Stream done! Duration: {data['duration_ms']}ms")
                            break
                        elif data["type"] == "error":
                            print(f"Stream error: {data['error']}")
                            break
                    elif msg.type in (
                        aiohttp.WSMsgType.ERROR,
                        aiohttp.WSMsgType.CLOSE,
                    ):
                        break
            finally:
                if not send_task.done():
                    send_task.cancel()
    return utterances
# ============================================================
# STEP 3: EMOTION THRESHOLDING & CLIP DETECTION
# ============================================================
class EmotionClipDetector:
    """
    Detects clip-worthy moments based on emotion thresholding.
    Two trigger modes:
    1. Per-emotion: If any single high-energy emotion appears N times in a window
    2. Overall score: If the sum of all emotion weights in a window exceeds threshold
    """
    def __init__(self, thresholds: ClipThresholds = None):
        self.thresholds = thresholds or ClipThresholds()
    def detect_clips(self, utterances: list[Utterance]) -> list[ClipCandidate]:
        """
        Scan through utterances with a sliding window and detect clip triggers.
        Returns a list of ClipCandidate objects.
        """
        if not utterances:
            return []
        clips = []
        window_ms = self.thresholds.window_size_ms
        # Sort by start time
        sorted_utts = sorted(utterances, key=lambda u: u.start_ms)
        total_duration_ms = max(u.end_ms for u in sorted_utts)
        # Slide window across the timeline
        step_ms = 5000  # 5s step
        for window_start in range(0, total_duration_ms, step_ms):
            window_end = window_start + window_ms
            # Get utterances in this window
            window_utts = [
                u for u in sorted_utts
                if u.start_ms >= window_start and u.start_ms < window_end
            ]
            if not window_utts:
                continue
            # --- CHECK 1: Per-emotion threshold ---
            emotion_counts = defaultdict(int)
            for u in window_utts:
                if u.emotion and u.emotion in HIGH_ENERGY_EMOTIONS:
                    emotion_counts[u.emotion] += 1
            for emotion, count in emotion_counts.items():
                if count >= self.thresholds.per_emotion_count_threshold:
                    trigger_time = self._find_peak_time(window_utts, emotion)
                    clips.append(ClipCandidate(
                        start_ms=max(0, trigger_time - self.thresholds.buffer_before_ms),
                        end_ms=min(total_duration_ms, trigger_time + self.thresholds.buffer_after_ms),
                        trigger_emotion=emotion,
                        trigger_type="per_emotion",
                        peak_score=EMOTION_WEIGHTS[emotion] * count,
                        utterances=window_utts,
                    ))
            # --- CHECK 2: Overall score threshold ---
            total_score = sum(u.emotion_score for u in window_utts)
            if total_score >= self.thresholds.overall_score_threshold:
                peak_time = self._find_score_peak(window_utts)
                dominant = max(
                    emotion_counts.items(),
                    key=lambda x: x[1],
                    default=("Mixed", 0),
                )
                clips.append(ClipCandidate(
                    start_ms=max(0, peak_time - self.thresholds.buffer_before_ms),
                    end_ms=min(total_duration_ms, peak_time + self.thresholds.buffer_after_ms),
                    trigger_emotion=dominant[0],
                    trigger_type="overall_score",
                    peak_score=total_score,
                    utterances=window_utts,
                ))
        # Merge overlapping clips
        merged = self._merge_clips(clips, total_duration_ms)
        return merged
    def _find_peak_time(self, utterances: list[Utterance], target_emotion: str) -> int:
        """Find the timestamp where the target emotion is most concentrated."""
        matching = [u for u in utterances if u.emotion == target_emotion]
        if not matching:
            return utterances[0].start_ms
        # Return the midpoint of matching utterances
        avg_start = sum(u.start_ms for u in matching) // len(matching)
        return avg_start
    def _find_score_peak(self, utterances: list[Utterance]) -> int:
        """Find the timestamp where overall emotion score peaks."""
        if not utterances:
            return 0
        best = max(utterances, key=lambda u: u.emotion_score)
        return best.start_ms
    def _merge_clips(
        self, clips: list[ClipCandidate], total_duration_ms: int
    ) -> list[ClipCandidate]:
        """Merge overlapping or nearby clip candidates."""
        if not clips:
            return []
        # Sort by start time
        sorted_clips = sorted(clips, key=lambda c: c.start_ms)
        merged = [sorted_clips[0]]
        for clip in sorted_clips[1:]:
            last = merged[-1]
            if clip.start_ms <= last.end_ms + self.thresholds.merge_gap_ms:
                # Merge: extend the end, keep the higher score
                last.end_ms = max(last.end_ms, clip.end_ms)
                last.peak_score = max(last.peak_score, clip.peak_score)
                if clip.peak_score > last.peak_score:
                    last.trigger_emotion = clip.trigger_emotion
                    last.trigger_type = clip.trigger_type
                # Deduplicate utterances by uuid
                seen_uuids = {u.uuid for u in last.utterances}
                for u in clip.utterances:
                    if u.uuid not in seen_uuids:
                        last.utterances.append(u)
                        seen_uuids.add(u.uuid)
            else:
                merged.append(clip)
        # Enforce min/max clip duration
        final = []
        for clip in merged:
            duration = clip.duration_ms
            if duration < self.thresholds.min_clip_duration_ms:
                # Extend symmetrically
                extend = (self.thresholds.min_clip_duration_ms - duration) // 2
                clip.start_ms = max(0, clip.start_ms - extend)
                clip.end_ms = min(total_duration_ms, clip.end_ms + extend)
            elif duration > self.thresholds.max_clip_duration_ms:
                # Trim from edges, keeping the peak
                clip.end_ms = clip.start_ms + self.thresholds.max_clip_duration_ms
            final.append(clip)
        return final
# ============================================================
# STEP 4: EMOTION DISTRIBUTION GRAPH FOR HEAD/TAIL TRIMMING
# ============================================================
def plot_emotion_distribution(
    utterances: list[Utterance],
    clip: ClipCandidate,
    output_path: str = "emotion_distribution.png",
):
    """
    Generate a real-time emotion distribution graph for a clip.
    Shows where emotion is most concentrated and suggests head/tail trim points.
    """
    # Filter utterances within the clip
    clip_utts = [
        u for u in utterances
        if u.start_ms >= clip.start_ms and u.end_ms <= clip.end_ms
    ]
    if not clip_utts:
        print("No utterances in clip range.")
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1.5, 1]})
    fig.suptitle(
        f"Emotion Distribution | Clip {clip.start_ms/1000:.1f}s - {clip.end_ms/1000:.1f}s | "
        f"Trigger: {clip.trigger_emotion} ({clip.trigger_type})",
        fontsize=13,
        fontweight="bold",
    )
    # === Plot 1: Emotion score timeline ===
    ax1 = axes[0]
    times = [(u.start_ms - clip.start_ms) / 1000 for u in clip_utts]  # relative seconds
    scores = [u.emotion_score for u in clip_utts]
    colors = [
        "red" if u.emotion in HIGH_ENERGY_EMOTIONS
        else "gray" if u.emotion in LOW_ENERGY_EMOTIONS
        else "orange"
        for u in clip_utts
    ]
    ax1.bar(times, scores, width=0.8, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    # Annotate each bar with the emotion label
    for t, s, u in zip(times, scores, clip_utts):
        if s > 0.3:
            ax1.text(t, s + 0.02, u.emotion, ha="center", va="bottom", fontsize=7, rotation=45)
    ax1.set_ylabel("Emotion Score")
    ax1.set_title("Per-Utterance Emotion Intensity")
    ax1.set_ylim(0, 1.15)
    ax1.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="High threshold")
    ax1.legend(loc="upper right")
    # === Plot 2: Rolling average emotion density ===
    ax2 = axes[1]
    # Create a continuous timeline at 1-second resolution
    clip_duration_s = (clip.end_ms - clip.start_ms) / 1000
    timeline = np.zeros(int(clip_duration_s) + 1)
    for u in clip_utts:
        rel_start_s = (u.start_ms - clip.start_ms) / 1000
        rel_end_s = (u.end_ms - clip.start_ms) / 1000
        start_idx = max(0, int(rel_start_s))
        end_idx = min(len(timeline) - 1, int(rel_end_s))
        for i in range(start_idx, end_idx + 1):
            timeline[i] += u.emotion_score
    # Rolling average (5-second window)
    kernel_size = min(5, len(timeline))
    if kernel_size > 0:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(timeline, kernel, mode="same")
    else:
        smoothed = timeline
    x_seconds = np.arange(len(smoothed))
    ax2.fill_between(x_seconds, smoothed, alpha=0.4, color="purple")
    ax2.plot(x_seconds, smoothed, color="purple", linewidth=2)
    ax2.set_ylabel("Emotion Density")
    ax2.set_title("Rolling Emotion Density (5s window)")
    # Find peak concentration zone
    if len(smoothed) > 0:
        peak_idx = np.argmax(smoothed)
        ax2.axvline(x=peak_idx, color="red", linestyle="--", alpha=0.7, label=f"Peak @ {peak_idx}s")
        ax2.legend()
    # === Plot 3: Suggested trim visualization ===
    ax3 = axes[2]
    # Find suggested trim points based on emotion density
    trim_start, trim_end = suggest_trim_points(smoothed, clip_duration_s)
    ax3.barh(0, clip_duration_s, height=0.4, color="lightgray", edgecolor="black")
    ax3.barh(0, trim_end - trim_start, left=trim_start, height=0.4, color="green", alpha=0.6, edgecolor="black")
    ax3.set_xlim(0, clip_duration_s)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title(
        f"Suggested Trim: {trim_start:.1f}s - {trim_end:.1f}s "
        f"(cut {trim_start:.1f}s head, {clip_duration_s - trim_end:.1f}s tail)"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Emotion distribution graph saved: {output_path}")
    return trim_start, trim_end
def suggest_trim_points(
    emotion_density: np.ndarray,
    total_duration_s: float,
    threshold_ratio: float = 0.2,
) -> tuple[float, float]:
    """
    Suggest head/tail trim points based on where emotion density
    drops below a threshold of the peak.
    """
    if len(emotion_density) == 0:
        return 0.0, total_duration_s
    peak = np.max(emotion_density)
    threshold = peak * threshold_ratio
    # Find first second above threshold (trim head)
    trim_start = 0.0
    for i, val in enumerate(emotion_density):
        if val >= threshold:
            trim_start = max(0, i - 2)  # 2s padding
            break
    # Find last second above threshold (trim tail)
    trim_end = total_duration_s
    for i in range(len(emotion_density) - 1, -1, -1):
        if emotion_density[i] >= threshold:
            trim_end = min(total_duration_s, i + 2)  # 2s padding
            break
    return float(trim_start), float(trim_end)
# ============================================================
# STEP 4B: ACCUMULATE EMOTIONS ACROSS THE FULL STREAM
# ============================================================
def accumulate_emotions(utterances: list[Utterance]) -> dict[str, int]:
    """
    Count how many utterances had each emotion across the entire stream.
    Returns a dict of {emotion_label: count} for all 26 emotions,
    sorted by count descending.
    """
    totals: dict[str, int] = {emotion: 0 for emotion in ALL_EMOTIONS}
    for u in utterances:
        if u.emotion and u.emotion in totals:
            totals[u.emotion] += 1
    return dict(sorted(totals.items(), key=lambda x: -x[1]))
# ============================================================
# STEP 5: CUT THE CLIP FROM VIDEO
# ============================================================
def generate_srt_subtitles(utterances: list, start_ms: int, end_ms: int, output_path: str) -> str:
    """
    Generate an SRT subtitle file from utterances for a clip.
    Returns the path to the SRT file.
    """
    # Filter utterances within this clip
    clip_utts = [u for u in utterances if u.start_ms >= start_ms and u.end_ms <= end_ms]
    
    if not clip_utts:
        return None
    
    def ms_to_srt_time(ms: int) -> str:
        """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        millis = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
    
    srt_content = []
    for i, u in enumerate(clip_utts, 1):
        # Calculate relative time within the clip
        rel_start_ms = u.start_ms - start_ms
        rel_end_ms = u.end_ms - start_ms
        
        start_time = ms_to_srt_time(rel_start_ms)
        end_time = ms_to_srt_time(rel_end_ms)
        
        # Clean the text (remove special chars that might break ffmpeg)
        text = u.text.replace("'", "'").replace('"', "'").replace("\\", "")
        
        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")  # Empty line between entries
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))
    
    return output_path


def cut_clip_from_video(
    video_path: str,
    start_ms: int,
    end_ms: int,
    output_path: str,
    utterances: list = None,
    show_emotions: bool = False,
    show_subtitles: bool = False,
):
    """
    Use ffmpeg to cut a clip from the source video.
    If show_emotions=True, overlay emotion timestamps on the video.
    If show_subtitles=True, burn in subtitles from transcription.
    """
    start_s = start_ms / 1000
    duration_s = (end_ms - start_ms) / 1000
    
    if utterances and (show_emotions or show_subtitles):
        # Filter utterances within this clip (include overlapping utterances)
        clip_utts = [u for u in utterances if u.start_ms < end_ms and u.end_ms > start_ms]
        
        if clip_utts:
            filter_parts = []
            srt_path = None
            
            # Generate subtitles if requested
            if show_subtitles:
                srt_path = tempfile.mktemp(suffix=".srt")
                generate_srt_subtitles(utterances, start_ms, end_ms, srt_path)
                # Escape Windows path for ffmpeg filter
                srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
                # Subtitle filter with styling
                filter_parts.append(
                    f"subtitles='{srt_escaped}':force_style='FontSize=20,FontName=Arial,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Shadow=1,MarginV=60'"
                )
            
            # Add emotion overlay if requested
            if show_emotions:
                for u in clip_utts:
                    if u.emotion:
                        rel_start = (u.start_ms - start_ms) / 1000
                        rel_end = (u.end_ms - start_ms) / 1000
                        
                        if u.emotion in HIGH_ENERGY_EMOTIONS:
                            color = "red"
                        elif u.emotion in LOW_ENERGY_EMOTIONS:
                            color = "gray"
                        else:
                            color = "yellow"
                        
                        # Emotion label at top of screen (subtitles are at bottom)
                        filter_parts.append(
                            f"drawtext=text='{u.emotion}':fontsize=24:fontcolor={color}:"
                            f"x=(w-text_w)/2:y=30:enable='between(t,{rel_start:.2f},{rel_end:.2f})'"
                        )
            
            if filter_parts:
                filter_str = ",".join(filter_parts)
                # Use -ss BEFORE -i for input seeking - this makes video start at time 0
                # matching our SRT timestamps
                cmd = [
                    "ffmpeg",
                    "-ss", str(start_s),  # Input seeking - seek first, then read
                    "-i", video_path,
                    "-t", str(duration_s),
                    "-vf", filter_str,
                    "-c:a", "aac",
                    "-y",
                    output_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Cleanup temp SRT file
                if srt_path and os.path.exists(srt_path):
                    os.remove(srt_path)
                
                if result.returncode != 0:
                    print(f"Warning: Overlay/subtitles failed, falling back to simple cut: {result.stderr[:300]}")
                else:
                    features = []
                    if show_subtitles:
                        features.append("subtitles")
                    if show_emotions:
                        features.append("emotions")
                    print(f"Clip saved with {'+'.join(features)}: {output_path} ({duration_s:.1f}s)")
                    return
    
    # Simple cut without overlay (faster)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", str(start_s),
        "-t", str(duration_s),
        "-c", "copy",  # fast copy without re-encoding
        "-y",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip cut failed: {result.stderr}")
    print(f"Clip saved: {output_path} ({duration_s:.1f}s)")
# ============================================================
# MAIN PIPELINE - TEST ON A GENERAL VIDEO
# ============================================================
def process_video(
    video_path: str,
    output_dir: str = "./clips",
    thresholds: ClipThresholds = None,
    use_streaming: bool = False,
    max_chunks: int = None,
):
    """
    Full pipeline: Video -> Audio -> Modulate Emotion -> Clip Detection -> Trim -> Export
    Args:
        video_path: Path to the video file to process
        output_dir: Directory to save clips and graphs
        thresholds: Custom thresholds (or use defaults)
        use_streaming: Use streaming API instead of batch
        max_chunks: Limit chunks to process (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(f"Processing: {video_path}")
    print("=" * 60)
    # Step 1: Extract audio
    print("\\n[1/5] Extracting audio...")
    audio_path = extract_audio_from_video(video_path)
    # Step 2: Transcribe with emotion
    print("\\n[2/5] Transcribing with Modulate (emotion_signal=true)...")
    if use_streaming:
        utterances = asyncio.run(transcribe_streaming_with_emotion(audio_path))
    else:
        utterances = transcribe_chunked(audio_path, max_chunks=max_chunks)
    # Step 3: Accumulate emotions + detect clip-worthy moments
    print("\\n[3/5] Accumulating emotions and detecting clip-worthy moments...")
    emotion_totals = accumulate_emotions(utterances)
    print("\\nStream emotion totals:")
    for emotion, count in emotion_totals.items():
        if count > 0:
            print(f"  {emotion}: {count}")
    detector = EmotionClipDetector(thresholds or ClipThresholds())
    clips = detector.detect_clips(utterances)
    print(f"\\nFound {len(clips)} clip candidates:")
    for i, clip in enumerate(clips):
        print(
            f"  Clip {i+1}: {clip.start_ms/1000:.1f}s - {clip.end_ms/1000:.1f}s "
            f"({clip.duration_ms/1000:.1f}s) | Trigger: {clip.trigger_emotion} "
            f"({clip.trigger_type}) | Score: {clip.peak_score:.2f}"
        )
    # Step 4: Generate emotion graphs and trim suggestions
    print("\\n[4/5] Generating emotion distribution graphs...")
    for i, clip in enumerate(clips):
        graph_path = os.path.join(output_dir, f"clip_{i+1}_emotion.png")
        trim_result = plot_emotion_distribution(utterances, clip, graph_path)
        if trim_result:
            trim_start_s, trim_end_s = trim_result
            # Adjust clip boundaries based on trim suggestion
            clip.start_ms += int(trim_start_s * 1000)
            clip.end_ms = clip.start_ms + int((trim_end_s - trim_start_s) * 1000)
    # Step 5: Cut clips from original video and save transcripts
    print("\\n[5/5] Cutting clips from video...")
    for i, clip in enumerate(clips):
        clip_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
        cut_clip_from_video(video_path, clip.start_ms, clip.end_ms, clip_path, utterances, show_emotions=False, show_subtitles=True)
        save_clip_transcript(clip, utterances, i + 1, output_dir)
    # Cleanup temp audio
    if os.path.exists(audio_path):
        os.remove(audio_path)
    print("\\n" + "=" * 60)
    print(f"Done! {len(clips)} clips saved to {output_dir}")
    print("=" * 60)
    return clips, emotion_totals
# ============================================================
# REAL-TIME STREAMING CLIP DETECTOR (for live Twitch streams)
# ============================================================
class LiveEmotionClipDetector:
    """
    Real-time clip detector for live streams.
    Maintains a sliding window of utterances and triggers clips
    as emotions cross thresholds.
    Usage:
        detector = LiveEmotionClipDetector()
        # For each utterance from Modulate streaming:
        clip = detector.feed_utterance(utterance)
        if clip:
            # Trigger clip recording!
    """
    def __init__(self, thresholds: ClipThresholds = None):
        self.thresholds = thresholds or ClipThresholds()
        self.utterance_buffer: list[Utterance] = []
        self.triggered_clips: list[ClipCandidate] = []
        self._last_trigger_ms: int = -999999  # cooldown tracking
        # Cumulative emotion counts across the entire stream
        self.emotion_totals: dict[str, int] = {e: 0 for e in ALL_EMOTIONS}
    def get_emotion_totals(self) -> dict[str, int]:
        """Return accumulated emotion counts sorted by frequency."""
        return dict(sorted(self.emotion_totals.items(), key=lambda x: -x[1]))
    def feed_utterance(self, utterance: Utterance) -> Optional[ClipCandidate]:
        """
        Feed a new utterance from Modulate streaming API.
        Returns a ClipCandidate if a clip should be triggered, else None.
        """
        # Accumulate into stream-wide totals
        if utterance.emotion and utterance.emotion in self.emotion_totals:
            self.emotion_totals[utterance.emotion] += 1
        self.utterance_buffer.append(utterance)
        # Prune old utterances outside the window
        window_start = utterance.start_ms - self.thresholds.window_size_ms
        self.utterance_buffer = [
            u for u in self.utterance_buffer if u.start_ms >= window_start
        ]
        # Cooldown: don't trigger too frequently
        if utterance.start_ms - self._last_trigger_ms < self.thresholds.merge_gap_ms:
            return None
        # Check per-emotion threshold
        emotion_counts = defaultdict(int)
        for u in self.utterance_buffer:
            if u.emotion and u.emotion in HIGH_ENERGY_EMOTIONS:
                emotion_counts[u.emotion] += 1
        for emotion, count in emotion_counts.items():
            if count >= self.thresholds.per_emotion_count_threshold:
                clip = ClipCandidate(
                    start_ms=max(0, utterance.start_ms - self.thresholds.buffer_before_ms),
                    end_ms=utterance.start_ms + self.thresholds.buffer_after_ms,
                    trigger_emotion=emotion,
                    trigger_type="per_emotion",
                    peak_score=EMOTION_WEIGHTS[emotion] * count,
                    utterances=list(self.utterance_buffer),
                )
                self._last_trigger_ms = utterance.start_ms
                self.triggered_clips.append(clip)
                return clip
        # Check overall score threshold
        total_score = sum(u.emotion_score for u in self.utterance_buffer)
        if total_score >= self.thresholds.overall_score_threshold:
            dominant = max(
                emotion_counts.items(),
                key=lambda x: x[1],
                default=("Mixed", 0),
            )
            clip = ClipCandidate(
                start_ms=max(0, utterance.start_ms - self.thresholds.buffer_before_ms),
                end_ms=utterance.start_ms + self.thresholds.buffer_after_ms,
                trigger_emotion=dominant[0],
                trigger_type="overall_score",
                peak_score=total_score,
                utterances=list(self.utterance_buffer),
            )
            self._last_trigger_ms = utterance.start_ms
            self.triggered_clips.append(clip)
            return clip
        return None


# ============================================================
# LIVE STREAM PROCESSING (Twitch/YouTube)
# ============================================================
def download_stream_segment(stream_url: str, duration_s: int = 300, output_path: str = None) -> str:
    """
    Download a segment of a live stream using streamlink + ffmpeg.
    Requires: pip install streamlink
    
    Args:
        stream_url: Twitch/YouTube live stream URL
        duration_s: How many seconds to capture (default 5 minutes)
        output_path: Output file path (or auto-generate temp file)
    
    Returns:
        Path to the downloaded video segment
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp4")
    
    print(f"Capturing {duration_s}s from live stream: {stream_url}")
    
    # Use streamlink to get the stream, pipe to ffmpeg for duration limit
    cmd = [
        "streamlink",
        stream_url,
        "best",  # best quality
        "-O",    # output to stdout
    ]
    
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "pipe:0",  # read from stdin
        "-t", str(duration_s),
        "-c", "copy",
        "-y",
        output_path,
    ]
    
    # Pipe streamlink output to ffmpeg
    try:
        streamlink_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=streamlink_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        streamlink_proc.stdout.close()  # Allow streamlink to receive SIGPIPE
        
        stdout, stderr = ffmpeg_proc.communicate(timeout=duration_s + 60)
        streamlink_proc.terminate()
        
        if ffmpeg_proc.returncode != 0:
            raise RuntimeError(f"Stream capture failed: {stderr.decode()[:500]}")
        
        print(f"Captured stream segment: {output_path}")
        return output_path
        
    except FileNotFoundError:
        raise RuntimeError("streamlink not found. Install with: pip install streamlink")


async def process_live_stream(
    stream_url: str,
    output_dir: str = "./live_clips",
    thresholds: ClipThresholds = None,
    segment_duration_s: int = 300,
    num_segments: int = None,
):
    """
    Process a live stream in real-time segments.
    
    Args:
        stream_url: Twitch/YouTube live URL (e.g., "https://twitch.tv/jasontheween")
        output_dir: Directory to save clips
        thresholds: Clip detection thresholds
        segment_duration_s: Duration of each segment to process (default 5 min)
        num_segments: Number of segments to process (None = infinite until stopped)
    """
    os.makedirs(output_dir, exist_ok=True)
    thresholds = thresholds or ClipThresholds()
    
    print("=" * 60)
    print(f"LIVE STREAM MODE: {stream_url}")
    print(f"Segment duration: {segment_duration_s}s | Output: {output_dir}")
    print("=" * 60)
    
    all_clips = []
    all_emotion_totals = {e: 0 for e in ALL_EMOTIONS}
    segment_num = 0
    
    try:
        while num_segments is None or segment_num < num_segments:
            segment_num += 1
            print(f"\\n{'='*40}")
            print(f"SEGMENT {segment_num}")
            print(f"{'='*40}")
            
            # Capture stream segment
            segment_path = os.path.join(output_dir, f"segment_{segment_num}.mp4")
            try:
                download_stream_segment(stream_url, segment_duration_s, segment_path)
            except Exception as e:
                print(f"Stream capture error: {e}")
                print("Stream may have ended. Stopping.")
                break
            
            # Process the segment
            try:
                clips, emotion_totals = process_video(
                    segment_path,
                    output_dir=os.path.join(output_dir, f"segment_{segment_num}_clips"),
                    thresholds=thresholds,
                )
                all_clips.extend(clips)
                
                # Accumulate emotion totals
                for emotion, count in emotion_totals.items():
                    all_emotion_totals[emotion] = all_emotion_totals.get(emotion, 0) + count
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
            finally:
                # Optionally cleanup segment file
                # os.remove(segment_path)
                pass
            
            # Print running totals
            print(f"\\nRunning totals after segment {segment_num}:")
            print(f"  Total clips: {len(all_clips)}")
            sorted_totals = sorted(all_emotion_totals.items(), key=lambda x: -x[1])
            for emotion, count in sorted_totals[:5]:
                if count > 0:
                    print(f"  {emotion}: {count}")
    
    except KeyboardInterrupt:
        print("\\n\\nStopped by user.")
    
    # Final summary
    print("\\n" + "=" * 60)
    print("FINAL STREAM SUMMARY")
    print("=" * 60)
    print(f"Segments processed: {segment_num}")
    print(f"Total clips generated: {len(all_clips)}")
    print("\\nFinal emotion totals:")
    for emotion, count in sorted(all_emotion_totals.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {emotion}: {count}")
    
    return all_clips, all_emotion_totals


# ============================================================
# ENTRY POINT - TEST ON A VIDEO
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ClipperModulate.py <video_path> [output_dir] [--test]")
        print("  python ClipperModulate.py --live <stream_url> [output_dir] [--segments N]")
        print("\\nExamples:")
        print("  python ClipperModulate.py stream_recording.mp4 ./clips")
        print("  python ClipperModulate.py stream_recording.mp4 ./clips --test  # Only first 5 chunks")
        print("  python ClipperModulate.py --live https://twitch.tv/jasontheween ./live_clips")
        print("  python ClipperModulate.py --live https://twitch.tv/jasontheween ./live_clips --segments 3")
        print("\\nSet MODULATE_API_KEY environment variable before running.")
        sys.exit(1)
    
    # Custom thresholds (tune these for JasonTheWeen's stream style)
    custom_thresholds = ClipThresholds(
        per_emotion_count_threshold=3,
        overall_score_threshold=5.0,
        window_size_ms=30_000,
        buffer_before_ms=15_000,
        buffer_after_ms=10_000,
        min_clip_duration_ms=15_000,
        max_clip_duration_ms=90_000,
    )
    
    # Check for live stream mode
    if sys.argv[1] == "--live":
        if len(sys.argv) < 3:
            print("Error: --live requires a stream URL")
            sys.exit(1)
        
        stream_url = sys.argv[2]
        out_dir = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "./live_clips"
        
        # Parse --segments N
        num_segments = None
        if "--segments" in sys.argv:
            idx = sys.argv.index("--segments")
            if idx + 1 < len(sys.argv):
                num_segments = int(sys.argv[idx + 1])
        
        print(f"LIVE STREAM MODE")
        print(f"URL: {stream_url}")
        print(f"Output: {out_dir}")
        print(f"Segments: {num_segments if num_segments else 'unlimited (Ctrl+C to stop)'}")
        
        asyncio.run(process_live_stream(
            stream_url,
            out_dir,
            custom_thresholds,
            num_segments=num_segments,
        ))
    else:
        # Regular video file mode
        video = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "./clips"
        test_mode = "--test" in sys.argv
        
        # Check for --chunks N argument
        max_chunks = None
        for i, arg in enumerate(sys.argv):
            if arg == "--chunks" and i + 1 < len(sys.argv):
                max_chunks = int(sys.argv[i + 1])
                break
        
        if max_chunks is None and test_mode:
            max_chunks = 5
        
        if max_chunks:
            print(f"*** TEST MODE: Processing only first {max_chunks} chunk(s) ({max_chunks * 5} minutes) ***\\n")
        
        clips, emotion_totals = process_video(video, out_dir, custom_thresholds, max_chunks=max_chunks)
        
        # Print clip summary
        for i, clip in enumerate(clips):
            print(f"\\nClip {i+1} Details:")
            print(f"  Time: {clip.start_ms/1000:.1f}s - {clip.end_ms/1000:.1f}s ({clip.duration_ms/1000:.1f}s)")
            print(f"  Trigger: {clip.trigger_emotion} ({clip.trigger_type})")
            print(f"  Score: {clip.peak_score:.2f}")
            emotions_in_clip = defaultdict(int)
            for u in clip.utterances:
                if u.emotion:
                    emotions_in_clip[u.emotion] += 1
            print(f"  Emotions: {dict(emotions_in_clip)}")
        
        # Print accumulated emotion totals for the entire stream
        print("\\n" + "=" * 60)
        print("ACCUMULATED EMOTION TOTALS (entire stream)")
        print("=" * 60)
        for emotion, count in emotion_totals.items():
            if count > 0:
                print(f"  {emotion}: {count}")