"""
Modulate Emotion Clipping Engine
=================================
Extracts audio from video, sends to Modulate API for emotion detection,
applies per-emotion and overall thresholding to trigger clips,
and generates emotion distribution graphs for optimal head/tail trimming.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCH_URL = "https://modulate-developer-apis.com/api/velma-2-stt-batch"
STREAM_URL = "wss://modulate-developer-apis.com/api/velma-2-stt-streaming"

ALL_EMOTIONS = [
    "Neutral", "Calm", "Happy", "Amused", "Excited", "Proud",
    "Affectionate", "Interested", "Hopeful", "Frustrated", "Angry",
    "Contemptuous", "Concerned", "Afraid", "Sad", "Ashamed",
    "Bored", "Tired", "Surprised", "Anxious", "Stressed",
    "Disgusted", "Disappointed", "Confused", "Relieved", "Confident",
]

HIGH_ENERGY_EMOTIONS = {
    "Excited", "Angry", "Frustrated", "Surprised", "Afraid",
    "Amused", "Happy", "Stressed", "Anxious", "Contemptuous",
    "Proud", "Disgusted",
}

LOW_ENERGY_EMOTIONS = {
    "Neutral", "Calm", "Bored", "Tired",
}

EMOTION_WEIGHTS = {
    "Excited": 1.0, "Angry": 0.95, "Surprised": 0.9,
    "Afraid": 0.85, "Frustrated": 0.85, "Contemptuous": 0.8,
    "Stressed": 0.8, "Anxious": 0.75, "Disgusted": 0.75,
    "Amused": 0.7, "Happy": 0.6, "Proud": 0.6,
    "Confident": 0.55, "Concerned": 0.5, "Hopeful": 0.5,
    "Affectionate": 0.45, "Interested": 0.45, "Confused": 0.45,
    "Disappointed": 0.4, "Sad": 0.4, "Ashamed": 0.4,
    "Relieved": 0.35, "Calm": 0.15, "Neutral": 0.1,
    "Bored": 0.05, "Tired": 0.05,
}


def _get_api_key() -> str:
    key = os.environ.get("MODULATE_API_KEY")
    if not key:
        raise EnvironmentError("MODULATE_API_KEY environment variable not set.")
    return key


@dataclass
class ClipThresholds:
    per_emotion_count_threshold: int = 3
    overall_score_threshold: float = 5.0
    window_size_ms: int = 30_000
    buffer_before_ms: int = 15_000
    buffer_after_ms: int = 10_000
    min_clip_duration_ms: int = 15_000
    max_clip_duration_ms: int = 90_000
    merge_gap_ms: int = 10_000


@dataclass
class Utterance:
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
    start_ms: int
    end_ms: int
    trigger_emotion: str
    trigger_type: str
    peak_score: float
    utterances: list = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


# -- Audio extraction -------------------------------------------------------

def extract_audio_from_video(video_path: str, output_format: str = "mp3") -> str:
    output_path = tempfile.mktemp(suffix=f".{output_format}")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame" if output_format == "mp3" else "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-y",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    print(f"[clipper] Extracted audio: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
    return output_path


def get_audio_duration_ms(audio_path: str) -> int:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return int(float(result.stdout.strip()) * 1000)


# -- Batch transcription ----------------------------------------------------

def transcribe_batch_with_emotion(
    audio_path: str, time_offset_ms: int = 0, max_retries: int = 3,
) -> list[Utterance]:
    api_key = _get_api_key()
    print(f"[clipper] Sending {audio_path} to Modulate Batch API...")

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as f:
                response = requests.post(
                    BATCH_URL,
                    headers={"X-API-Key": api_key},
                    files={"upload_file": f},
                    data={
                        "speaker_diarization": "true",
                        "emotion_signal": "true",
                        "accent_signal": "false",
                        "pii_phi_tagging": "false",
                    },
                    timeout=300,
                )
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"[clipper] API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                raise last_error  # type: ignore[misc]

    result = response.json()  # type: ignore[possibly-undefined]
    utterances = []
    for u in result["utterances"]:
        utterances.append(Utterance(
            uuid=u["utterance_uuid"], text=u["text"],
            start_ms=u["start_ms"] + time_offset_ms,
            duration_ms=u["duration_ms"], speaker=u["speaker"],
            language=u["language"], emotion=u["emotion"], accent=u["accent"],
        ))
    print(f"[clipper] Got {len(utterances)} utterances, duration: {result['duration_ms']}ms")
    return utterances


def transcribe_chunked(
    audio_path: str, chunk_duration_ms: int = 300_000, max_chunks: int | None = None,
) -> list[Utterance]:
    total_duration_ms = get_audio_duration_ms(audio_path)
    if total_duration_ms <= chunk_duration_ms:
        return transcribe_batch_with_emotion(audio_path)

    total_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms
    chunks_to_process = min(total_chunks, max_chunks) if max_chunks else total_chunks

    all_utterances: list[Utterance] = []
    for i in range(chunks_to_process):
        offset_ms = i * chunk_duration_ms
        chunk_path = tempfile.mktemp(suffix=".mp3")
        start_s = offset_ms / 1000
        duration_s = min(chunk_duration_ms, total_duration_ms - offset_ms) / 1000
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-ss", str(start_s), "-t", str(duration_s),
            "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-y",
            chunk_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        try:
            chunk_utterances = transcribe_batch_with_emotion(chunk_path, time_offset_ms=offset_ms)
            all_utterances.extend(chunk_utterances)
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
    return all_utterances


# -- Streaming transcription ------------------------------------------------

async def transcribe_streaming_with_emotion(
    audio_path: str, on_utterance=None, chunk_size: int = 8192,
) -> list[Utterance]:
    api_key = _get_api_key()
    url = (
        f"{STREAM_URL}?api_key={api_key}"
        f"&speaker_diarization=true&emotion_signal=true"
        f"&accent_signal=false&pii_phi_tagging=false"
    )
    utterances: list[Utterance] = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            async def send_audio():
                with open(audio_path, "rb") as f:
                    while chunk := f.read(chunk_size):
                        await ws.send_bytes(chunk)
                        await asyncio.sleep(chunk_size / 4000)
                await ws.send_str("")

            send_task = asyncio.create_task(send_audio())
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data["type"] == "utterance":
                            u_data = data["utterance"]
                            u = Utterance(
                                uuid=u_data["utterance_uuid"], text=u_data["text"],
                                start_ms=u_data["start_ms"], duration_ms=u_data["duration_ms"],
                                speaker=u_data["speaker"], language=u_data["language"],
                                emotion=u_data["emotion"], accent=u_data["accent"],
                            )
                            utterances.append(u)
                            if on_utterance:
                                on_utterance(u)
                        elif data["type"] == "done":
                            break
                        elif data["type"] == "error":
                            print(f"[clipper] Stream error: {data['error']}")
                            break
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                        break
            finally:
                if not send_task.done():
                    send_task.cancel()
    return utterances


# -- Clip detection ---------------------------------------------------------

class EmotionClipDetector:
    def __init__(self, thresholds: ClipThresholds | None = None):
        self.thresholds = thresholds or ClipThresholds()

    def detect_clips(self, utterances: list[Utterance]) -> list[ClipCandidate]:
        if not utterances:
            return []
        clips: list[ClipCandidate] = []
        window_ms = self.thresholds.window_size_ms
        sorted_utts = sorted(utterances, key=lambda u: u.start_ms)
        total_duration_ms = max(u.end_ms for u in sorted_utts)
        step_ms = 5000

        for window_start in range(0, total_duration_ms, step_ms):
            window_end = window_start + window_ms
            window_utts = [u for u in sorted_utts if window_start <= u.start_ms < window_end]
            if not window_utts:
                continue

            emotion_counts: dict[str, int] = defaultdict(int)
            for u in window_utts:
                if u.emotion and u.emotion in HIGH_ENERGY_EMOTIONS:
                    emotion_counts[u.emotion] += 1

            for emotion, count in emotion_counts.items():
                if count >= self.thresholds.per_emotion_count_threshold:
                    trigger_time = self._find_peak_time(window_utts, emotion)
                    clips.append(ClipCandidate(
                        start_ms=max(0, trigger_time - self.thresholds.buffer_before_ms),
                        end_ms=min(total_duration_ms, trigger_time + self.thresholds.buffer_after_ms),
                        trigger_emotion=emotion, trigger_type="per_emotion",
                        peak_score=EMOTION_WEIGHTS[emotion] * count,
                        utterances=window_utts,
                    ))

            total_score = sum(u.emotion_score for u in window_utts)
            if total_score >= self.thresholds.overall_score_threshold:
                peak_time = self._find_score_peak(window_utts)
                dominant = max(emotion_counts.items(), key=lambda x: x[1], default=("Mixed", 0))
                clips.append(ClipCandidate(
                    start_ms=max(0, peak_time - self.thresholds.buffer_before_ms),
                    end_ms=min(total_duration_ms, peak_time + self.thresholds.buffer_after_ms),
                    trigger_emotion=dominant[0], trigger_type="overall_score",
                    peak_score=total_score, utterances=window_utts,
                ))

        return self._merge_clips(clips, total_duration_ms)

    @staticmethod
    def _find_peak_time(utterances: list[Utterance], target_emotion: str) -> int:
        matching = [u for u in utterances if u.emotion == target_emotion]
        if not matching:
            return utterances[0].start_ms
        return sum(u.start_ms for u in matching) // len(matching)

    @staticmethod
    def _find_score_peak(utterances: list[Utterance]) -> int:
        if not utterances:
            return 0
        return max(utterances, key=lambda u: u.emotion_score).start_ms

    def _merge_clips(self, clips: list[ClipCandidate], total_duration_ms: int) -> list[ClipCandidate]:
        if not clips:
            return []
        sorted_clips = sorted(clips, key=lambda c: c.start_ms)
        merged = [sorted_clips[0]]
        for clip in sorted_clips[1:]:
            last = merged[-1]
            if clip.start_ms <= last.end_ms + self.thresholds.merge_gap_ms:
                last.end_ms = max(last.end_ms, clip.end_ms)
                if clip.peak_score > last.peak_score:
                    last.peak_score = clip.peak_score
                    last.trigger_emotion = clip.trigger_emotion
                    last.trigger_type = clip.trigger_type
                seen = {u.uuid for u in last.utterances}
                for u in clip.utterances:
                    if u.uuid not in seen:
                        last.utterances.append(u)
                        seen.add(u.uuid)
            else:
                merged.append(clip)

        final: list[ClipCandidate] = []
        for clip in merged:
            dur = clip.duration_ms
            if dur < self.thresholds.min_clip_duration_ms:
                extend = (self.thresholds.min_clip_duration_ms - dur) // 2
                clip.start_ms = max(0, clip.start_ms - extend)
                clip.end_ms = min(total_duration_ms, clip.end_ms + extend)
            elif dur > self.thresholds.max_clip_duration_ms:
                clip.end_ms = clip.start_ms + self.thresholds.max_clip_duration_ms
            final.append(clip)
        return final


# -- Emotion graphing -------------------------------------------------------

def suggest_trim_points(
    emotion_density: np.ndarray, total_duration_s: float, threshold_ratio: float = 0.2,
) -> tuple[float, float]:
    if len(emotion_density) == 0:
        return 0.0, total_duration_s
    peak = np.max(emotion_density)
    threshold = peak * threshold_ratio
    trim_start = 0.0
    for i, val in enumerate(emotion_density):
        if val >= threshold:
            trim_start = max(0, i - 2)
            break
    trim_end = total_duration_s
    for i in range(len(emotion_density) - 1, -1, -1):
        if emotion_density[i] >= threshold:
            trim_end = min(total_duration_s, i + 2)
            break
    return float(trim_start), float(trim_end)


def plot_emotion_distribution(
    utterances: list[Utterance], clip: ClipCandidate, output_path: str = "emotion_distribution.png",
):
    clip_utts = [u for u in utterances if u.start_ms >= clip.start_ms and u.end_ms <= clip.end_ms]
    if not clip_utts:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.suptitle(
        f"Emotion Distribution | {clip.start_ms/1000:.1f}s-{clip.end_ms/1000:.1f}s | "
        f"Trigger: {clip.trigger_emotion} ({clip.trigger_type})",
        fontsize=13, fontweight="bold",
    )

    ax1 = axes[0]
    times = [(u.start_ms - clip.start_ms) / 1000 for u in clip_utts]
    scores = [u.emotion_score for u in clip_utts]
    colors = [
        "red" if u.emotion in HIGH_ENERGY_EMOTIONS
        else "gray" if u.emotion in LOW_ENERGY_EMOTIONS
        else "orange"
        for u in clip_utts
    ]
    ax1.bar(times, scores, width=0.8, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    for t, s, u in zip(times, scores, clip_utts):
        if s > 0.3:
            ax1.text(t, s + 0.02, u.emotion, ha="center", va="bottom", fontsize=7, rotation=45)
    ax1.set_ylabel("Emotion Score")
    ax1.set_ylim(0, 1.15)
    ax1.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="High threshold")
    ax1.legend(loc="upper right")

    ax2 = axes[1]
    clip_duration_s = (clip.end_ms - clip.start_ms) / 1000
    timeline = np.zeros(int(clip_duration_s) + 1)
    for u in clip_utts:
        s_idx = max(0, int((u.start_ms - clip.start_ms) / 1000))
        e_idx = min(len(timeline) - 1, int((u.end_ms - clip.start_ms) / 1000))
        for i in range(s_idx, e_idx + 1):
            timeline[i] += u.emotion_score
    kernel_size = min(5, len(timeline))
    smoothed = np.convolve(timeline, np.ones(kernel_size) / kernel_size, mode="same") if kernel_size else timeline
    ax2.fill_between(np.arange(len(smoothed)), smoothed, alpha=0.4, color="purple")
    ax2.plot(np.arange(len(smoothed)), smoothed, color="purple", linewidth=2)
    ax2.set_ylabel("Emotion Density")
    if len(smoothed) > 0:
        ax2.axvline(x=np.argmax(smoothed), color="red", linestyle="--", alpha=0.7)

    ax3 = axes[2]
    trim_start, trim_end = suggest_trim_points(smoothed, clip_duration_s)
    ax3.barh(0, clip_duration_s, height=0.4, color="lightgray", edgecolor="black")
    ax3.barh(0, trim_end - trim_start, left=trim_start, height=0.4, color="green", alpha=0.6, edgecolor="black")
    ax3.set_xlim(0, clip_duration_s)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[clipper] Emotion graph saved: {output_path}")
    return trim_start, trim_end


# -- Clip cutting -----------------------------------------------------------

def _generate_srt_from_utterances(
    utterances: list,
    clip_start_ms: int,
    clip_end_ms: int,
    output_path: str,
) -> str:
    """Generate SRT file directly from utterances list (not from JSON file)."""
    # Filter utterances within clip range
    clip_utts = [u for u in utterances if u.start_ms < clip_end_ms and u.end_ms > clip_start_ms]
    
    srt_lines = []
    for i, u in enumerate(clip_utts, start=1):
        text = u.text.strip() if hasattr(u, 'text') else str(u.get('text', '')).strip()
        if not text:
            continue
        
        # Get times - handle both Utterance objects and dicts
        if hasattr(u, 'start_ms'):
            start_ms = u.start_ms
            end_ms = u.end_ms
        else:
            start_ms = u.get('start_ms', 0)
            end_ms = u.get('end_ms', start_ms + u.get('duration_ms', 2000))
        
        # Convert to relative (clip-local) timestamps
        rel_start = max(0, start_ms - clip_start_ms)
        rel_end = max(0, end_ms - clip_start_ms)
        
        start_time = _ms_to_srt_time(rel_start)
        end_time = _ms_to_srt_time(rel_end)
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    
    return output_path


def cut_clip_from_video(
    video_path: str, start_ms: int, end_ms: int, output_path: str,
    utterances: list | None = None,
    burn_subtitles: bool = False,
    font_size: int = 24,
    position: str = "bottom",
):
    """
    Cut a clip from video, optionally burning subtitles directly from utterances.
    
    Args:
        video_path: Source video path
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds  
        output_path: Output clip path
        utterances: List of Utterance objects (from Modulate transcription)
        burn_subtitles: If True and utterances provided, burn subtitles into video
        font_size: Subtitle font size
        position: Subtitle position ("bottom", "top", "center")
    """
    start_s = start_ms / 1000
    duration_s = (end_ms - start_ms) / 1000
    
    if burn_subtitles and utterances:
        # Generate temp SRT from utterances
        srt_path = tempfile.mktemp(suffix=".srt")
        _generate_srt_from_utterances(utterances, start_ms, end_ms, srt_path)
        
        # Escape path for ffmpeg filter
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        
        margin_v = 30 if position in ("bottom", "top") else 0
        alignment = 2 if position == "bottom" else 6 if position == "top" else 5
        
        subtitle_filter = (
            f"subtitles='{srt_escaped}':"
            f"force_style='FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"BorderStyle=1,Outline=2,Shadow=1,"
            f"MarginV={margin_v},Alignment={alignment}'"
        )
        
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ss", str(start_s), "-t", str(duration_s),
            "-vf", subtitle_filter,
            "-c:a", "aac",
            "-y", output_path,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp SRT
        if os.path.exists(srt_path):
            os.remove(srt_path)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg clip+subtitle failed: {result.stderr}")
        print(f"[clipper] Clip saved with subtitles: {output_path} ({duration_s:.1f}s)")
    else:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ss", str(start_s), "-t", str(duration_s),
            "-c", "copy", "-y", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg clip cut failed: {result.stderr}")
        print(f"[clipper] Clip saved: {output_path} ({duration_s:.1f}s)")


def accumulate_emotions(utterances: list[Utterance]) -> dict[str, int]:
    totals: dict[str, int] = {e: 0 for e in ALL_EMOTIONS}
    for u in utterances:
        if u.emotion and u.emotion in totals:
            totals[u.emotion] += 1
    return dict(sorted(totals.items(), key=lambda x: -x[1]))


# -- Subtitle generation ----------------------------------------------------

def _ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm)."""
    if ms < 0:
        ms = 0
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def generate_srt_from_transcript(
    transcript_path: str,
    output_path: str | None = None,
    use_relative_times: bool = True,
) -> str:
    """
    Convert a clip transcript JSON to SRT subtitle format.
    
    Args:
        transcript_path: Path to the transcript JSON file
        output_path: Optional output path for SRT file (defaults to same dir as transcript)
        use_relative_times: If True, use relative_start_ms/relative_end_ms (for clipped videos).
                           If False, use absolute start_ms/end_ms.
    
    Returns:
        Path to the generated SRT file
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    
    utterances = transcript.get("utterances", [])
    if not utterances:
        raise ValueError(f"No utterances found in {transcript_path}")
    
    srt_lines = []
    for i, utt in enumerate(utterances, start=1):
        text = utt.get("text", "").strip()
        if not text:
            continue
        
        # Choose timing based on use_relative_times
        if use_relative_times and "relative_start_ms" in utt:
            start_ms = max(0, utt["relative_start_ms"])
            end_ms = max(0, utt["relative_end_ms"])
        else:
            start_ms = utt.get("start_ms", 0)
            end_ms = utt.get("end_ms", start_ms + utt.get("duration_ms", 2000))
        
        start_time = _ms_to_srt_time(start_ms)
        end_time = _ms_to_srt_time(end_ms)
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line separator
    
    srt_content = "\n".join(srt_lines)
    
    if output_path is None:
        output_path = transcript_path.replace(".json", ".srt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    print(f"[clipper] SRT generated: {output_path} ({len(utterances)} subtitles)")
    return output_path


def burn_subtitles_to_video(
    video_path: str,
    srt_path: str,
    output_path: str | None = None,
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
    outline_width: int = 2,
    position: str = "bottom",  # "bottom", "top", "center"
) -> str:
    """
    Burn subtitles into a video using ffmpeg.
    
    Args:
        video_path: Path to the input video
        srt_path: Path to the SRT subtitle file
        output_path: Output path (defaults to video_path with '_subtitled' suffix)
        font_size: Subtitle font size
        font_color: Subtitle text color
        outline_color: Subtitle outline/border color
        outline_width: Outline thickness
        position: Vertical position ("bottom", "top", "center")
    
    Returns:
        Path to the output video with burned subtitles
    """
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_subtitled{ext}"
    
    # Escape path for ffmpeg filter (Windows needs special handling)
    # Replace backslashes with forward slashes and escape colons
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    
    # Build subtitle filter with styling
    margin_v = 30 if position == "bottom" else 30 if position == "top" else 0
    alignment = 2 if position == "bottom" else 6 if position == "top" else 5
    
    # Use ASS styling for better control
    subtitle_filter = (
        f"subtitles='{srt_escaped}':"
        f"force_style='FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"  # White in ASS format (AABBGGRR)
        f"OutlineColour=&H00000000,"  # Black outline
        f"BorderStyle=1,"
        f"Outline={outline_width},"
        f"Shadow=1,"
        f"MarginV={margin_v},"
        f"Alignment={alignment}'"
    )
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", subtitle_filter,
        "-c:a", "copy",
        "-y", output_path,
    ]
    
    print(f"[clipper] Burning subtitles into video...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle burn failed: {result.stderr}")
    
    print(f"[clipper] Subtitled video saved: {output_path}")
    return output_path


def add_subtitles_from_transcript(
    video_path: str,
    transcript_path: str,
    output_path: str | None = None,
    **subtitle_style,
) -> str:
    """
    Convenience function: Generate SRT from transcript and burn into video.
    
    Args:
        video_path: Input video path
        transcript_path: Transcript JSON path
        output_path: Output video path (optional)
        **subtitle_style: Styling options passed to burn_subtitles_to_video
    
    Returns:
        Path to the subtitled video
    """
    srt_path = generate_srt_from_transcript(transcript_path, use_relative_times=True)
    
    try:
        return burn_subtitles_to_video(video_path, srt_path, output_path, **subtitle_style)
    finally:
        # Optionally clean up SRT file
        pass  # Keep SRT for debugging/manual edits


# -- Lightweight emotion-only analysis -------------------------------------

def analyze_emotions(
    video_path: str,
    output_dir: str = "./clips",
    use_streaming: bool = False,
    max_chunks: int | None = None,
) -> dict:
    """Extract audio, send to Modulate for emotion detection, and return results.

    Unlike ``process_video`` this does NOT detect sub-clips, trim, or re-cut
    the video.  It just keeps the original clip as-is and returns the emotion
    ratings alongside it.

    Returns a dict with:
        - ``video_path``: the original clip (unchanged)
        - ``utterances``: list of Utterance objects with emotion labels
        - ``emotion_totals``: {emotion: count} sorted by frequency
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[clipper] Analyzing emotions: {video_path}")

    audio_path = extract_audio_from_video(video_path)

    try:
        if use_streaming:
            utterances = asyncio.run(transcribe_streaming_with_emotion(audio_path))
        else:
            utterances = transcribe_chunked(audio_path, max_chunks=max_chunks)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    emotion_totals = accumulate_emotions(utterances)

    report_path = os.path.join(output_dir, "emotion_report.json")
    report = {
        "video": os.path.basename(video_path),
        "utterance_count": len(utterances),
        "emotions": emotion_totals,
        "utterances": [
            {
                "text": u.text,
                "start_ms": u.start_ms,
                "duration_ms": u.duration_ms,
                "emotion": u.emotion,
                "emotion_score": u.emotion_score,
            }
            for u in utterances
        ],
    }
    import json
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[clipper] Emotion report saved: {report_path}")

    top = [(e, c) for e, c in emotion_totals.items() if c > 0][:5]
    if top:
        print("[clipper] Top emotions: " + ", ".join(f"{e}({c})" for e, c in top))

    return {
        "video_path": video_path,
        "utterances": utterances,
        "emotion_totals": emotion_totals,
    }


# -- Full pipeline ----------------------------------------------------------

def process_video(
    video_path: str,
    output_dir: str = "./clips",
    thresholds: ClipThresholds | None = None,
    use_streaming: bool = False,
    max_chunks: int | None = None,
    burn_subtitles: bool = False,
    subtitle_font_size: int = 24,
    subtitle_position: str = "bottom",
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[clipper] Processing: {video_path}")

    audio_path = extract_audio_from_video(video_path)

    if use_streaming:
        utterances = asyncio.run(transcribe_streaming_with_emotion(audio_path))
    else:
        utterances = transcribe_chunked(audio_path, max_chunks=max_chunks)

    emotion_totals = accumulate_emotions(utterances)
    detector = EmotionClipDetector(thresholds or ClipThresholds())
    clips = detector.detect_clips(utterances)
    print(f"[clipper] Found {len(clips)} clip candidates")

    for i, clip in enumerate(clips):
        graph_path = os.path.join(output_dir, f"clip_{i+1}_emotion.png")
        trim_result = plot_emotion_distribution(utterances, clip, graph_path)
        if trim_result:
            trim_start_s, trim_end_s = trim_result
            clip.start_ms += int(trim_start_s * 1000)
            clip.end_ms = clip.start_ms + int((trim_end_s - trim_start_s) * 1000)

    for i, clip in enumerate(clips):
        clip_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
        cut_clip_from_video(
            video_path, clip.start_ms, clip.end_ms, clip_path,
            utterances=utterances,
            burn_subtitles=burn_subtitles,
            font_size=subtitle_font_size,
            position=subtitle_position,
        )

    if os.path.exists(audio_path):
        os.remove(audio_path)

    print(f"[clipper] Done! {len(clips)} clips saved to {output_dir}")
    return clips, emotion_totals
