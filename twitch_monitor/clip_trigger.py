"""Chat-speed-triggered clip recording with rolling stream buffer.

This module contains:
- StreamBuffer: manages a rolling 2-minute stream buffer via streamlink + ffmpeg segmenter
- ClipTrigger: state machine (IDLE / RECORDING / PROCESSING) that polls ChatAnalytics
  and orchestrates recording start/stop
- concat_segments: joins buffer segments + live recording into one file
- run_clip_pipeline: the main async loop that ties everything together
"""

from __future__ import annotations

import asyncio
import enum
import glob
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analytics import ChatAnalytics

from .clipper import analyze_emotions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClipTriggerConfig:
    start_mps: float = float(os.environ.get("CLIP_START_MPS", "1.5"))
    stop_mps: float = float(os.environ.get("CLIP_STOP_MPS", "1.0"))
    buffer_duration_s: int = int(os.environ.get("BUFFER_DURATION_S", "120"))
    segment_duration_s: int = int(os.environ.get("SEGMENT_DURATION_S", "30"))
    poll_interval_s: float = 2.0
    stream_url: str | None = os.environ.get("TWITCH_STREAM_URL")
    modulate_api_key: str | None = os.environ.get("MODULATE_API_KEY")
    clip_output_dir: str = os.environ.get("CLIP_OUTPUT_DIR", "./clips")

    @property
    def segments_to_keep(self) -> int:
        return max(1, self.buffer_duration_s // self.segment_duration_s)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TriggerState(enum.Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"


# ---------------------------------------------------------------------------
# StreamBuffer – rolling 2-min segment capture via streamlink + ffmpeg
# ---------------------------------------------------------------------------

class StreamBuffer:
    """Continuously captures a live Twitch stream into rolling .ts segment files."""

    def __init__(self, stream_url: str, config: ClipTriggerConfig) -> None:
        self.stream_url = stream_url
        self.config = config
        self._segment_dir = tempfile.mkdtemp(prefix="streambuf_")
        self._streamlink_proc: subprocess.Popen | None = None
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._running = False

    @property
    def segment_dir(self) -> str:
        return self._segment_dir

    async def start(self) -> None:
        """Launch the streamlink | ffmpeg pipeline in the background."""
        if self._running:
            return
        self._running = True
        logger.info("[buffer] Starting stream buffer → %s", self._segment_dir)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._launch_pipeline)
        asyncio.get_running_loop().create_task(self._prune_loop())

    def _launch_pipeline(self) -> None:
        seg_pattern = os.path.join(self._segment_dir, "seg_%05d.ts")

        streamlink_cmd = [
            "streamlink", self.stream_url, "best", "-O",
            "--twitch-disable-ads",
        ]
        ffmpeg_cmd = [
            "ffmpeg", "-i", "pipe:0",
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.config.segment_duration_s),
            "-reset_timestamps", "1",
            seg_pattern,
        ]

        self._streamlink_proc = subprocess.Popen(
            streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=self._streamlink_proc.stdout,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if self._streamlink_proc.stdout:
            self._streamlink_proc.stdout.close()

    async def _prune_loop(self) -> None:
        """Delete old segments to maintain the rolling window."""
        keep = self.config.segments_to_keep
        while self._running:
            await asyncio.sleep(self.config.segment_duration_s / 2)
            segments = sorted(glob.glob(os.path.join(self._segment_dir, "seg_*.ts")))
            while len(segments) > keep:
                os.remove(segments.pop(0))

    def snapshot_segments(self) -> list[str]:
        """Return copies of all current buffer segments (for use during recording)."""
        segments = sorted(glob.glob(os.path.join(self._segment_dir, "seg_*.ts")))
        snap_dir = tempfile.mkdtemp(prefix="snap_")
        copies = []
        for seg in segments:
            dest = os.path.join(snap_dir, os.path.basename(seg))
            shutil.copy2(seg, dest)
            copies.append(dest)
        return copies

    async def stop(self) -> None:
        self._running = False
        for proc in (self._ffmpeg_proc, self._streamlink_proc):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        logger.info("[buffer] Stream buffer stopped")


# ---------------------------------------------------------------------------
# concat_segments – join .ts segments + live recording into one .mp4
# ---------------------------------------------------------------------------

def concat_segments(segment_paths: list[str], output_path: str) -> str:
    """Use ffmpeg concat demuxer to join transport-stream segments into one mp4."""
    list_file = tempfile.mktemp(suffix=".txt")
    try:
        with open(list_file, "w") as f:
            for seg in segment_paths:
                f.write(f"file '{seg}'\n")

        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy", "-y",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr[:500]}")
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)

    logger.info("[concat] Merged %d segments → %s", len(segment_paths), output_path)
    return output_path


# ---------------------------------------------------------------------------
# ClipTrigger – polls ChatAnalytics and drives the state machine
# ---------------------------------------------------------------------------

class ClipTrigger:
    """State machine that monitors chat speed and triggers clip recording."""

    def __init__(
        self,
        analytics: "ChatAnalytics",
        buffer: StreamBuffer,
        config: ClipTriggerConfig | None = None,
    ) -> None:
        self.analytics = analytics
        self.buffer = buffer
        self.config = config or ClipTriggerConfig()
        self.state = TriggerState.IDLE
        self.clip_count = 0
        self._live_record_proc: subprocess.Popen | None = None
        self._live_record_path: str | None = None
        self._buffer_snapshot: list[str] = []
        self._last_clip_info: str = ""

    @property
    def status_text(self) -> str:
        if self.state == TriggerState.IDLE:
            return "IDLE"
        if self.state == TriggerState.RECORDING:
            return "RECORDING"
        if self.state == TriggerState.PROCESSING:
            return f"PROCESSING clip #{self.clip_count + 1}"
        return self.state.value

    # -- state transitions ---------------------------------------------------

    def _start_recording(self) -> None:
        """Transition IDLE → RECORDING."""
        logger.info("[trigger] Chat speed crossed %.1f msg/s – starting recording",
                     self.config.start_mps)
        self.state = TriggerState.RECORDING
        self._buffer_snapshot = self.buffer.snapshot_segments()
        self._live_record_path = tempfile.mktemp(suffix=".ts")

        cmd = [
            "streamlink", self.buffer.stream_url, "best", "-O",
            "--twitch-disable-ads",
        ]
        ffmpeg_cmd = [
            "ffmpeg", "-i", "pipe:0",
            "-c", "copy", "-y",
            self._live_record_path,
        ]
        streamlink_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._live_record_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=streamlink_proc.stdout,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if streamlink_proc.stdout:
            streamlink_proc.stdout.close()

    def _stop_recording(self) -> str:
        """Transition RECORDING → PROCESSING.  Returns the combined clip path."""
        logger.info("[trigger] Chat speed dropped below %.1f msg/s – stopping recording",
                     self.config.stop_mps)
        self.state = TriggerState.PROCESSING

        if self._live_record_proc and self._live_record_proc.poll() is None:
            self._live_record_proc.terminate()
            try:
                self._live_record_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._live_record_proc.kill()

        all_segments = list(self._buffer_snapshot)
        if self._live_record_path and os.path.exists(self._live_record_path):
            all_segments.append(self._live_record_path)

        os.makedirs(self.config.clip_output_dir, exist_ok=True)
        self.clip_count += 1
        combined_path = os.path.join(
            self.config.clip_output_dir, f"raw_clip_{self.clip_count}.mp4",
        )
        concat_segments(all_segments, combined_path)
        self._buffer_snapshot = []
        self._live_record_path = None
        return combined_path

    async def _process_clip(self, clip_path: str) -> None:
        """Send the clip for emotion analysis (no editing/trimming), then return to IDLE."""
        try:
            logger.info("[trigger] Analyzing emotions for clip: %s", clip_path)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                analyze_emotions,
                clip_path,
                os.path.join(self.config.clip_output_dir, f"clip_{self.clip_count}"),
            )
            totals = result.get("emotion_totals", {})
            top = [(e, c) for e, c in totals.items() if c > 0][:3]
            summary = ", ".join(f"{e}({c})" for e, c in top) if top else "none"
            self._last_clip_info = f"Clip #{self.clip_count}: {summary}"
        except Exception:
            logger.exception("[trigger] Error analyzing clip")
            self._last_clip_info = f"Clip #{self.clip_count} failed"
        finally:
            self.state = TriggerState.IDLE

    # -- main loop -----------------------------------------------------------

    async def run(self) -> None:
        """Poll chat speed and drive the state machine. Runs until cancelled."""
        print(f"[clip-trigger] Polling every {self.config.poll_interval_s}s  "
              f"start≥{self.config.start_mps} msg/s  stop≤{self.config.stop_mps} msg/s")
        while True:
            try:
                await asyncio.sleep(self.config.poll_interval_s)
                speed = self.analytics.compute_chat_speed()
                mps = speed["messages_per_second"]

                if self.state == TriggerState.IDLE:
                    if mps >= self.config.start_mps:
                        try:
                            self._start_recording()
                            print(f"[clip-trigger] RECORDING started (mps={mps:.2f})")
                        except Exception as exc:
                            print(f"[clip-trigger] Failed to start recording: {exc}")
                            self.state = TriggerState.IDLE

                elif self.state == TriggerState.RECORDING:
                    if mps <= self.config.stop_mps:
                        try:
                            clip_path = self._stop_recording()
                            print(f"[clip-trigger] Recording stopped → processing {clip_path}")
                            asyncio.get_running_loop().create_task(self._process_clip(clip_path))
                        except Exception as exc:
                            print(f"[clip-trigger] Failed to stop/process recording: {exc}")
                            self.state = TriggerState.IDLE

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[clip-trigger] Unexpected error in poll loop: {exc}")
                await asyncio.sleep(self.config.poll_interval_s)


# ---------------------------------------------------------------------------
# run_clip_pipeline – convenience function to wire everything together
# ---------------------------------------------------------------------------

async def run_clip_pipeline(
    analytics: "ChatAnalytics",
    config: ClipTriggerConfig | None = None,
    channel: str | None = None,
) -> tuple[StreamBuffer, ClipTrigger]:
    """Start the rolling buffer and clip-trigger loop.

    Returns the (StreamBuffer, ClipTrigger) so callers can inspect state or stop them.
    """
    config = config or ClipTriggerConfig()

    stream_url = config.stream_url
    if not stream_url and channel:
        stream_url = f"https://twitch.tv/{channel}"
    if not stream_url:
        stream_url = f"https://twitch.tv/{os.environ.get('TWITCH_CHANNEL', '')}"

    print(f"[clip-trigger] Stream URL: {stream_url}")
    print(f"[clip-trigger] Output dir: {config.clip_output_dir}")

    buf = StreamBuffer(stream_url, config)
    try:
        await buf.start()
        print("[clip-trigger] Stream buffer started OK")
    except Exception as exc:
        print(f"[clip-trigger] Stream buffer failed to start: {exc}")
        print("[clip-trigger] Trigger loop will still run but recording may fail")

    trigger = ClipTrigger(analytics, buf, config)
    asyncio.get_running_loop().create_task(trigger.run())

    return buf, trigger
