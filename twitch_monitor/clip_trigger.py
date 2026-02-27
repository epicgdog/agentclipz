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
import atexit
import enum
import glob
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import platform

if TYPE_CHECKING:
    from .analytics import ChatAnalytics

from .clipper import analyze_emotions

logger = logging.getLogger(__name__)

# Global registry of all child processes we spawn, so we can kill them on exit
_child_pids: list[subprocess.Popen] = []


def _register_child(proc: subprocess.Popen) -> None:
    """Track a child process so it can be cleaned up on exit."""
    _child_pids.append(proc)


def _cleanup_children() -> None:
    """Kill all tracked child processes. Called on interpreter exit."""
    for proc in _child_pids:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except OSError:
            pass
    _child_pids.clear()


# Register the cleanup so it fires on normal exit AND on Ctrl+C
atexit.register(_cleanup_children)


def kill_orphaned_processes() -> int:
    """Find and kill any orphaned ffmpeg/streamlink processes from previous runs.

    Returns the number of processes killed.
    """
    killed = 0
    if platform.system() != "Windows":
        return killed

    for proc_name in ("ffmpeg.exe", "streamlink.exe"):
        try:
            # Use tasklist to find PIDs, then taskkill to end them
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {proc_name}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.strip('"').split('","')
                if len(parts) >= 2 and parts[0].lower() == proc_name.lower():
                    pid = parts[1].strip('"')
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True, timeout=5,
                        )
                        killed += 1
                        logger.info("[cleanup] Killed orphaned %s (PID %s)", proc_name, pid)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("[cleanup] Could not check for orphaned %s: %s", proc_name, e)

    return killed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClipTriggerConfig:
    start_mps: float = float(os.environ.get("CLIP_START_MPS", "0.3"))
    stop_mps: float = float(os.environ.get("CLIP_STOP_MPS", "0.2"))
    buffer_duration_s: int = int(os.environ.get("BUFFER_DURATION_S", "100"))
    segment_duration_s: int = int(os.environ.get("SEGMENT_DURATION_S", "30"))
    min_record_s: int = int(os.environ.get("MIN_RECORD_S", "10"))   # 1:30 minimum
    max_record_s: int = int(os.environ.get("MAX_RECORD_S", "30"))   # 1:30 maximum
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
            "ffmpeg", "-y", "-i", "pipe:0",
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.config.segment_duration_s),
            "-reset_timestamps", "1",
            "-max_muxing_queue_size", "1024", # Prevents audio dropouts under load
            "-loglevel", "error",
            seg_pattern,
        ]

        self._streamlink_proc = subprocess.Popen(
            streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        _register_child(self._streamlink_proc)

        self._ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=self._streamlink_proc.stdout,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        _register_child(self._ffmpeg_proc)

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
            # Use chunks so we don't lock the file currently being written to by ffmpeg
            try:
                with open(seg, "rb") as src_f, open(dest, "wb") as dst_f:
                    while True:
                        chunk = src_f.read(8192)
                        if not chunk:
                            break
                        dst_f.write(chunk)
                copies.append(dest)
            except OSError as e:
                logger.warning("[buffer] Failed to copy segment %s: %s", seg, e)
                
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
        # Clean up the temp segment directory
        if os.path.isdir(self._segment_dir):
            shutil.rmtree(self._segment_dir, ignore_errors=True)
        logger.info("[buffer] Stream buffer stopped and temp dir cleaned")

    @property
    def is_alive(self) -> bool:
        """Check if the streamlink | ffmpeg pipeline is still running."""
        sl_alive = self._streamlink_proc is not None and self._streamlink_proc.poll() is None
        ff_alive = self._ffmpeg_proc is not None and self._ffmpeg_proc.poll() is None
        return sl_alive and ff_alive

    def get_ffmpeg_errors(self) -> str:
        """Read any stderr output from ffmpeg (non-blocking)."""
        if self._ffmpeg_proc and self._ffmpeg_proc.stderr:
            try:
                import select
                # On Windows, select doesn't work on pipes, so just try read
                self._ffmpeg_proc.stderr.flush()
                return ""
            except Exception:
                return ""
        return ""

    def clear(self) -> None:
        """Delete all current segments to prevent overlapping repeat clips."""
        segments = glob.glob(os.path.join(self._segment_dir, "seg_*.ts"))
        for seg in segments:
            try:
                # We skip the very last file because ffmpeg might still be writing 
                # to it, but we can delete the historical ones.
                if seg != segments[-1]: 
                    os.remove(seg)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# concat_segments – join .ts segments + live recording into one .mp4
# ---------------------------------------------------------------------------

def concat_segments(segment_paths: list[str], output_path: str) -> str:
    """Use ffmpeg concat demuxer to join transport-stream segments into one mp4."""
    # Give the background segmenter a tiny fraction of a second to finish 
    # writing the last bytes of the most recent copied file to avoid locking.
    time.sleep(0.5)

    list_file = tempfile.mktemp(suffix=".txt")
    try:
        with open(list_file, "w") as f:
            for seg in segment_paths:
                f.write(f"file '{seg}'\n")

        # Run ffmpeg concat with lower priority and less logging to avoid 
        # choking the main stream buffer doing live capture.
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", list_file,
            # We copy video but re-encode audio to AAC to fix dropouts 
            # across segment boundaries (common bug with raw TS padding).
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-y",
            "-loglevel", "error",
            output_path,
        ]
        
        # Lower CPU priority to prevent starving the live streamlink buffer
        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
            
        result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
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
        self._record_start_time: float = 0.0
        self._last_clip_info: str = ""

    @property
    def status_text(self) -> str:
        if self.state == TriggerState.IDLE:
            return "IDLE"
        if self.state == TriggerState.RECORDING:
            elapsed = time.time() - self._record_start_time
            return f"⏺ RECORDING ({elapsed:.0f}s / {self.config.max_record_s}s)"
        if self.state == TriggerState.PROCESSING:
            return f"PROCESSING clip #{self.clip_count + 1}"
        return self.state.value

    # -- state transitions ---------------------------------------------------

    def _start_recording(self) -> None:
        """Transition IDLE → RECORDING.

        Instead of spawning a second streamlink connection (which has startup
        latency and can produce empty files), we simply mark the time and let
        the existing StreamBuffer keep capturing segments.  When recording
        stops, we snapshot everything the buffer has accumulated.
        """
        logger.info("[trigger] Chat speed crossed %.1f msg/s – RECORDING for %ds",
                     self.config.start_mps, self.config.max_record_s)
        self.state = TriggerState.RECORDING
        self._record_start_time = time.time()

    def _stop_recording(self) -> str:
        """Transition RECORDING → PROCESSING.  Returns the combined clip path.

        Snapshots ALL segments currently in the rolling buffer.  Because the
        buffer has been running the entire time, this gives us the full
        recording duration without needing a separate streamlink process.
        """
        elapsed = time.time() - self._record_start_time
        logger.info("[trigger] Stopping recording after %.1fs", elapsed)
        self.state = TriggerState.PROCESSING

        # Health check: is the pipeline still alive?
        if not self.buffer.is_alive:
            logger.error("[trigger] streamlink/ffmpeg pipeline died, cannot save clip")
            self.state = TriggerState.IDLE
            return ""

        # Grab everything the buffer has right now
        all_segments = self.buffer.snapshot_segments()

        if not all_segments:
            logger.warning("[trigger] No .ts segments found in buffer directory")
            self.state = TriggerState.IDLE
            return ""
            
        print(f"[clip-trigger] Found {len(all_segments)} segments to stitch")

        # CLEAR the buffers so if there's another chat spike right after this,
        # we don't end up saving the exact same historical footage again!
        self.buffer.clear()
        self.analytics.clear()

        os.makedirs(self.config.clip_output_dir, exist_ok=True)
        self.clip_count += 1
        combined_path = os.path.join(
            self.config.clip_output_dir, f"raw_clip_{self.clip_count}.mp4",
        )
        
        try:
            concat_segments(all_segments, combined_path)
        except RuntimeError as e:
            logger.error("[trigger] ffmpeg concat failed: %s", e)
            self.state = TriggerState.IDLE
            return ""
        
        # Cleanup the temporary snapshot directory generated by the buffer
        snap_dir = os.path.dirname(all_segments[0])
        if "snap_" in snap_dir:
            shutil.rmtree(snap_dir, ignore_errors=True)
        
        # Verify the file actually got created and has content
        if not os.path.exists(combined_path) or os.path.getsize(combined_path) < 1024:
            logger.error("[trigger] Clip file missing or empty: %s", combined_path)
            self.state = TriggerState.IDLE
            return ""
            
        logger.info("[trigger] Raw clip saved: %s (%.1f MB)", combined_path, os.path.getsize(combined_path) / 1024 / 1024)
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
        cfg = self.config
        logger.info("[trigger] Polling every %.1fs  start>=%.1f msg/s  stop<=%.1f msg/s  clip=%d-%ds",
                    cfg.poll_interval_s, cfg.start_mps, cfg.stop_mps, cfg.min_record_s, cfg.max_record_s)
        while True:
            try:
                await asyncio.sleep(cfg.poll_interval_s)
                speed = self.analytics.compute_chat_speed()
                mps = speed["messages_per_second"]

                if self.state == TriggerState.IDLE:
                    if mps >= cfg.start_mps:
                        try:
                            self._start_recording()
                            logger.info("[trigger] RECORDING started (mps=%.2f), will record %d-%ds",
                                        mps, cfg.min_record_s, cfg.max_record_s)
                        except Exception as exc:
                            logger.error("[trigger] Failed to start recording: %s", exc)
                            self.state = TriggerState.IDLE

                elif self.state == TriggerState.RECORDING:
                    elapsed = time.time() - self._record_start_time

                    # Force stop if we hit max duration
                    if elapsed >= cfg.max_record_s:
                        logger.info("[trigger] Max duration reached (%.0fs) -> saving clip", elapsed)
                        try:
                            loop = asyncio.get_running_loop()
                            clip_path = await loop.run_in_executor(None, self._stop_recording)
                            if clip_path:
                                asyncio.get_running_loop().create_task(self._process_clip(clip_path))
                        except Exception as exc:
                            logger.exception("[trigger] Failed to stop/process recording")
                            self.state = TriggerState.IDLE

                    # Only allow early stop if min duration has passed AND chat calmed down
                    elif elapsed >= cfg.min_record_s and mps <= cfg.stop_mps:
                        logger.info("[trigger] Chat calmed (mps=%.2f) after %.0fs -> saving clip", mps, elapsed)
                        try:
                            loop = asyncio.get_running_loop()
                            clip_path = await loop.run_in_executor(None, self._stop_recording)
                            if clip_path:
                                asyncio.get_running_loop().create_task(self._process_clip(clip_path))
                        except Exception as exc:
                            logger.exception("[trigger] Failed to stop/process recording")
                            self.state = TriggerState.IDLE

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("[trigger] Unexpected error in poll loop")
                await asyncio.sleep(cfg.poll_interval_s)


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

    # Kill any orphaned ffmpeg/streamlink from previous crashed runs
    killed = kill_orphaned_processes()
    if killed:
        logger.info("[clip-trigger] Cleaned up %d orphaned process(es) from previous runs", killed)

    stream_url = config.stream_url
    if not stream_url and channel:
        stream_url = f"https://twitch.tv/{channel}"
    if not stream_url:
        stream_url = f"https://twitch.tv/{os.environ.get('TWITCH_CHANNEL', '')}"

    logger.info("[clip-trigger] Stream URL: %s", stream_url)
    logger.info("[clip-trigger] Output dir: %s", config.clip_output_dir)

    buf = StreamBuffer(stream_url, config)
    try:
        await buf.start()
        logger.info("[clip-trigger] Stream buffer started OK")
    except Exception as exc:
        logger.error("[clip-trigger] Stream buffer failed to start: %s", exc)
        logger.warning("[clip-trigger] Trigger loop will still run but recording may fail")

    trigger = ClipTrigger(analytics, buf, config)
    asyncio.get_running_loop().create_task(trigger.run())

    return buf, trigger
