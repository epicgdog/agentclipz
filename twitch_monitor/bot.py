"""TwitchIO bot that listens to chat and periodically reports analytics."""

from __future__ import annotations

import asyncio
import os
import time

from twitchio.ext import commands

from .analytics import ChatAnalytics

# ---------------------------------------------------------------------------
# Configuration (override via env vars or constructor args)
# ---------------------------------------------------------------------------

DEFAULT_REPORT_INTERVAL = 10  # seconds between console metric dumps
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_TOP_N_KEYWORDS = 10


class ChatMonitorBot(commands.Bot):
    """Connects to a Twitch channel, records every chat message, and
    periodically prints live analytics to the console."""

    def __init__(
        self,
        token: str | None = None,
        channel: str | None = None,
        prefix: str = "!",
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        top_n_keywords: int = DEFAULT_TOP_N_KEYWORDS,
        report_interval: int = DEFAULT_REPORT_INTERVAL,
    ) -> None:
        self._twitch_token = token or os.environ["TWITCH_TOKEN"]
        self._channel = channel or os.environ["TWITCH_CHANNEL"]
        self.report_interval = report_interval

        super().__init__(
            token=self._twitch_token,
            prefix=prefix,
            initial_channels=[self._channel],
        )

        self.analytics = ChatAnalytics(
            window_seconds=window_seconds,
            top_n_keywords=top_n_keywords,
        )

    # -- lifecycle events --------------------------------------------------

    async def event_ready(self) -> None:
        print(f"[monitor] Connected as {self.nick}")
        print(f"[monitor] Watching channel: {self._channel}")
        print(
            f"[monitor] Window={self.analytics.window_seconds}s  "
            f"Report every {self.report_interval}s  "
            f"Top keywords={self.analytics.top_n_keywords}"
        )
        self.loop.create_task(self._report_loop())

    async def event_message(self, message) -> None:  # type: ignore[override]
        if message.echo:
            return
        self.analytics.add_message(message.content, timestamp=time.time())
        await self.handle_commands(message)

    # -- periodic reporting ------------------------------------------------

    async def _report_loop(self) -> None:
        """Background task that prints a metrics snapshot on a fixed interval."""
        if self.report_interval <= 0:
            return
        await asyncio.sleep(self.report_interval)  # initial delay
        while True:
            metrics = self.analytics.get_chat_metrics()
            self._print_metrics(metrics)
            await asyncio.sleep(self.report_interval)

    @staticmethod
    def _print_metrics(m: dict) -> None:
        speed = (
            f"{m['messages_per_minute']:.1f} msg/min "
            f"({m['messages_per_second']:.2f} msg/s, "
            f"{m['messages_in_window']} in window)"
        )

        kw = ", ".join(f"{word}({count})" for word, count in m["top_keywords"][:5])
        if not kw:
            kw = "(none yet)"

        s = m["sentiment"]
        sentiment = (
            f"avg={s['avg_compound']:+.3f}  "
            f"+{s['positive_count']} / "
            f"~{s['neutral_count']} / "
            f"-{s['negative_count']}"
        )

        print(
            f"\n{'=' * 60}\n"
            f"  Speed:     {speed}\n"
            f"  Keywords:  {kw}\n"
            f"  Sentiment: {sentiment}\n"
            f"{'=' * 60}"
        )
