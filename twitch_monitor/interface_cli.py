"""Rich-based terminal dashboard for live Twitch chat analytics."""

from __future__ import annotations

import asyncio
import os
import logging

logger = logging.getLogger(__name__)

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analytics import ChatAnalytics

DEFAULT_REFRESH_INTERVAL = 3.0


def _build_speed_panel(metrics: dict) -> Panel:
    mps = metrics["messages_per_second"]
    total = metrics["messages_in_window"]
    window = metrics["window_seconds"]

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()

    table.add_row("Messages/sec", f"[cyan]{mps:.2f}[/]")
    table.add_row("In window", f"[cyan]{total}[/]")
    table.add_row("Window size", f"[dim]{window}s[/]")

    bar_len = min(int(mps * 60), 50)
    bar = "█" * bar_len + "░" * (50 - bar_len)
    table.add_row("Activity", f"[cyan]{bar}[/]")

    return Panel(table, title="[bold white]Chat Speed[/]", border_style="cyan")


def _build_keywords_panel(metrics: dict) -> Panel:
    keywords = metrics["top_keywords"]

    if not keywords:
        return Panel(
            Text("No messages yet...", style="dim italic"),
            title="[bold white]Top Keywords[/]",
            border_style="magenta",
        )

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("Keyword", ratio=3)
    table.add_column("Count", justify="right", width=6)
    table.add_column("Bar", ratio=4)

    max_count = keywords[0][1] if keywords else 1
    for i, (word, count) in enumerate(keywords[:10], start=1):
        bar_len = int((count / max_count) * 30) if max_count else 0
        bar = "▓" * bar_len
        table.add_row(
            str(i),
            f"[bold]{word}[/]",
            str(count),
            f"[magenta]{bar}[/]",
        )

    return Panel(table, title="[bold white]Top Keywords[/]", border_style="magenta")


def _build_clip_trigger_panel(trigger) -> Panel:
    """Build a small status panel for the clip-trigger state machine."""
    from .clip_trigger import TriggerState

    state = trigger.state if trigger else TriggerState.IDLE
    clips = trigger.clip_count if trigger else 0
    last_info = trigger._last_clip_info if trigger else ""

    state_color = {
        TriggerState.IDLE: "green",
        TriggerState.RECORDING: "bold red",
        TriggerState.PROCESSING: "bold yellow",
    }.get(state, "white")

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()

    table.add_row("State", f"[{state_color}]{state.value}[/]")
    table.add_row("Clips", f"[cyan]{clips}[/]")

    if trigger:
        cfg = trigger.config
        table.add_row("Start ≥", f"[dim]{cfg.start_mps:.1f} msg/s[/]")
        table.add_row("Stop ≤", f"[dim]{cfg.stop_mps:.1f} msg/s[/]")

    if last_info:
        table.add_row("Last", f"[dim]{last_info}[/]")

    return Panel(table, title="[bold white]Clip Trigger[/]", border_style="red")


def build_dashboard(metrics: dict, trigger=None) -> Layout:
    """Build a full Rich Layout from a metrics dict."""
    layout = Layout()
    layout.split_column(
        Layout(
            Panel(
                Text("Twitch Chat Monitor", justify="center", style="bold white"),
                border_style="bright_blue",
            ),
            name="header",
            size=3,
        ),
        Layout(name="body"),
        Layout(
            Panel(
                Text(
                    f"Refreshing every {DEFAULT_REFRESH_INTERVAL:.0f}s  |  "
                    f"Window: {metrics['window_seconds']}s  |  "
                    f"Press Ctrl+C to exit",
                    justify="center",
                    style="dim",
                ),
                border_style="bright_blue",
            ),
            name="footer",
            size=3,
        ),
    )

    layout["body"].split_row(
        Layout(_build_speed_panel(metrics), name="speed", ratio=1),
        Layout(_build_keywords_panel(metrics), name="keywords", ratio=2),
        Layout(_build_clip_trigger_panel(trigger), name="trigger", ratio=1),
    )

    return layout


async def run_cli_dashboard(
    analytics: ChatAnalytics,
    refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
    trigger=None,
) -> None:
    """Run the Rich live dashboard, refreshing on a fixed interval."""
    console = Console()

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        while True:
            metrics = analytics.get_chat_metrics()
            live.update(build_dashboard(metrics, trigger=trigger))
            await asyncio.sleep(refresh_interval)


async def _run_bot_and_dashboard(
    refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
) -> None:
    """Start the TwitchIO bot, the clip-trigger pipeline, and the CLI dashboard."""
    from .bot import ChatMonitorBot
    from .clip_trigger import ClipTriggerConfig, run_clip_pipeline

    bot = ChatMonitorBot()
    bot.report_interval = 0

    trigger = None
    buf = None
    config = ClipTriggerConfig()
    stream_url = config.stream_url or f"https://twitch.tv/{bot.channel_name}"
    config.stream_url = stream_url

    try:
        buf, trigger = await run_clip_pipeline(
            bot.analytics, config=config, channel=bot.channel_name,
        )
    except Exception as exc:
        logger.error("[clip-trigger] Could not start clip pipeline: %s", exc)
        logger.error("[clip-trigger] Dashboard will run without clip recording.")

    dashboard_task = asyncio.create_task(
        run_cli_dashboard(bot.analytics, refresh_interval, trigger=trigger)
    )

    try:
        await bot.start()
    finally:
        dashboard_task.cancel()
        if buf:
            await buf.stop()


def main() -> None:
    """Entry point for ``python -m twitch_monitor.interface_cli``."""
    try:
        asyncio.run(_run_bot_and_dashboard())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
