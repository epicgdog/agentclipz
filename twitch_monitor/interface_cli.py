"""Rich-based terminal dashboard for live Twitch chat analytics."""

from __future__ import annotations

import asyncio
import os
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analytics import ChatAnalytics

DEFAULT_REFRESH_INTERVAL = 3.0


def _sentiment_color(score: float) -> str:
    if score >= 0.25:
        return "bright_green"
    if score >= 0.05:
        return "green"
    if score > -0.05:
        return "yellow"
    if score > -0.25:
        return "red"
    return "bright_red"


def _build_speed_panel(metrics: dict) -> Panel:
    mps = metrics["messages_per_second"]
    mpm = metrics["messages_per_minute"]
    total = metrics["messages_in_window"]
    window = metrics["window_seconds"]

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()

    table.add_row("Messages/sec", f"[cyan]{mps:.2f}[/]")
    table.add_row("Messages/min", f"[cyan]{mpm:.1f}[/]")
    table.add_row("In window", f"[cyan]{total}[/]")
    table.add_row("Window size", f"[dim]{window}s[/]")

    bar_len = min(int(mpm), 50)
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


def _build_sentiment_panel(metrics: dict) -> Panel:
    s = metrics["sentiment"]
    avg = s["avg_compound"]
    pos = s["positive_count"]
    neu = s["neutral_count"]
    neg = s["negative_count"]
    total = s["total"]

    color = _sentiment_color(avg)

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()

    if avg >= 0.05:
        label = "Positive"
    elif avg <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    table.add_row("Avg compound", f"[{color}]{avg:+.4f}[/]")
    table.add_row("Mood", f"[{color} bold]{label}[/]")
    table.add_row("", "")

    total_safe = total or 1
    pos_pct = pos / total_safe * 100
    neu_pct = neu / total_safe * 100
    neg_pct = neg / total_safe * 100

    table.add_row("Positive", f"[green]{pos}[/] [dim]({pos_pct:.0f}%)[/]")
    table.add_row("Neutral", f"[yellow]{neu}[/] [dim]({neu_pct:.0f}%)[/]")
    table.add_row("Negative", f"[red]{neg}[/] [dim]({neg_pct:.0f}%)[/]")

    pos_bar = "+" * min(int(pos_pct / 2), 50)
    neu_bar = "~" * min(int(neu_pct / 2), 50)
    neg_bar = "-" * min(int(neg_pct / 2), 50)
    table.add_row("", f"[green]{pos_bar}[/][yellow]{neu_bar}[/][red]{neg_bar}[/]")

    return Panel(table, title="[bold white]Sentiment[/]", border_style=color)


def build_dashboard(metrics: dict) -> Layout:
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
        Layout(name="left"),
        Layout(name="right"),
    )

    layout["left"].split_column(
        Layout(_build_speed_panel(metrics), name="speed"),
        Layout(_build_sentiment_panel(metrics), name="sentiment"),
    )
    layout["right"].update(_build_keywords_panel(metrics))

    return layout


async def run_cli_dashboard(
    analytics: ChatAnalytics,
    refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
) -> None:
    """Run the Rich live dashboard, refreshing on a fixed interval."""
    console = Console()

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        while True:
            metrics = analytics.get_chat_metrics()
            live.update(build_dashboard(metrics))
            await asyncio.sleep(refresh_interval)


async def _run_bot_and_dashboard(
    refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
) -> None:
    """Start the TwitchIO bot and the CLI dashboard concurrently."""
    from .bot import ChatMonitorBot

    bot = ChatMonitorBot()
    bot.report_interval = 0  # disable plain-text reporting when CLI is active

    dashboard_task = asyncio.create_task(
        run_cli_dashboard(bot.analytics, refresh_interval)
    )

    try:
        await bot.start()
    finally:
        dashboard_task.cancel()


def main() -> None:
    """Entry point for ``python -m twitch_monitor.interface_cli``."""
    try:
        asyncio.run(_run_bot_and_dashboard())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
