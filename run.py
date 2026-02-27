#!/usr/bin/env python3
"""Convenience launcher: ``python run.py``."""

from twitch_monitor.bot import ChatMonitorBot


def main() -> None:
    bot = ChatMonitorBot()
    bot.run()


if __name__ == "__main__":
    main()
