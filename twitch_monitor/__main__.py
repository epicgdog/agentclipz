"""Entry point: ``python -m twitch_monitor``."""

from .bot import ChatMonitorBot


def main() -> None:
    bot = ChatMonitorBot()
    bot.run()


if __name__ == "__main__":
    main()
