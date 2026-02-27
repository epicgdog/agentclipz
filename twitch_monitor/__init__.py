import os as _os
import ssl as _ssl

if not _os.path.exists(_ssl.get_default_verify_paths().openssl_cafile or ""):
    try:
        import certifi as _certifi
        _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    except ImportError:
        pass

from .analytics import ChatAnalytics
from .bot import ChatMonitorBot
from .interface_cli import run_cli_dashboard

__all__ = ["ChatAnalytics", "ChatMonitorBot", "run_cli_dashboard"]
