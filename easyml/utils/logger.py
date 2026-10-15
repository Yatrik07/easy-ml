import os
import logging
import tempfile
import time
from logging.handlers import RotatingFileHandler
from threading import Lock

from easyml.config.paths import LOGS_DIR


_LOGGER_LOCK = Lock()
_ACTIVE_LOG_DIR = None
RETRY_ROLLOVER_SECONDS = 60


def _env_number(name, default):
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return float(default)


def max_log_bytes():
    """Size at which a log file is rotated (``EASYML_LOG_MAX_MB``, default 5 MB; 0 = never)."""
    return int(_env_number("EASYML_LOG_MAX_MB", 5) * 1024 * 1024)


def log_backups():
    """How many rotated files are kept next to each log (``EASYML_LOG_BACKUPS``, default 3)."""
    return int(_env_number("EASYML_LOG_BACKUPS", 3))


def log_file_name(base):
    """``logs.log`` for the studio; long-running commands set ``EASYML_LOG_NAME`` (``serve_logs.log``, ...)."""
    prefix = "".join(c for c in os.environ.get("EASYML_LOG_NAME", "") if c.isalnum() or c in "-_")
    return f"{prefix}_{base}" if prefix else base


class SafeRotatingFileHandler(RotatingFileHandler):
    """
    Rotates like RotatingFileHandler, but never fails a log call when the rename is refused (on Windows
    another process may have the file open): it keeps writing to the current file and tries again later.
    """

    _retry_at = 0.0

    def shouldRollover(self, record):
        if time.monotonic() < self._retry_at:
            return False
        return super().shouldRollover(record)

    def doRollover(self):
        try:
            super().doRollover()
        except OSError:
            self._retry_at = time.monotonic() + RETRY_ROLLOVER_SECONDS
            if self.stream is None:
                self.stream = self._open()

class InfoLevelFilter(logging.Filter):
    """
    Keep INFO and WARNING records for logs.log; errors go to error_logs.log.
    """
    def filter(self, record):
        return logging.INFO <= record.levelno < logging.ERROR

logger = logging.getLogger('custom_logger')
logger.setLevel(logging.INFO)
logger.propagate = False

def _resolve_log_dir(preferred_dir=LOGS_DIR):
    candidates = [
        preferred_dir,
        os.path.join(tempfile.gettempdir(), "easy-ml", "logs"),
    ]

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            probe_path = os.path.join(candidate, ".write_test")
            with open(probe_path, "a", encoding="utf-8"):
                pass
            try:
                os.remove(probe_path)
            except OSError:
                pass
            return candidate
        except OSError:
            continue

    return None


def _build_formatter():
    return logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def _rotating_handler(path):
    return SafeRotatingFileHandler(path, maxBytes=max_log_bytes(), backupCount=log_backups(), encoding='utf-8')


def _build_file_handler(filename, level, add_info_filter=False):
    global _ACTIVE_LOG_DIR

    if _ACTIVE_LOG_DIR is None:
        _ACTIVE_LOG_DIR = _resolve_log_dir()

    if _ACTIVE_LOG_DIR is None:
        return None

    filename = log_file_name(filename)
    try:
        handler = _rotating_handler(os.path.join(_ACTIVE_LOG_DIR, filename))
    except OSError:
        fallback_dir = _resolve_log_dir(os.path.join(tempfile.gettempdir(), "easy-ml", "logs"))
        if fallback_dir is None:
            return None
        _ACTIVE_LOG_DIR = fallback_dir
        handler = _rotating_handler(os.path.join(_ACTIVE_LOG_DIR, filename))

    handler.setFormatter(_build_formatter())
    handler.setLevel(level)
    if add_info_filter:
        handler.addFilter(InfoLevelFilter())
    return handler


def configure_logger(force=False):
    global _ACTIVE_LOG_DIR

    with _LOGGER_LOCK:
        if force:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

        if logger.handlers and not force:
            return logger

        _ACTIVE_LOG_DIR = _resolve_log_dir()

        file_handler = _build_file_handler("logs.log", logging.INFO, add_info_filter=True)
        error_handler = _build_file_handler("error_logs.log", logging.ERROR)

        if file_handler is not None:
            logger.addHandler(file_handler)
        if error_handler is not None:
            logger.addHandler(error_handler)

        # Library users only see warnings on the console; the `easy-ml` CLI raises this to INFO.
        if os.environ.get("EASYML_NO_CONSOLE_LOGS") != "1":
            console_level = os.environ.get("EASYML_CONSOLE_LOG_LEVEL", "WARNING").upper()
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(_build_formatter())
            console_handler.setLevel(getattr(logging, console_level, logging.WARNING))
            logger.addHandler(console_handler)

        return logger


def get_logger():
    """
    Retrieves the global custom logger instance.

    Returns:
        logging.Logger: The configured custom logger object.
    """
    return configure_logger()


def get_log_directory():
    configure_logger()
    return _ACTIVE_LOG_DIR or LOGS_DIR

def write_log(message: str, level: str, module_name: str):
    """
    Writes a log message to the configured log files and console.

    Args:
        message (str): The log message content.
        level (str): The logging level ('info', 'error', 'warning'). Defaults to 'info'.
        module_name (str): The name of the module or file where the log originated (often os.path.basename(__file__)).
    """
    level_map = {
        'debug': logging.DEBUG,  # dropped unless the logger level is lowered: routine per-call messages
        'info': logging.INFO,
        'error': logging.ERROR,
        'warning': logging.WARNING
    }
    level_no = level_map.get(level, logging.INFO)
    if not logger.isEnabledFor(level_no):
        return
    configure_logger()
    logger.log(level_no, f"{module_name} - {message}")


def close_all_file_handlers():
    """
    Closes the Easy-ML log file handlers.
    Call this BEFORE deleting the logs directory to avoid WinError 32 on Windows.
    """
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def reinit_file_handlers():
    """
    Re-creates file handlers after the logs directory has been recreated.
    Call this AFTER os.makedirs(LOGS_DIR) to restore logging.
    """
    configure_logger(force=True)
