"""ADIP structured logging utilities.

Provides a consistent logging interface across all ADIP modules with
support for plain-text and JSON-structured output, file handlers, and
automatic suppression of noisy third-party loggers.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------
# Noisy loggers to suppress by default
# ---------------------------------------------------------------
_NOISY_LOGGERS = [
    'prophet',
    'cmdstanpy',
    'matplotlib',
    'matplotlib.font_manager',
    'PIL',
    'urllib3',
    'numexpr',
]

_DEFAULT_FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
_DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# ---------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields emitted: ``timestamp``, ``name``, ``level``, ``message``,
    and optionally ``exc_info``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string.
        """
        log_entry: dict = {
            'timestamp': datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def get_logger(
    name: str,
    level: str = 'INFO',
    json_format: bool = False,
) -> logging.Logger:
    """Create or retrieve a named logger with a consistent format.

    If the logger already has handlers attached, this function returns
    it as-is to avoid duplicate output.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
        level: Logging level string (e.g. ``'DEBUG'``, ``'INFO'``).
        json_format: If ``True``, emit structured JSON logs instead of
            plain text.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)

    if json_format:
        console_handler.setFormatter(_JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
        )

    logger.addHandler(console_handler)

    # Prevent propagation to the root logger
    logger.propagate = False

    # Suppress noisy third-party loggers
    _suppress_noisy_loggers()

    return logger


def setup_file_handler(
    logger: logging.Logger,
    log_dir: str,
    filename: Optional[str] = None,
    json_format: bool = False,
) -> logging.FileHandler:
    """Attach a file handler to an existing logger.

    The log directory is created if it does not exist.

    Args:
        logger: The logger to which the file handler will be added.
        log_dir: Directory path where log files are stored.
        filename: Optional filename override.  Defaults to
            ``<logger_name>.log``.
        json_format: If ``True``, the file handler uses JSON formatting.

    Returns:
        The newly created :class:`logging.FileHandler`.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        safe_name = logger.name.replace('.', '_')
        filename = f'{safe_name}.log'

    file_path = log_path / filename
    file_handler = logging.FileHandler(str(file_path), encoding='utf-8')
    file_handler.setLevel(logger.level)

    if json_format:
        file_handler.setFormatter(_JSONFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
        )

    logger.addHandler(file_handler)
    return file_handler


def _suppress_noisy_loggers(level: int = logging.WARNING) -> None:
    """Set noisy third-party loggers to WARNING or above.

    Args:
        level: The minimum level to set on noisy loggers.
    """
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)
