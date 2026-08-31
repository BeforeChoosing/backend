"""Structured operational logs. Never serialize exception messages or payloads."""
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from app.services.request_context import get_request_context

logger = logging.getLogger('before_choosing.runtime')
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
_configured_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger.setLevel(getattr(logging, _configured_level, logging.INFO))
logger.propagate = False

_METHODS = {
    'debug': logger.debug,
    'info': logger.info,
    'warn': logger.warning,
    'error': logger.error,
}


def log_event(event: str, *, level: str = 'info', error: Exception | None = None, **fields):
    if level not in _METHODS:
        raise ValueError(f'unsupported log level: {level}')
    context = get_request_context()
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'level': level, 'event': event,
        'request_id': context.request_id, 'user_id': context.user_id or None,
    }
    # An allowlist prevents accidental logging of credentials, answers or headers.
    allowed = {'method', 'route', 'status_code', 'duration_ms', 'error_code',
               'client_request_id', 'model', 'input_tokens', 'output_tokens', 'upstream_status'}
    record.update({k: v for k, v in fields.items() if k in allowed})
    if error is not None:
        record['exception_type'] = type(error).__name__
        record['frames'] = [
            {'file': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name}
            for frame in traceback.extract_tb(error.__traceback__)[-8:]
        ]
    _METHODS[level](json.dumps(record, ensure_ascii=False))
