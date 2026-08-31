"""Rate-limited, payload-free severe error alerts through Resend."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
import json
import os
from threading import Lock
import time
import urllib.request

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='error-alert')
_lock = Lock()
_last_sent: dict[str, float] = {}


def queue_error_alert(event: str, fields: dict) -> None:
    if os.getenv('ERROR_ALERT_ENABLED', '').lower() not in {'1', 'true', 'yes'}:
        return
    api_key = os.getenv('RESEND_API_KEY', '').strip()
    recipient = os.getenv('ERROR_ALERT_TO', '').strip()
    sender = os.getenv('ERROR_ALERT_FROM', '').strip()
    if not api_key or not recipient or not sender:
        return
    safe = {key: str(fields.get(key, ''))[:160] for key in (
        'request_id', 'method', 'route', 'status_code', 'error_code', 'exception_type'
    )}
    fingerprint = '|'.join([event, safe['route'], safe['error_code'], safe['exception_type']])
    now = time.monotonic()
    cooldown = max(60, int(os.getenv('ERROR_ALERT_COOLDOWN_SECONDS', '900')))
    with _lock:
        if now - _last_sent.get(fingerprint, 0) < cooldown:
            return
        _last_sent[fingerprint] = now
    _executor.submit(_send, api_key, recipient, sender, event, safe)


def _send(api_key: str, recipient: str, sender: str, event: str, fields: dict) -> str | None:
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = ''.join(
        f'<tr><th style="text-align:left;padding:6px 12px 6px 0">{escape(key)}</th>'
        f'<td style="padding:6px 0">{escape(value or "-")}</td></tr>'
        for key, value in {'event': event, 'time_utc': timestamp, **fields}.items()
    )
    payload = {
        'from': sender, 'to': [recipient],
        'subject': f'[Before Choosing] 严重错误 {fields.get("error_code") or event}',
        'html': '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:640px">'
                '<h2>服务器严重错误告警</h2><p>以下信息已脱敏，请按请求编号查看服务器日志。</p>'
                f'<table>{rows}</table></div>',
    }
    request = urllib.request.Request(
        'https://api.resend.com/emails', method='POST',
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'},
    )
    # Failure intentionally stays inside the worker; it must not recurse into alerts.
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            result = json.loads(response.read())
            return str(result.get('id') or '') or None
    except Exception:
        return None


def _reset_for_tests() -> None:
    with _lock:
        _last_sent.clear()
