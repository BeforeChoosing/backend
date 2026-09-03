"""Password reset email delivery through the configured Resend account."""

from __future__ import annotations

import json
import os
import urllib.request


def send_password_reset_email(recipient: str, code: str) -> bool:
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    sender = os.getenv("PASSWORD_RESET_FROM", "").strip() or os.getenv("ERROR_ALERT_FROM", "").strip()
    if not api_key or not sender:
        return False
    payload = {
        "from": sender,
        "to": [recipient],
        "subject": "【选择之前】重置密码验证码",
        "text": f"你的重置密码验证码是：{code}\n验证码 10 分钟内有效。如非本人操作，请忽略此邮件。",
    }
    request = urllib.request.Request(
        "https://api.resend.com/emails",
        method="POST",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "before-choosing-auth/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return 200 <= response.status < 300
    except Exception:
        return False
