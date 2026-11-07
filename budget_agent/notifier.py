"""Alerts: always logged to the workspace; optionally pushed to Slack / any webhook / email."""
from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path


def _post_json(url: str, payload: dict) -> str:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 (user-configured URL)
        return f"{r.status}"


def send_alerts(alerts: list[dict], log_path: Path) -> list[str]:
    """Deliver alerts on every configured channel. Returns the channels used."""
    if not alerts:
        return []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log_path.open("a") as f:
        for a in alerts:
            f.write(json.dumps({"timestamp": ts, **a}) + "\n")
    used = ["log"]
    text = "*Budget Agent alerts*\n" + "\n".join(f"• [{a['severity'].upper()}] {a['message']}" for a in alerts)

    if url := os.getenv("SLACK_WEBHOOK_URL"):
        try:
            _post_json(url, {"text": text})
            used.append("slack")
        except Exception as e:  # never let a notification failure break the agent
            used.append(f"slack-failed: {e}")
    if url := os.getenv("ALERT_WEBHOOK_URL"):
        try:
            _post_json(url, {"alerts": alerts, "timestamp": ts})
            used.append("webhook")
        except Exception as e:
            used.append(f"webhook-failed: {e}")
    if (host := os.getenv("SMTP_HOST")) and (to := os.getenv("ALERT_EMAIL_TO")):
        try:
            msg = EmailMessage()
            msg["Subject"] = f"Budget Agent: {len(alerts)} alert(s)"
            msg["From"] = os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "budget-agent@localhost"))
            msg["To"] = to
            msg.set_content(text.replace("*", ""))
            with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=10) as s:
                s.starttls()
                if os.getenv("SMTP_USER"):
                    s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD", ""))
                s.send_message(msg)
            used.append("email")
        except Exception as e:
            used.append(f"email-failed: {e}")
    return used


def read_alerts(log_path: Path, limit: int = 50) -> list[dict]:
    if not log_path.exists():
        return []
    lines = log_path.read_text().strip().splitlines()
    return [json.loads(x) for x in lines[-limit:]][::-1]
