"""Native macOS notification helper (osascript wrapper).

Usage:
    from ufc_pred.notify import notify
    notify(title="🥊 UFC Bet", subtitle="Muhammad vs Bonfim", message="$7.42 on Bonfim")

A separate CLI mode lets you smoke-test the wiring:
    python -m ufc_pred.notify "title" "subtitle" "message"
"""

from __future__ import annotations

import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request


def _osa_escape(s: str) -> str:
    """Escape a string for inclusion in an osascript string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _ssl_context() -> ssl.SSLContext | None:
    """CA bundle for the ntfy POST, independent of the interpreter's build.

    A conda env bakes an absolute CA path in at build time, so renaming the
    directory above it silently breaks every HTTPS verify from that env.
    certifi ships with requests (a hard dependency), so prefer it and fall
    back to the stdlib default if it is somehow missing.
    """
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def push_phone(
    title: str,
    subtitle: str = "",
    message: str = "",
) -> bool:
    """Relay a notification to the phone via ntfy.sh. No-op if unconfigured.

    Reads the topic from NTFY_TOPIC (and optional NTFY_SERVER, default
    https://ntfy.sh). Subscribe to that topic in the ntfy iOS app to receive
    these on the iPhone. Returns True on a successful POST, False otherwise.
    Failure here never raises — the Mac notification must still go out.
    """
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return False
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")

    # ntfy body is the notification text; title/tags go in headers. Combine the
    # subtitle into the body so the phone shows the same info as the Mac banner.
    body = message if not subtitle else f"{subtitle}\n{message}"
    headers = {
        "Title": title.encode("utf-8"),
        "Priority": "high",
        "Tags": "boxing_glove",
    }
    req = urllib.request.Request(
        f"{server}/{topic}",
        data=body.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5, context=_ssl_context()):
            return True
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[notify] ntfy push failed: {e}", file=sys.stderr)
        return False


def notify(
    title: str,
    subtitle: str = "",
    message: str = "",
    sound: str = "Glass",
) -> bool:
    """Send a macOS notification. Returns True on success, False on failure.

    Title length is hard-capped by macOS (~50 chars before truncation in the
    banner); message wraps at ~140. We don't enforce a length cap — the
    caller can truncate if needed.
    """
    parts = [f'display notification "{_osa_escape(message)}"', f'with title "{_osa_escape(title)}"']
    if subtitle:
        parts.append(f'subtitle "{_osa_escape(subtitle)}"')
    if sound:
        parts.append(f'sound name "{_osa_escape(sound)}"')
    script = " ".join(parts)
    # Relay to the phone first — best effort, never blocks the Mac banner.
    push_phone(title, subtitle, message)
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            timeout=5,
            capture_output=True,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"[notify] failed: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    # Load .env so NTFY_TOPIC is available when smoke-testing standalone.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    args = sys.argv[1:]
    if not args:
        notify("UFC bet runner", "Smoke test", "If you see this, notifications work.")
    else:
        notify(*args)
