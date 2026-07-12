import json
import re
from datetime import datetime, timezone
from pathlib import Path


class SessionStore:
    """Persist conversational memory under outputs/sessions/<session_id>/."""

    def __init__(self, root="outputs/sessions"):
        self.root = Path(root)

    def session_dir(self, session_id):
        safe_id = _safe_session_id(session_id)
        path = self.root / safe_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append_message(self, session_id, role, content, metadata=None):
        path = self.session_dir(session_id) / "messages.jsonl"
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content or "",
            "metadata": metadata or {},
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def recent_messages(self, session_id, limit=8):
        path = self.session_dir(session_id) / "messages.jsonl"
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records[-limit:]

    def context_text(self, session_id, limit=8, max_chars=5000):
        messages = self.recent_messages(session_id, limit=limit)
        lines = []
        for message in messages:
            content = _compact_text(message.get("content", ""), max_chars=1200)
            if content:
                lines.append(f"{message.get('role', 'unknown')}: {content}")
        text = "\n".join(lines)
        return text[-max_chars:]

    def read_state(self, session_id, name):
        path = self.session_dir(session_id) / f"{_safe_state_name(name)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def write_state(self, session_id, name, payload):
        path = self.session_dir(session_id) / f"{_safe_state_name(name)}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    def clear_state(self, session_id, name):
        path = self.session_dir(session_id) / f"{_safe_state_name(name)}.json"
        if path.exists():
            path.unlink()


def _safe_session_id(session_id):
    if not session_id:
        return "default"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(session_id)).strip(".-")
    return safe or "default"


def _safe_state_name(name):
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(name or "")).strip(".-")
    return safe or "state"


def _compact_text(text, max_chars=500):
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."
