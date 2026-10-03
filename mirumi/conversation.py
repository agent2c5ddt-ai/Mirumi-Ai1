"""Append-only conversation and summary storage with tolerant JSONL reads."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import uuid


@dataclass(frozen=True)
class Turn:
    time: str
    role: str
    text: str


class ConversationStore:
    def __init__(self, root):
        self.root = Path(root)
        self.conversations_dir = self.root / "memory" / "conversations"
        self.summaries_path = self.root / "memory" / "summaries.jsonl"
        self.warnings = []

    def new_session_path(self):
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        return self.conversations_dir / f"session_{stamp}_{uuid.uuid4().hex[:8]}.jsonl"

    def append_message(self, path, role, text):
        record = {
            "time": datetime.now().astimezone().isoformat(timespec="microseconds"),
            "role": role,
            "text": str(text),
        }
        self._append_jsonl(Path(path), record)
        return Turn(record["time"], role, record["text"])

    def recent_messages(self, max_messages=20, after_time=None, max_files=8):
        if max_messages <= 0 or not self.conversations_dir.exists():
            return []
        paths = sorted(
            self.conversations_dir.glob("session_*.jsonl"),
            key=lambda item: (item.stat().st_mtime, item.name),
        )[-max_files:]
        collected = []
        cutoff = _timestamp(after_time) if after_time else None
        for path in paths:
            tail = deque(maxlen=max_messages)
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if len(line) > 20000:
                            self.warnings.append(
                                f"Skipped oversized conversation line "
                                f"{path.name}:{line_number}."
                            )
                            continue
                        try:
                            value = json.loads(line)
                        except (json.JSONDecodeError, TypeError):
                            self.warnings.append(
                                f"Skipped malformed conversation line "
                                f"{path.name}:{line_number}."
                            )
                            continue
                        turn = _turn_from_record(value)
                        if turn is None:
                            continue
                        if cutoff is not None:
                            turn_time = _timestamp(turn.time)
                            if turn_time is not None and turn_time <= cutoff:
                                continue
                        tail.append(turn)
            except OSError as exc:
                self.warnings.append(f"Could not read {path.name}: {exc}")
                continue
            collected.extend(tail)
        return collected[-max_messages:]

    def latest_summary(self):
        values = self._read_jsonl(self.summaries_path)
        for value in reversed(values):
            if (
                isinstance(value, dict)
                and isinstance(value.get("summary"), str)
                and value["summary"].strip()
            ):
                return value
        return None

    def save_summary(self, summary, through_time):
        summary = str(summary).strip()
        if not summary or len(summary) > 5000:
            raise ValueError("A summary must contain 1–5000 characters.")
        record = {
            "id": uuid.uuid4().hex[:12],
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "through_time": through_time or "",
            "source": "local_model_summary",
            "status": "context_only",
            "summary": summary,
        }
        self._append_jsonl(self.summaries_path, record)
        return record

    def _read_jsonl(self, path):
        records = []
        if not path.exists():
            return records
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    if len(line) > 20000:
                        self.warnings.append(
                            f"Skipped oversized JSONL line {path.name}:{line_number}."
                        )
                        continue
                    try:
                        records.append(json.loads(line))
                    except (json.JSONDecodeError, TypeError):
                        self.warnings.append(
                            f"Skipped malformed JSONL line {path.name}:{line_number}."
                        )
        except OSError as exc:
            self.warnings.append(f"Could not read {path.name}: {exc}")
        return records

    @staticmethod
    def _append_jsonl(path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()


def _turn_from_record(value):
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    role = value.get("role")
    when = value.get("time")
    if not isinstance(text, str) or not text.strip() or not isinstance(role, str):
        return None
    if role.lower() in {"mirumi", "assistant"}:
        role = "assistant"
    elif role.lower() == "user":
        role = "user"
    else:
        return None
    if not isinstance(when, str):
        when = ""
    return Turn(when, role, text.strip())


def _timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None