"""Structured, append-only memory with provenance and explicit review."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import uuid


MEMORY_CATEGORIES = {
    "episodic",
    "factual",
    "relationship",
    "preference",
    "important_event",
    "learned_behavior",
    "temporary_context",
}
_ACTIONS = {"approved", "rejected", "forgotten"}
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*")
_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "been", "being",
    "but", "can", "could", "did", "does", "for", "from", "have", "her",
    "his", "into", "its", "just", "not", "our", "she", "that", "their",
    "them", "there", "these", "they", "this", "was", "were", "what",
    "when", "where", "which", "who", "with", "would", "you", "your",
}


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    content: str
    category: str
    source: str
    confidence: float
    created_at: str
    status: str
    evidence: str = ""
    legacy_sources: tuple = ()


class MemoryStore:
    def __init__(self, root):
        self.root = Path(root)
        self.records_path = self.root / "memory" / "records.jsonl"
        self.decisions_path = self.root / "memory" / "decisions.jsonl"
        self.warnings = []

    def add(self, content, category="factual", source="user_explicit",
            confidence=1.0, status="approved", evidence=""):
        content = _validate_content(content)
        if category not in MEMORY_CATEGORIES:
            raise ValueError(f"Unsupported memory category: {category}")
        if status not in {"approved", "pending"}:
            raise ValueError("New memories must be approved or pending.")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("Memory confidence must be between 0 and 1.")
        if status == "approved" and source != "user_explicit":
            raise ValueError(
                "Only an explicit user remember command can create an "
                "immediately approved memory."
            )
        record = {
            "id": uuid.uuid4().hex[:12],
            "content": content,
            "category": category,
            "source": str(source)[:80],
            "confidence": round(float(confidence), 3),
            "created_at": _now(),
            "status": status,
            "evidence": str(evidence)[:1200],
        }
        self._append(self.records_path, record)
        return self._record_from_dict(record)

    def all_records(self):
        records = []
        seen = set()
        for value in self._read_jsonl(self.records_path):
            record = self._record_from_dict(value)
            if record and record.id not in seen:
                records.append(record)
                seen.add(record.id)
        records.extend(item for item in self._legacy_records() if item.id not in seen)
        decisions = self._decisions()
        result = []
        for record in records:
            status = decisions.get(record.id, record.status)
            result.append(
                MemoryRecord(
                    record.id, record.content, record.category, record.source,
                    record.confidence, record.created_at, status, record.evidence,
                    record.legacy_sources,
                )
            )
        return result

    def approved(self):
        return [record for record in self.all_records() if record.status == "approved"]

    def pending(self):
        return [record for record in self.all_records() if record.status == "pending"]

    def get(self, record_id):
        return next(
            (item for item in self.all_records() if item.id == record_id), None
        )

    def approve(self, record_id):
        return self._decide(record_id, "approved")

    def reject(self, record_id):
        return self._decide(record_id, "rejected")

    def forget(self, record_id):
        return self._decide(record_id, "forgotten")

    def retrieve(self, query, limit=3):
        query_words = _meaningful_words(query)
        if not query_words:
            return []
        ranked = []
        for record in self.approved():
            memory_words = _meaningful_words(record.content)
            overlap = query_words & memory_words
            if not overlap:
                continue
            coverage = len(overlap) / max(1, len(query_words))
            density = len(overlap) / max(1, len(memory_words))
            score = (coverage * 0.7 + density * 0.3) * record.confidence
            ranked.append((score, record.created_at, record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in ranked[:max(0, limit)]]

    def _decide(self, record_id, action):
        if action not in _ACTIONS:
            raise ValueError("Invalid memory decision.")
        record = self.get(record_id)
        if record is None or (
            record.status in {"rejected", "forgotten"} and action != "approved"
        ):
            return False
        self._append(
            self.decisions_path,
            {
                "id": uuid.uuid4().hex[:12],
                "record_id": record_id,
                "action": action,
                "actor": "user",
                "created_at": _now(),
            },
        )
        return True

    def _decisions(self):
        decisions = {}
        for value in self._read_jsonl(self.decisions_path):
            if not isinstance(value, dict):
                continue
            record_id, action = value.get("record_id"), value.get("action")
            if isinstance(record_id, str) and action in _ACTIONS:
                decisions[record_id] = action
        return decisions

    def _legacy_records(self):
        candidates = []
        for path in (
            self.root / "character_memory.txt",
            self.root / "relationship_memory.txt",
            self.root / "self_developed.txt",
            self.root / "development_history.txt",
        ):
            candidates.extend(self._legacy_file_candidates(path))
        development_dir = self.root / "memory" / "self_development"
        if development_dir.exists():
            for path in sorted(development_dir.glob("*.txt")):
                candidates.extend(self._legacy_file_candidates(path))

        merged = []
        for candidate in candidates:
            words = _meaningful_words(candidate.content)
            match_index = None
            for index, existing in enumerate(merged):
                if candidate.category != existing.category:
                    continue
                prior = _meaningful_words(existing.content)
                union = words | prior
                similarity = len(words & prior) / max(1, len(union))
                if similarity >= 0.55:
                    match_index = index
                    break
            if match_index is None:
                merged.append(candidate)
            else:
                existing = merged[match_index]
                merged[match_index] = MemoryRecord(
                    existing.id,
                    existing.content,
                    existing.category,
                    existing.source,
                    existing.confidence,
                    existing.created_at,
                    existing.status,
                    existing.evidence,
                    tuple(sorted(set(existing.legacy_sources + candidate.legacy_sources))),
                )
        return merged

    def _legacy_file_candidates(self, path):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError) as exc:
            self.warnings.append(f"Could not read legacy memory {path.name}: {exc}")
            return []
        if not text:
            return []

        candidates = []
        if path.name == "self_developed.txt":
            candidates = re.findall(
                r"(?im)^Trait:\s*(.+?)(?=\n\s*\n|$)", text
            )
        elif path.name == "development_history.txt":
            candidates = re.findall(
                r"(?im)^Development\s+\d+:\s*(.+?)(?=\n\s*\n|$)", text
            )
        elif path.parent.name == "self_development":
            candidates = re.findall(
                r"(?ims)^Development:\s*(.+?)(?=\n\s*\n|$)", text
            )
        else:
            candidates = [
                item.strip()
                for item in re.split(r"\n\s*\n", text)
                if item.strip()
            ]

        result = []
        is_development = (
            path.name in {"self_developed.txt", "development_history.txt"}
            or path.parent.name == "self_development"
        )
        category = (
            "learned_behavior"
            if is_development
            else "relationship"
            if path.name == "relationship_memory.txt"
            else "factual"
        )
        for raw in candidates:
            try:
                content = _validate_content(raw, maximum=1200, minimum=5)
            except ValueError:
                continue
            fingerprint = hashlib.sha256(
                " ".join(_WORD.findall(content.lower())).encode("utf-8")
            ).hexdigest()[:12]
            result.append(
                MemoryRecord(
                    id=f"legacy-{fingerprint}",
                    content=content,
                    category=category,
                    source="legacy_unverified",
                    confidence=0.35,
                    created_at="",
                    status="pending",
                    evidence="",
                    legacy_sources=(str(path.relative_to(self.root)),),
                )
            )
        return result

    def _read_jsonl(self, path):
        values = []
        if not path.exists():
            return values
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    if len(line) > 20000:
                        self.warnings.append(
                            f"Skipped oversized memory line {path.name}:{line_number}."
                        )
                        continue
                    try:
                        values.append(json.loads(line))
                    except (json.JSONDecodeError, TypeError):
                        self.warnings.append(
                            f"Skipped malformed memory line {path.name}:{line_number}."
                        )
        except OSError as exc:
            self.warnings.append(f"Could not read {path.name}: {exc}")
        return values

    @staticmethod
    def _append(path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            handle.flush()

    @staticmethod
    def _record_from_dict(value):
        if not isinstance(value, dict):
            return None
        content = value.get("content")
        category = value.get("category")
        record_id = value.get("id")
        status = value.get("status")
        confidence = value.get("confidence")
        if (
            not isinstance(record_id, str)
            or not record_id
            or not isinstance(content, str)
            or not content.strip()
            or category not in MEMORY_CATEGORIES
            or status not in {"approved", "pending"}
        ):
            return None
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return None
        if not 0.0 <= confidence <= 1.0:
            return None
        sources = value.get("legacy_sources", ())
        if not isinstance(sources, (list, tuple)):
            sources = ()
        return MemoryRecord(
            id=record_id,
            content=content.strip()[:1200],
            category=category,
            source=str(value.get("source", "unknown"))[:80],
            confidence=confidence,
            created_at=str(value.get("created_at", ""))[:80],
            status=status,
            evidence=str(value.get("evidence", ""))[:1200],
            legacy_sources=tuple(str(item)[:240] for item in sources),
        )


def _validate_content(content, maximum=1200, minimum=3):
    if not isinstance(content, str):
        raise ValueError("Memory content must be text.")
    content = content.strip()
    if len(content) < minimum or len(content) > maximum:
        raise ValueError(
            f"Memory content must contain {minimum}–{maximum} characters."
        )
    if "\x00" in content:
        raise ValueError("Memory content cannot contain null characters.")
    return content


def _meaningful_words(text):
    return {
        word
        for word in _WORD.findall(text.lower())
        if len(word) > 2 and word not in _STOP_WORDS
    }


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")