"""Character configuration loader; source text files remain authoritative."""

import re
from pathlib import Path


CHARACTER_FILES = {
    "identity": "identity.txt",
    "personality": "personality.txt",
    "behavior": "behavior.txt",
    "speech_style": "speech_style.txt",
    "stable_facts": "character_facts.txt",
    "rules": "character_rules.txt",
}

_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*")


class CharacterProfile:
    def __init__(self, root, sections, legacy_core="", warnings=None):
        self.root = Path(root)
        self.sections = sections
        self.legacy_core = legacy_core
        self.warnings = warnings or []

    @classmethod
    def load(cls, root):
        root = Path(root)
        sections = {}
        warnings = []
        for key, filename in CHARACTER_FILES.items():
            path = root / filename
            try:
                sections[key] = path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                sections[key] = ""
            except (OSError, UnicodeError) as exc:
                sections[key] = ""
                warnings.append(f"Could not read {filename}: {exc}")
        try:
            legacy_core = (root / "core.txt").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            legacy_core = ""
        except (OSError, UnicodeError) as exc:
            legacy_core = ""
            warnings.append(f"Could not read core.txt: {exc}")
        return cls(root, sections, legacy_core, warnings)

    def render(self, query="", max_chars=3000):
        max_chars = max(200, int(max_chars))
        blocks = []
        identity = self.sections.get("identity", "")
        if identity:
            blocks.append(("IDENTITY", identity, 350))

        legacy_rule = _section(self.legacy_core, "Identity rule:", "Personality:")
        if legacy_rule:
            blocks.append(("IDENTITY BOUNDARY", legacy_rule, 260))

        if not any(self.sections.values()) and self.legacy_core:
            blocks.append(("CORE CHARACTER PROFILE", self.legacy_core, max_chars))
        else:
            # The compact core remains a compatibility fallback for facts that
            # are absent from an older/incomplete set of split character files.
            missing = [
                key
                for key in ("personality", "behavior", "speech_style")
                if not self.sections.get(key)
            ]
            if missing and self.legacy_core:
                blocks.append(
                    ("LEGACY CORE PROFILE", self.legacy_core, min(900, max_chars))
                )
            for key, heading, cap in (
                ("personality", "PERSONALITY", 620),
                ("behavior", "BEHAVIOR", 650),
                ("speech_style", "SPEECH STYLE", 480),
                ("stable_facts", "STABLE CHARACTER FACTS", 430),
                ("rules", "CHARACTER RULES", 440),
            ):
                text = self.sections.get(key, "")
                if text:
                    blocks.append((heading, text, cap))

        output = []
        remaining = max_chars
        for heading, text, section_cap in blocks:
            if remaining <= 0:
                break
            allowance = min(section_cap, remaining)
            excerpt = _relevant_excerpt(text, query, allowance)
            if not excerpt:
                continue
            block = f"{heading}\n{excerpt}"
            if len(block) > remaining:
                block = _clip(block, remaining)
            output.append(block)
            remaining -= len(block) + 2

        if not output:
            output.append(
                "CHARACTER CONFIGURATION\n"
                "Use the established Mirumi character files when available. "
                "Admit uncertainty rather than inventing character facts."
            )
        return "\n\n".join(output)


def _section(text, start, end):
    if not text or start not in text:
        return ""
    value = text.split(start, 1)[1]
    if end in value:
        value = value.split(end, 1)[0]
    return value.strip()


def _relevant_excerpt(text, query, limit):
    if limit <= 0:
        return ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return _clip(text.strip(), limit)
    query_words = {word for word in _WORD.findall(query.lower()) if len(word) > 2}
    ranked = []
    for index, paragraph in enumerate(paragraphs):
        words = set(_WORD.findall(paragraph.lower()))
        score = len(words & query_words)
        # Preserve the introductory characterization when the request does not
        # point to any one section; query-matched paragraphs can still displace
        # lower-value detail as the budget gets tight.
        ranked.append((score, -index, index, paragraph))
    ranked.sort(reverse=True)
    chosen = []
    used = 0
    for _, _, index, paragraph in ranked:
        cost = len(paragraph) + (2 if chosen else 0)
        if used + cost <= limit:
            chosen.append((index, paragraph))
            used += cost
        elif not chosen:
            chosen.append((index, _clip(paragraph, limit)))
            break
    chosen.sort(key=lambda item: item[0])
    return "\n\n".join(paragraph for _, paragraph in chosen)


def _clip(text, limit):
    if len(text) <= limit:
        return text
    if limit < 8:
        return text[:limit]
    clipped = text[: limit - 4]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip() + " ..."