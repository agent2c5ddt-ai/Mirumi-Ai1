"""Lightweight task assessment and targeted response verification."""

import re
from dataclasses import dataclass


_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")
_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "before",
    "being", "between", "can", "could", "does", "for", "from", "have",
    "into", "just", "make", "more", "need", "please", "should", "that",
    "them", "then", "there", "these", "they", "this", "through", "with",
    "would", "your", "you", "what", "when", "where", "which", "while",
}
_COMPLEX_MARKERS = {
    "analyze", "analyse", "architecture", "compare", "debug", "design",
    "implement", "investigate", "plan", "review", "tradeoff", "workflow",
}


@dataclass(frozen=True)
class TaskAssessment:
    tasks: tuple
    is_complex: bool
    explicit_checklist: bool

    def guidance(self):
        if not self.is_complex:
            return "Answer the current request directly and naturally."
        task_lines = "\n".join(
            f"{index}. {task}" for index, task in enumerate(self.tasks, 1)
        )
        return (
            "Treat the following as an ordered task checklist:\n"
            f"{task_lines or '1. Identify the request and its constraints.'}\n"
            "Work through the items in order and verify the response covers "
            "each one. Keep private scratch reasoning private; present the "
            "useful conclusions and necessary steps."
        )


def assess_task(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_tasks = []
    for line in lines:
        match = _BULLET.match(line)
        if match:
            bullet_tasks.append(match.group(1).strip())

    explicit = len(bullet_tasks) >= 2
    tasks = tuple(bullet_tasks) if bullet_tasks else (text.strip(),)
    words = set(re.findall(r"[a-z]+", text.lower()))
    complex_task = (
        explicit
        or len(tasks) > 1
        or bool(words & _COMPLEX_MARKERS)
        or len(text) > 700
    )
    return TaskAssessment(
        tasks=tasks,
        is_complex=complex_task,
        explicit_checklist=explicit,
    )


def find_uncovered_items(assessment, answer):
    """Return explicit checklist items with weak lexical coverage.

    This is deliberately conservative: ordinary prose is never sent through
    an automatic second model pass merely because of a keyword mismatch.
    """
    if not assessment.explicit_checklist:
        return ()
    answer_words = _meaningful_words(answer)
    missing = []
    for task in assessment.tasks:
        terms = _meaningful_words(task)
        if not terms:
            continue
        overlap = len(terms & answer_words)
        threshold = 1 if len(terms) <= 4 else 2
        if overlap < threshold:
            missing.append(task)
    return tuple(missing)


def _meaningful_words(text):
    return {
        word
        for word in re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())
        if len(word) > 2 and word not in _STOP_WORDS
    }