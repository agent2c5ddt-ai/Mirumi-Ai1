"""Non-destructive snapshots of memory and legacy development data."""

from datetime import datetime
from pathlib import Path
import shutil


_ROOT_DATA_FILES = (
    "character_memory.txt",
    "relationship_memory.txt",
    "self_developed.txt",
    "development_history.txt",
)


def create_backup(root):
    root = Path(root)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    destination = root / "memory" / "backups" / stamp
    copied = []

    memory_root = root / "memory"
    if memory_root.exists():
        for source in memory_root.rglob("*"):
            if not source.is_file() or "backups" in source.relative_to(memory_root).parts:
                continue
            target = destination / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(target)

    for filename in _ROOT_DATA_FILES:
        source = root / filename
        if not source.is_file():
            continue
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)

    if not copied:
        destination.mkdir(parents=True, exist_ok=True)
    return destination, copied