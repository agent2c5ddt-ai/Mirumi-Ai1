from pathlib import Path
import tempfile
import unittest

from mirumi.backup import create_backup
from mirumi.memory import MemoryStore


class BackupTests(unittest.TestCase):
    def test_snapshot_copies_memory_without_changing_or_recursing_into_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversation = root / "memory" / "conversations" / "session.jsonl"
            conversation.parent.mkdir(parents=True)
            conversation.write_text('{"role":"user","text":"keep"}\n', encoding="utf-8")
            legacy = root / "self_developed.txt"
            legacy.write_text("Original trait\n", encoding="utf-8")
            (root / "memory" / "backups" / "prior").mkdir(parents=True)
            (root / "memory" / "backups" / "prior" / "old.txt").write_text(
                "previous backup", encoding="utf-8"
            )

            backup, copied = create_backup(root)

            self.assertEqual(
                conversation.read_text(encoding="utf-8"),
                (backup / "memory" / "conversations" / "session.jsonl").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertEqual("Original trait\n", legacy.read_text(encoding="utf-8"))
            self.assertTrue((backup / "self_developed.txt").is_file())
            self.assertFalse((backup / "memory" / "backups").exists())
            self.assertEqual(2, len(copied))

    def test_duplicate_legacy_development_is_one_unverified_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "self_developed.txt").write_text(
                "Trait: Mirumi likes spending time in nature and quiet moments.\n",
                encoding="utf-8",
            )
            (root / "development_history.txt").write_text(
                "Development 001:\nMirumi likes spending time in nature and quiet moments.\n",
                encoding="utf-8",
            )
            development_dir = root / "memory" / "self_development"
            development_dir.mkdir(parents=True)
            (development_dir / "001.txt").write_text(
                "Development:\nMirumi likes spending time in nature and quiet moments.\n",
                encoding="utf-8",
            )

            candidates = MemoryStore(root).pending()

            self.assertEqual(1, len(candidates))
            self.assertEqual("legacy_unverified", candidates[0].source)
            self.assertEqual(3, len(candidates[0].legacy_sources))


if __name__ == "__main__":
    unittest.main()