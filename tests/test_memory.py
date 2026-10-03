import json
from pathlib import Path
import tempfile
import unittest

from mirumi.memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = MemoryStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_only_explicit_or_approved_memories_are_retrieved(self):
        approved = self.store.add(
            "The user prefers quiet study sessions.",
            category="preference",
            source="user_explicit",
            status="approved",
        )
        candidate = self.store.add(
            "Mirumi may prefer quiet study sessions.",
            category="learned_behavior",
            source="assistant_reflection",
            confidence=0.35,
            status="pending",
            evidence="I like quiet study sessions.",
        )

        self.assertEqual([approved], self.store.retrieve("quiet study", limit=5))
        self.assertTrue(self.store.approve(candidate.id))
        self.assertEqual(
            {approved.id, candidate.id},
            {item.id for item in self.store.retrieve("quiet study", limit=5)},
        )

    def test_non_user_source_cannot_write_directly_as_approved(self):
        with self.assertRaises(ValueError):
            self.store.add(
                "A generated claim",
                source="assistant",
                status="approved",
            )
        self.assertEqual([], self.store.approved())

    def test_legacy_development_is_pending_and_source_is_unchanged(self):
        source = self.root / "self_developed.txt"
        original = (
            "SELF-DEVELOPED CHARACTER TRAITS\n\n"
            "Entry 001\n"
            "Trait: Mirumi expressed an apparent liking for quiet moments in nature.\n"
        )
        source.write_text(original, encoding="utf-8")

        candidates = self.store.pending()

        self.assertEqual(1, len(candidates))
        self.assertEqual("legacy_unverified", candidates[0].source)
        self.assertIn("self_developed.txt", candidates[0].legacy_sources[0])
        self.assertEqual(original, source.read_text(encoding="utf-8"))
        self.assertEqual([], self.store.retrieve("quiet nature", limit=5))

        self.assertTrue(self.store.approve(candidates[0].id))
        self.assertEqual(1, len(self.store.retrieve("quiet nature", limit=5)))

    def test_malformed_jsonl_is_skipped_without_losing_valid_records(self):
        self.store.records_path.parent.mkdir(parents=True, exist_ok=True)
        valid = {
            "id": "test-record",
            "content": "The user likes drawing.",
            "category": "preference",
            "source": "user_explicit",
            "confidence": 1.0,
            "created_at": "2026-10-03T10:00:00+00:00",
            "status": "approved",
            "evidence": "",
        }
        self.store.records_path.write_text(
            "{not json}\n" + json.dumps(valid) + "\n", encoding="utf-8"
        )

        self.assertEqual(["test-record"], [item.id for item in self.store.approved()])
        self.assertTrue(any("malformed memory line" in item for item in self.store.warnings))

    def test_forget_is_an_append_only_decision_not_deletion(self):
        record = self.store.add("The user likes hiking.", category="preference")
        self.assertTrue(self.store.forget(record.id))

        self.assertEqual([], self.store.approved())
        self.assertTrue(self.store.records_path.exists())
        self.assertEqual(1, len(self.store.records_path.read_text(encoding="utf-8").splitlines()))
        self.assertEqual("forgotten", json.loads(
            self.store.decisions_path.read_text(encoding="utf-8").splitlines()[-1]
        )["action"])


if __name__ == "__main__":
    unittest.main()