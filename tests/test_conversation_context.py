import json
from pathlib import Path
import tempfile
import unittest

from mirumi.character import CharacterProfile
from mirumi.config import Settings
from mirumi.context import ContextBuilder
from mirumi.conversation import ConversationStore, Turn


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ConversationStore(self.root)
        self.store.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.session = self.store.new_session_path()

    def tearDown(self):
        self.temp.cleanup()

    def test_both_sides_of_conversation_are_persisted_and_legacy_role_loads(self):
        self.store.append_message(self.session, "user", "Hello")
        self.store.append_message(self.session, "assistant", "Hi there")
        with self.session.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "time": "2026-10-03T12:00:00+05:30",
                "role": "mirumi",
                "text": "Older response",
            }) + "\n")

        turns = self.store.recent_messages(max_messages=10)

        self.assertEqual(["user", "assistant", "assistant"], [item.role for item in turns])
        self.assertEqual(["Hello", "Hi there", "Older response"], [item.text for item in turns])

    def test_corrupt_lines_are_skipped_not_fatal(self):
        self.session.write_text(
            "invalid json\n"
            + json.dumps({
                "time": "2026-10-03T12:00:00+05:30",
                "role": "user",
                "text": "survives",
            })
            + "\n",
            encoding="utf-8",
        )

        turns = self.store.recent_messages()

        self.assertEqual(["survives"], [item.text for item in turns])
        self.assertTrue(any("malformed conversation line" in item for item in self.store.warnings))

    def test_context_is_bounded_and_retains_latest_user_request(self):
        settings = Settings(
            root=self.root,
            llama_cli=self.root / "llama-cli",
            model_path=self.root / "model.gguf",
            context_size=2048,
            max_new_tokens=180,
            threads=2,
            timeout_seconds=10,
            max_history_chars=1800,
        )
        profile = CharacterProfile.load(self.root)
        history = [Turn("2026-10-03T11:00:00+05:30", "user", "old " * 500)]
        system, prompt = ContextBuilder(settings).build(
            query=("Important request at the beginning. " + "details " * 1400
                   + " Keep the final constraint."),
            profile=profile,
            memories=[],
            history=history,
            summary=None,
            task_guidance="Answer directly.",
        )
        budget = max(
            1200,
            (settings.context_size - settings.max_new_tokens - 180) * 3,
        )

        self.assertLessEqual(len(system) + len(prompt), budget)
        self.assertIn("CURRENT USER MESSAGE", prompt)
        self.assertIn("Important request at the beginning.", prompt)
        self.assertIn("Keep the final constraint.", prompt)
        self.assertIn("omitted to fit", prompt)

    def test_minimum_context_drops_summary_before_exceeding_budget(self):
        (self.root / "identity.txt").write_text(
            "Identity: " + ("Mirumi is gentle and attentive. " * 100),
            encoding="utf-8",
        )
        settings = Settings(
            root=self.root,
            llama_cli=self.root / "llama-cli",
            model_path=self.root / "model.gguf",
            context_size=1024,
            max_new_tokens=512,
            threads=1,
            timeout_seconds=10,
            max_history_chars=300,
        )
        system, prompt = ContextBuilder(settings).build(
            query="Keep a concise answer.",
            profile=CharacterProfile.load(self.root),
            memories=[],
            history=[],
            summary={"summary": "older context " * 400},
            task_guidance="Answer directly.",
        )
        budget = max(
            1200,
            (settings.context_size - settings.max_new_tokens - 180) * 3,
        )

        self.assertLessEqual(len(system) + len(prompt), budget)
        self.assertNotIn("PRIOR CONVERSATION SUMMARY", prompt)
        self.assertIn("Keep a concise answer.", prompt)


if __name__ == "__main__":
    unittest.main()