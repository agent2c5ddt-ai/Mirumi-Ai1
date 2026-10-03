import json
from pathlib import Path
import tempfile
import unittest

from mirumi.app import MirumiApp
from mirumi.config import Settings
from mirumi.conversation import ConversationStore
from mirumi.memory import MemoryStore
from mirumi.model import LlamaCppCLI, ModelError


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system_prompt, prompt, max_tokens=None):
        self.calls.append((system_prompt, prompt, max_tokens))
        return self.responses.pop(0)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(
            root=self.root,
            llama_cli=self.root / "missing-llama-cli",
            model_path=self.root / "missing-model.gguf",
            context_size=2048,
            max_new_tokens=180,
            threads=2,
            timeout_seconds=10,
            max_history_chars=1800,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_assistant_response_is_logged_and_missing_checklist_item_gets_targeted_revision(self):
        model = FakeModel([
            "Added durable memory tests.",
            "Added durable memory tests and documented offline model setup.",
        ])
        output = []
        app = MirumiApp(self.settings, model=model, output=output.append)

        app._chat(
            "Please complete both:\n"
            "- Add durable memory tests\n"
            "- Document offline model setup"
        )

        turns = app.conversations.recent_messages(max_messages=10)
        self.assertEqual(["user", "assistant"], [item.role for item in turns])
        self.assertEqual(2, len(model.calls))
        self.assertIn("documented offline model setup", turns[-1].text.lower())

    def test_memory_reflection_creates_pending_candidate_only_with_verbatim_evidence(self):
        conversations = ConversationStore(self.root)
        conversations.conversations_dir.mkdir(parents=True, exist_ok=True)
        session = conversations.new_session_path()
        quote = "I like asking one clear follow-up question."
        for index in range(3):
            conversations.append_message(session, "user", f"Question {index}")
            conversations.append_message(session, "assistant", quote)
        model = FakeModel([
            json.dumps({
                "proposal": "Mirumi may prefer one clear follow-up question.",
                "evidence_quote": quote,
            })
        ])
        output = []
        app = MirumiApp(self.settings, model=model, output=output.append)

        app._reflect()

        pending = app.memories.pending()
        self.assertEqual(1, len(pending))
        self.assertEqual("assistant_reflection", pending[0].source)
        self.assertEqual(0.35, pending[0].confidence)
        self.assertEqual([], app.memories.approved())

    def test_reflection_rejects_invented_quote(self):
        conversations = ConversationStore(self.root)
        conversations.conversations_dir.mkdir(parents=True, exist_ok=True)
        session = conversations.new_session_path()
        for index in range(3):
            conversations.append_message(session, "user", f"Question {index}")
            conversations.append_message(session, "assistant", "A normal answer.")
        model = FakeModel([
            json.dumps({
                "proposal": "A permanent preference",
                "evidence_quote": "This phrase was never said.",
            })
        ])
        app = MirumiApp(self.settings, model=model, output=lambda _: None)

        app._reflect()

        self.assertEqual([], app.memories.pending())

    def test_missing_local_model_reports_clear_error(self):
        model = LlamaCppCLI(self.settings)
        with self.assertRaisesRegex(ModelError, "llama-cli was not found"):
            model.generate("system", "prompt")


if __name__ == "__main__":
    unittest.main()