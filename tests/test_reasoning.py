import unittest

from mirumi.reasoning import assess_task, find_uncovered_items


class ReasoningTests(unittest.TestCase):
    def test_explicit_checklist_is_planned_and_missing_item_is_detected(self):
        plan = assess_task(
            "Please complete both:\n"
            "- Add durable memory tests\n"
            "- Document offline model setup"
        )

        self.assertTrue(plan.is_complex)
        self.assertTrue(plan.explicit_checklist)
        self.assertEqual(
            ("Document offline model setup",),
            find_uncovered_items(plan, "Added durable memory tests."),
        )
        self.assertEqual(
            (),
            find_uncovered_items(
                plan,
                "Added durable memory tests and documented offline model setup.",
            ),
        )

    def test_simple_request_does_not_trigger_a_second_pass(self):
        plan = assess_task("Can you say hello?")
        self.assertFalse(plan.is_complex)
        self.assertEqual((), find_uncovered_items(plan, "Hello!"))


if __name__ == "__main__":
    unittest.main()