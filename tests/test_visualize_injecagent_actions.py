import importlib.util
import pathlib
import sys
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "visualize_injecagent_actions.py"
SPEC = importlib.util.spec_from_file_location("visualize_injecagent_actions", SCRIPT_PATH)
viz = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = viz
SPEC.loader.exec_module(viz)


class VisualizeInjecAgentActionsTest(unittest.TestCase):
    def _row(self):
        return {
            "case_id": 7,
            "eval": "succ",
            "attack_type": "Financial Harm",
            "attacker_tools": ["BankManagerPayBill"],
            "user_tool": "TwitterManagerReadTweet",
            "action_kind": "tool_action",
            "action_text": "Action: BankManagerPayBill\nAction Input: {'amount': 500}",
            "action_tool": "BankManagerPayBill",
            "valid_for_stats": True,
            "values": {
                "empty": -5.0,
                "auth": -4.0,
                "data_fact": -4.5,
                "data_attack": -2.0,
                "auth+data_fact+data_attack": -1.0,
            },
            "players": ["auth", "data_fact", "data_attack"],
            "phi_auth": 0.5,
            "phi_data_fact": 0.25,
            "phi_data_attack": 2.0,
            "shapley_attack_dominant": True,
        }

    def test_attack_margin(self):
        self.assertAlmostEqual(viz.attack_margin(self._row()), 1.25)

    def test_merge_attention_by_case_id(self):
        row = viz.merge_attention(
            self._row(),
            {
                "7": {
                    "case_id": 7,
                    "attention_shift_attack": True,
                    "region_scores": {"auth": 0.1, "data_fact": 0.2, "data_attack": 0.8},
                    "top_tokens": [{"i": 1, "t": "Bank", "s": 0.8, "r": "data_attack"}],
                }
            },
        )
        self.assertTrue(row["attention_shift_attack"])
        self.assertIn("attention", row)

    def test_render_html_contains_core_sections(self):
        row = viz.merge_attention(
            self._row(),
            {
                "7": {
                    "case_id": 7,
                    "attention_shift_attack": True,
                    "region_scores": {"auth": 0.1, "data_fact": 0.2, "data_attack": 0.8},
                    "top_tokens": [{"i": 1, "t": "Bank", "s": 0.8, "r": "data_attack"}],
                }
            },
        )
        html = viz.render_html([row], {"shapley": "s.jsonl", "attention": "a.jsonl"}, "Demo")
        self.assertIn("Explained Action", html)
        self.assertNotIn("Coalition Values", html)
        self.assertIn("Tokens", html)
        self.assertIn('data-filter="auth"', html)
        self.assertIn("ATTACK MARGIN", html)


if __name__ == "__main__":
    unittest.main()
