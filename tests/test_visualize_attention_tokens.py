import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "visualize_attention_tokens.py"
SPEC = importlib.util.spec_from_file_location("visualize_attention_tokens", SCRIPT_PATH)
viz = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(viz)


class VisualizeAttentionTokensTest(unittest.TestCase):
    def test_region_scores_count_data_as_parent_region(self):
        record = {
            "tokens": [
                {"i": 1, "t": "fact", "s": 1.0, "r": "data_fact"},
                {"i": 2, "t": "attack", "s": 2.0, "r": "data_attack"},
                {"i": 3, "t": "other", "s": 3.0, "r": "data"},
            ],
            "top_tokens": [],
            "num_input_tokens": 3,
        }

        filtered = viz.drop_special_attention(record)

        self.assertEqual(filtered["region_scores"]["data"], 6.0)
        self.assertEqual(filtered["region_scores"]["data_fact"], 1.0)
        self.assertEqual(filtered["region_scores"]["data_attack"], 2.0)
        self.assertEqual(len(filtered["tokens"]), 3)

    def test_special_tokens_are_visible_scored_and_filterable(self):
        record = {
            "tokens": [
                {"i": 0, "t": "<|im_start|>", "s": 4.0, "r": "special"},
                {"i": 1, "t": "auth", "s": 1.0, "r": "auth"},
            ],
            "top_tokens": [],
            "num_input_tokens": 2,
        }

        filtered = viz.drop_special_attention(record)
        filters_html = viz.render_filters(filtered["tokens"])

        self.assertEqual(filtered["region_scores"]["special"], 4.0)
        self.assertEqual(len(filtered["tokens"]), 2)
        self.assertIn('value="special"', filters_html)


if __name__ == "__main__":
    unittest.main()
