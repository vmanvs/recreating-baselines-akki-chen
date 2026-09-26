import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from go1_benchmark.evaluate_ppo import _latest_checkpoint
from go1_benchmark.train_ppo import (
    DEFAULT_CONFIG,
    _environment_for_profile,
    _load_config,
)


class TrainingConfigTests(unittest.TestCase):
    def test_fixed_command_matches_paper_task(self):
        document, _ = _load_config(DEFAULT_CONFIG, "paper_budget_proxy")
        self.assertEqual(
            document["environment"]["command_m_s_rad_s"], [0.5, 0.0, 0.0]
        )

    def test_paper_proxy_preserves_reported_ppo_values(self):
        _, profile = _load_config(DEFAULT_CONFIG, "paper_budget_proxy")
        self.assertEqual(profile["num_timesteps"], 1_000_000)
        self.assertEqual(profile["learning_rate"], 5.9e-4)
        self.assertEqual(profile["clipping_epsilon"], 0.2)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown profile"):
            _load_config(DEFAULT_CONFIG, "not-a-profile")

    def test_gait_profile_is_separate_from_reference_environment(self):
        document, reference = _load_config(DEFAULT_CONFIG, "playground_reference")
        _, gait = _load_config(DEFAULT_CONFIG, "gait_aware_walk")
        self.assertNotIn("gait", _environment_for_profile(document, reference))
        effective = _environment_for_profile(document, gait)
        self.assertEqual(effective["gait"]["swing_order"], ["RL", "FL", "RR", "FR"])
        self.assertLess(effective["gait"]["swing_fraction"], 0.25)
        self.assertNotIn("gait", document["environment"])

    def test_latest_checkpoint_uses_highest_numeric_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "2").mkdir()
            (root / "10").mkdir()
            (root / "metadata").mkdir()
            self.assertEqual(_latest_checkpoint(root), root / "10")


if __name__ == "__main__":
    unittest.main()
