import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "threshold_sensitivity.py"


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Selection imports need optional PyTorch")
class ApplyRuleTests(unittest.TestCase):
    """Synthetic medians only; these are not measurements."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("threshold_sensitivity", SCRIPT)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_wider_allowance_admits_slower_cheaper_candidate(self):
        candidates = {"fast": {"seconds": 100.0, "joules": 10.0}, "slow": {"seconds": 110.0, "joules": 8.0}}
        self.assertEqual(self.module.apply_rule(candidates, 0.05), ("fast", "fast"))
        self.assertEqual(self.module.apply_rule(candidates, 0.20), ("fast", "slow"))

    def test_energy_ties_break_by_runtime_then_id(self):
        candidates = {"b": {"seconds": 100.0, "joules": 5.0}, "a": {"seconds": 100.0, "joules": 5.0},
                      "c": {"seconds": 101.0, "joules": 5.0}}
        self.assertEqual(self.module.apply_rule(candidates, 0.05), ("a", "a"))


if __name__ == "__main__":
    unittest.main()
