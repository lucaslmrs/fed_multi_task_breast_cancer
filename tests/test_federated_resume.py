import tempfile
import unittest
from pathlib import Path

import torch

from src.training_federated import _completed_fold_state, _preserve_partial_fold


class FederatedFoldResumeTests(unittest.TestCase):
    def _write_final_state(self, path, round_number=50):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"round": round_number}, path)

    def test_federated_fold_is_reusable_only_with_all_final_states(self):
        roster = [("client_a", "A", "seg"), ("client_b", "B", "cls")]
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            fold_dir = run_path / "fold_0"
            fold_dir.mkdir()
            (fold_dir / "aggregation_history.json").write_text("[]", encoding="utf-8")
            self._write_final_state(fold_dir / "client_client_a" / "state.pt")
            self._write_final_state(fold_dir / "client_client_b" / "state.pt", 49)
            torch.save({"round": 50, "arrays": []}, fold_dir / "global_shared.pt")

            self.assertFalse(_completed_fold_state(run_path, 0, roster, False, 50))

            self._write_final_state(fold_dir / "client_client_b" / "state.pt")
            self.assertTrue(_completed_fold_state(run_path, 0, roster, False, 50))

    def test_standalone_fold_does_not_require_a_global_state(self):
        roster = [("client_a", "A", "seg")]
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            fold_dir = run_path / "fold_0"
            fold_dir.mkdir()
            (fold_dir / "aggregation_history.json").write_text("[]", encoding="utf-8")
            self._write_final_state(fold_dir / "client_client_a" / "state.pt")

            self.assertTrue(_completed_fold_state(run_path, 0, roster, True, 50))

    def test_partial_fold_is_preserved_before_a_clean_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = Path(directory) / "fold_2"
            fold_dir.mkdir()
            (fold_dir / "partial.txt").write_text("diagnostic", encoding="utf-8")

            preserved = _preserve_partial_fold(fold_dir)

            self.assertFalse(fold_dir.exists())
            self.assertTrue((preserved / "partial.txt").exists())
            self.assertIn("fold_2.incomplete_", preserved.name)


if __name__ == "__main__":
    unittest.main()
