"""E4 C4: the committed run config differs from the E3 winner only where E4 intends to.

The checks that need untracked artefacts (the resolved E3 config under ``runs/`` and the E4 master
under ``data/``) are skipped when those are absent, e.g. on a fresh clone.
"""
import unittest
from pathlib import Path

import yaml

from src.experiments.study_runner import (_partition_metadata_path, _partition_signature,
                                          _sha256_file, _stable_hash)
from src.federated.config import validate_federated_config

ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = ROOT / "studies" / "e4_preprocessing" / "resolution_256.yaml"
E3_WINNER = ROOT / "runs" / "studies" / "e3_loss_ablation" / "resolved" / "seed_1993" / "init_seed_dataset.yaml"
EXPECTED_DIFF = {
    "datasets.TCGA_LGG.variant",
    "datasets.SIIM_ACR.variant",
    "federated.partition_file",
    "federated.rounds",
    "experiment.study_id",
    "experiment.arm_id",
    "experiment.method_id",
    "runtime.federated.client_resources.num_gpus",
}


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(child, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, value


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class E4RunConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = _load(RUN_CONFIG)

    def test_resolution_and_run_settings(self):
        validate_federated_config(self.config)
        variants = {name: entry["variant"] for name, entry in self.config["datasets"].items()}
        self.assertEqual(variants, {"Curated_BUSI": "processed_128", "ISIC_2018": "processed_128",
                                    "TCGA_LGG": "processed_256", "SIIM_ACR": "processed_256"})
        self.assertEqual(self.config["federated"]["rounds"], 400)
        self.assertFalse(self.config["federated"]["share_stem"])
        self.assertNotIn("shared_modules", self.config["federated"])  # E4 stays trunk-only
        self.assertEqual(self.config["experiment"], {
            "study_id": "e4_preprocessing", "arm_id": "resolution_256",
            "method_id": "resolution_256", "seed": 1993, "setup": "federated"})
        self.assertEqual(self.config["runtime"]["federated"]["client_resources"]["num_gpus"], 1.0)

    @unittest.skipUnless(E3_WINNER.exists(), "resolved E3 winner config not present")
    def test_differs_from_the_e3_winner_only_where_intended(self):
        base, ours = dict(_flatten(_load(E3_WINNER))), dict(_flatten(self.config))
        changed = {key for key in set(base) | set(ours)
                   if base.get(key, "<absent>") != ours.get(key, "<absent>")}
        self.assertEqual(changed, EXPECTED_DIFF)

    def test_master_matches_its_metadata(self):
        master = Path(self.config["federated"]["partition_file"])
        metadata_path = _partition_metadata_path(master)
        if not (master.exists() and metadata_path.exists()):
            self.skipTest("E4 master not present")
        metadata = _load(metadata_path)
        signature = _partition_signature(self.config)
        self.assertEqual(metadata["partition_signature_sha256"], _stable_hash(signature))
        self.assertEqual(metadata["csv_sha256"], _sha256_file(master))
        self.assertEqual(metadata["rows"], 46555)


if __name__ == "__main__":
    unittest.main()
