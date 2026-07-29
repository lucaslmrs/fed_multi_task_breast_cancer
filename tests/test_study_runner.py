import unittest
from pathlib import Path

from src.experiments.study_runner import (
    build_execution_plan,
    load_manifest,
    resolve_arm_config,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "studies/multi_dataset_balance_v1.yaml"


class StudyRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(MANIFEST)

    def test_manifest_declares_eight_arms_and_component_comparisons(self):
        self.assertEqual(len(self.manifest["arms"]), 8)
        comparison_ids = {row["comparison_id"] for row in self.manifest["comparisons"]}
        self.assertTrue({
            "primary_vs_local", "effect_local_budget", "effect_uniform_weighting",
            "effect_hierarchy", "effect_ce_vs_focal",
        }.issubset(comparison_ids))

    def test_smoke_is_separated_and_reduces_step_exposure(self):
        full_rows, full_root, _ = build_execution_plan(
            self.manifest, [1993], self.manifest["arms"][:1], smoke=False
        )
        smoke_rows, smoke_root, _ = build_execution_plan(
            self.manifest, [1993], self.manifest["arms"][:1], smoke=True
        )
        self.assertNotEqual(full_root, smoke_root)
        self.assertEqual(smoke_root, full_root / "smoke")
        smoke_config = smoke_rows[0]["config"]
        self.assertEqual(smoke_config["federated"]["local_training"]["steps_per_round"], 1)
        self.assertTrue(all(
            dataset["batch_size"] == 1 for dataset in smoke_config["datasets"].values()
        ))
        self.assertNotEqual(full_rows[0]["config_sha256"], smoke_rows[0]["config_sha256"])

    def test_focal_arm_changes_only_isic_classification_policy(self):
        base, _, _ = build_execution_plan(
            self.manifest, [1993], self.manifest["arms"][:1], smoke=False
        )
        base_config = base[0]["config"]
        focal_arm = next(arm for arm in self.manifest["arms"] if arm["arm_id"] == "ablation_focal")
        resolved = resolve_arm_config(
            base_config, self.manifest, focal_arm, 1993,
            Path(base_config["federated"]["partition_file"]),
        )
        self.assertEqual(resolved["datasets"]["ISIC_2018"]["classification_criterion"], "Focal")
        self.assertEqual(resolved["datasets"]["ISIC_2018"]["class_weighting"], "none")
        self.assertEqual(resolved["datasets"]["Curated_BUSI"]["classification_criterion"], "CE")


if __name__ == "__main__":
    unittest.main()
