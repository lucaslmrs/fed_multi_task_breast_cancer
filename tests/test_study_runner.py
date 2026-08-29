import copy
import unittest
from pathlib import Path

from src.experiments.study_runner import (
    _config_signature,
    _partition_signature,
    _validate_arm_overrides,
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

    def test_manifest_declares_every_arm_and_component_comparison(self):
        base_arms = [arm for arm in self.manifest["arms"] if not arm.get("partition_variant")]
        self.assertEqual(len(base_arms), 8)
        self.assertEqual(len(self.manifest["arms"]), 10)
        comparison_ids = {row["comparison_id"] for row in self.manifest["comparisons"]}
        self.assertTrue({
            "primary_vs_local", "effect_local_budget", "effect_uniform_weighting",
            "effect_hierarchy", "effect_ce_vs_focal",
            "multitask_vs_local", "effect_client_topology",
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

    def test_holdout_size_only_affects_effective_cv1_signatures(self):
        full_rows, _, _ = build_execution_plan(
            self.manifest, [1993], self.manifest["arms"][:1], smoke=False
        )
        cv_config = full_rows[0]["config"]
        without_holdout = _config_signature(cv_config)
        changed_cv = copy.deepcopy(cv_config)
        changed_cv["training"]["holdout_test_size"] = 0.45
        self.assertEqual(without_holdout, _config_signature(changed_cv))
        self.assertNotIn("holdout_test_size", _partition_signature(cv_config)["training"])

        holdout = copy.deepcopy(cv_config)
        holdout["training"]["CV"] = 1
        holdout["training"]["holdout_test_size"] = 0.30
        changed_holdout = _config_signature(holdout)
        changed_holdout["training"]["holdout_test_size"] = 0.40
        self.assertNotEqual(holdout, changed_holdout)
        self.assertEqual(
            _partition_signature(holdout)["training"]["holdout_test_size"], 0.30
        )


if __name__ == "__main__":
    unittest.main()


class PartitionVariantTests(unittest.TestCase):
    """A topology change needs its own master, without disturbing the paired base arms."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(MANIFEST)

    def _arm(self, arm_id):
        return next(arm for arm in self.manifest["arms"] if arm["arm_id"] == arm_id)

    def test_base_arms_keep_the_historical_partition_path(self):
        rows, _, _ = build_execution_plan(self.manifest, [1993], [self._arm("primary")])
        self.assertTrue(
            rows[0]["partition_path"].endswith("seed_1993/federated_mapping.csv"),
            rows[0]["partition_path"],
        )
        self.assertEqual(rows[0]["partition_variant"], "base")

    def test_variant_arms_get_their_own_nested_partition(self):
        rows, _, _ = build_execution_plan(
            self.manifest, [1993],
            [self._arm("primary"), self._arm("multitask_primary")],
        )
        base, variant = rows[0]["partition_path"], rows[1]["partition_path"]
        self.assertNotEqual(base, variant)
        self.assertTrue(variant.endswith("seed_1993/multitask/federated_mapping.csv"), variant)
        self.assertEqual(rows[1]["partition_variant"], "multitask")

    def test_the_two_topologies_do_not_share_a_partition_signature(self):
        rows, _, _ = build_execution_plan(
            self.manifest, [1993],
            [self._arm("primary"), self._arm("multitask_primary")],
        )
        self.assertNotEqual(
            _partition_signature(rows[0]["config"]),
            _partition_signature(rows[1]["config"]),
        )

    def test_multitask_arms_declare_the_topology_only_for_busi(self):
        rows, _, _ = build_execution_plan(
            self.manifest, [1993], [self._arm("multitask_primary")]
        )
        datasets = rows[0]["config"]["datasets"]
        self.assertEqual(datasets["Curated_BUSI"]["client_topology"], "multi_task")
        self.assertEqual(datasets["ISIC_2018"]["client_topology"], "single_task")
        # Four BUSI clients keep the `steps` budget identical to the single-task arms.
        self.assertEqual(
            rows[0]["config"]["federated"]["n_clients"]["Curated_BUSI"], {"seg": 4, "cls": 4}
        )
        self.assertFalse(any(datasets["Curated_BUSI"]["oversampling"].values()))

    def test_partition_overrides_stay_forbidden_without_an_explicit_variant(self):
        overrides = {"datasets": {"Curated_BUSI": {"client_topology": "multi_task"}}}
        with self.assertRaises(ValueError):
            _validate_arm_overrides(overrides, "rogue")
        # Declaring a variant is the opt-in that makes the same override legitimate.
        _validate_arm_overrides(overrides, "declared", partition_variant="multitask")


