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
MANIFEST = ROOT / "studies/example_multi_dataset.yaml"


class StudyRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(MANIFEST)

    def test_manifest_declares_every_arm_and_component_comparison(self):
        base_arms = [arm for arm in self.manifest["arms"] if not arm.get("partition_variant")]
        self.assertEqual(len(base_arms), 3)
        self.assertEqual(len(self.manifest["arms"]), 5)
        comparison_ids = {row["comparison_id"] for row in self.manifest["comparisons"]}
        self.assertTrue({
            "primary_vs_local", "effect_flat_aggregation",
            "single_task_vs_local", "effect_client_topology",
        }.issubset(comparison_ids))

    def test_manifest_pins_the_protocol_instead_of_inheriting_it(self):
        rows, _, _ = build_execution_plan(
            self.manifest, [1993], self.manifest["arms"][:1], smoke=False
        )
        config = rows[0]["config"]
        self.assertEqual(config["training"]["CV"], 1)
        self.assertEqual(config["training"]["holdout_test_size"], 0.30)
        self.assertEqual(config["training"]["precision"], "bf16")
        self.assertTrue(config["training"]["cuda_benchmark"])
        self.assertEqual(config["datasets"]["Curated_BUSI"]["client_topology"], "multi_task")
        self.assertEqual(config["datasets"]["ISIC_2018"]["client_topology"], "single_task")
        self.assertFalse(any(config["datasets"]["Curated_BUSI"]["oversampling"].values()))
        self.assertEqual(config["federated"]["local_training"]["mode"], "steps")
        self.assertEqual(config["federated"]["aggregation"]["mode"], "hierarchical")
        # Deep merge keeps the base task/dataset weights the overrides do not mention.
        self.assertIn("task_weights", config["federated"]["aggregation"])

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

    def test_flat_arm_changes_only_the_aggregation_policy(self):
        base, _, _ = build_execution_plan(
            self.manifest, [1993], self.manifest["arms"][:1], smoke=False
        )
        base_config = base[0]["config"]
        flat_arm = next(arm for arm in self.manifest["arms"] if arm["arm_id"] == "ablation_flat")
        resolved = resolve_arm_config(
            base_config, self.manifest, flat_arm, 1993,
            Path(base_config["federated"]["partition_file"]),
        )
        self.assertEqual(resolved["federated"]["aggregation"]["mode"], "flat")
        self.assertEqual(resolved["federated"]["aggregation"]["client_weighting"], "num_examples")
        self.assertEqual(resolved["datasets"], base_config["datasets"])
        self.assertEqual(resolved["federated"]["local_training"],
                         base_config["federated"]["local_training"])
        self.assertEqual(_partition_signature(resolved), _partition_signature(base_config))

    def test_holdout_size_only_affects_effective_cv1_signatures(self):
        full_rows, _, _ = build_execution_plan(
            self.manifest, [1993], self.manifest["arms"][:1], smoke=False
        )
        cv_config = copy.deepcopy(full_rows[0]["config"])
        cv_config["training"]["CV"] = 5
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


class PartitionVariantTests(unittest.TestCase):
    """A topology change needs its own master, without disturbing the paired base arms."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(MANIFEST)

    def _arm(self, arm_id):
        return next(arm for arm in self.manifest["arms"] if arm["arm_id"] == arm_id)

    def test_base_arms_share_the_seed_partition_path(self):
        rows, _, _ = build_execution_plan(
            self.manifest, [1993],
            [self._arm("primary"), self._arm("local_only"), self._arm("ablation_flat")],
        )
        paths = {row["partition_path"] for row in rows}
        self.assertEqual(len(paths), 1)
        self.assertTrue(
            rows[0]["partition_path"].endswith("seed_1993/federated_mapping.csv"),
            rows[0]["partition_path"],
        )
        self.assertTrue(all(row["partition_variant"] == "base" for row in rows))

    def test_variant_arms_get_their_own_nested_partition(self):
        rows, _, _ = build_execution_plan(
            self.manifest, [1993],
            [self._arm("primary"), self._arm("single_task_primary"), self._arm("single_task_local")],
        )
        base, variant, paired = (row["partition_path"] for row in rows)
        self.assertNotEqual(base, variant)
        self.assertEqual(variant, paired)
        self.assertTrue(variant.endswith("seed_1993/single_task/federated_mapping.csv"), variant)
        self.assertEqual(rows[1]["partition_variant"], "single_task")

    def test_the_two_topologies_do_not_share_a_partition_signature(self):
        rows, _, _ = build_execution_plan(
            self.manifest, [1993],
            [self._arm("primary"), self._arm("single_task_primary")],
        )
        self.assertNotEqual(
            _partition_signature(rows[0]["config"]),
            _partition_signature(rows[1]["config"]),
        )

    def test_single_task_arms_change_only_the_busi_topology(self):
        rows, _, _ = build_execution_plan(
            self.manifest, [1993], [self._arm("primary"), self._arm("single_task_primary")]
        )
        base, variant = rows[0]["config"], rows[1]["config"]
        self.assertEqual(base["datasets"]["Curated_BUSI"]["client_topology"], "multi_task")
        self.assertEqual(variant["datasets"]["Curated_BUSI"]["client_topology"], "single_task")
        self.assertEqual(variant["datasets"]["ISIC_2018"], base["datasets"]["ISIC_2018"])
        self.assertEqual(variant["federated"]["n_clients"], base["federated"]["n_clients"])
        # Single-task clients may oversample classification; multi-task clients refuse it.
        self.assertTrue(variant["datasets"]["Curated_BUSI"]["oversampling"]["cls"])
        self.assertFalse(base["datasets"]["Curated_BUSI"]["oversampling"]["cls"])

    def test_partition_overrides_stay_forbidden_without_an_explicit_variant(self):
        overrides = {"datasets": {"Curated_BUSI": {"client_topology": "single_task"}}}
        with self.assertRaises(ValueError):
            _validate_arm_overrides(overrides, "rogue")
        # Declaring a variant is the opt-in that makes the same override legitimate.
        _validate_arm_overrides(overrides, "declared", partition_variant="single_task")


if __name__ == "__main__":
    unittest.main()
