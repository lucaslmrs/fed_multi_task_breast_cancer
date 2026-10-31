"""E6: ``federated.shared_modules`` selects the modules FedAvg averages.

Absent (or null), the legacy ``share_stem`` prefix rule must reproduce today's shared keys byte for
byte on every architecture. Present, an explicit list of top-level modules is matched exactly and
validated twice before Flower starts: as config (``shared_modules``) and against the built model
(``validate_sharing``). The key stays out of ``src/config.yaml``: every study hashes its config.

The M1 gradient probes (E6-C2) must survive shared blocks a client's objective never reaches and
shared buffers, and keep every probe's block set identical so ``m1_pairs`` can pair them.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml
from flwr.common import parameters_to_ndarrays

from scripts.smoke_federated import _verify_m1_coverage
from src.federated import conflict_diagnostics as cd
from src.federated.config import active_datasets, shared_modules, validate_federated_config
from src.federated.local_trainer import local_objective
from src.federated.model_split import (get_personalized_state, get_shared_state, is_shared,
                                       resolved_shared_modules, set_personalized_state,
                                       set_shared_state, shared_keys, validate_sharing)
from src.models.multitask.MTnnUNet import MTnnUNet
from src.training_federated import _initial_shared_parameters, _log_sharing
from src.utils.experiment_init import (init_criterion_classification, init_criterion_segmentation,
                                       init_multitask_model)
from src.utils.supervision import LOG_VARS_ATTR, attach_task_uncertainty, resolve_task_lambdas
from src.utils.training_runtime import PrecisionPolicy

ROOT = Path(__file__).resolve().parents[1]
WIDE = ("encoder2", "encoder3", "encoder4", "encoder5", "bottleneck",
        "decoder5", "decoder4", "decoder3", "decoder2", "decoder1",
        "upsample5", "upsample4", "upsample3", "upsample2", "upsample1",
        "process_encoder_5", "process_decoder_5")
PERSONAL = {"encoder1", "output1", "output2", "output3", "output4", "classifier"}
MTNNUNET_TOTAL = 15_819_799
TRUNK_VALUES = 4_663_296
WIDE_VALUES = 10_135_840


def _legacy_keys(model, share_stem):
    """The pre-E6 rule written out: state_dict-key PREFIX match on the trunk (+ the stem)."""
    prefixes = ("encoder2", "encoder3", "encoder4", "encoder5", "bottleneck")
    if share_stem:
        prefixes = ("encoder1",) + prefixes
    return [key for key in model.state_dict() if key.startswith(prefixes)]


def _mtnnunet(channels=1, n_classes=3, kendall=False, seed=0):
    torch.manual_seed(seed)
    model = MTnnUNet(sequences=channels, regions=1, n_classes=n_classes)
    if kendall:
        attach_task_uncertainty(model, ("seg", "cls"))
    return model


def _top(keys):
    return {str(key).split(".", 1)[0] for key in keys}


def _numel(model, keys):
    state = model.state_dict()
    return sum(state[key].numel() for key in keys)


def _two_modality_config(**federated):
    """Smallest config ``validate_federated_config`` accepts: a 1ch and a 3ch dataset."""
    base = {
        "datasets": ["gray", "rgb"],
        "share_stem": False,
        "aggregation": {"mode": "hierarchical", "task_weights": {"seg": 1, "cls": 1},
                        "dataset_weights": {"gray": 1, "rgb": 1}},
    }
    base.update(federated)
    return {
        "model": {"sequences": 1},
        "data": {"dataset": "gray", "classes": ["a"], "augmentation": {}},
        "datasets": {"gray": {"channels": 1, "classes": ["a"]},
                     "rgb": {"channels": 3, "classes": ["b"]}},
        "federated": base,
    }


def _repo_config(**federated):
    config = yaml.safe_load((ROOT / "src" / "config.yaml").read_text(encoding="utf-8"))
    config["federated"].update(federated)
    return config


class ResolverTests(unittest.TestCase):
    def test_absent_or_null_key_returns_the_legacy_bool(self):
        self.assertIs(shared_modules({"federated": {}}), True)
        self.assertIs(shared_modules({"federated": {"share_stem": False}}), False)
        self.assertIs(shared_modules({"federated": {"share_stem": True, "shared_modules": None}}),
                      True)

    def test_explicit_list_returns_a_tuple(self):
        self.assertEqual(shared_modules({"federated": {"share_stem": False,
                                                       "shared_modules": list(WIDE)}}), WIDE)
        with_stem = ["encoder1", *WIDE]
        self.assertEqual(shared_modules({"federated": {"share_stem": True,
                                                       "shared_modules": with_stem}}),
                         tuple(with_stem))

    def test_valid_explicit_list_passes_config_validation(self):
        validate_federated_config(_two_modality_config(shared_modules=list(WIDE)))


class ValidationTests(unittest.TestCase):
    def test_config_rejects_malformed_lists(self):
        for value in ([], "encoder2", {"encoder2": True}, ["encoder2", "encoder2"],
                      ["encoder2.conv1"], [LOG_VARS_ATTR], [""], [" encoder2"], [3]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "shared_modules"):
                validate_federated_config(_two_modality_config(shared_modules=value))

    def test_encoder1_must_agree_with_share_stem(self):
        with self.assertRaisesRegex(ValueError, "share_stem"):
            shared_modules({"federated": {"share_stem": False,
                                          "shared_modules": ["encoder1", *WIDE]}})
        # share_stem defaults to true when absent, so a list without encoder1 needs it set.
        with self.assertRaisesRegex(ValueError, "share_stem"):
            shared_modules({"federated": {"shared_modules": list(WIDE)}})
        with self.assertRaisesRegex(ValueError, "share_stem"):
            validate_federated_config(_two_modality_config(share_stem=True,
                                                           shared_modules=list(WIDE)))

    def test_model_rejects_unknown_and_tensorless_modules(self):
        model = _mtnnunet()
        with self.assertRaisesRegex(ValueError, "decoder6"):
            validate_sharing(model, (*WIDE, "decoder6"))
        # ``downsample`` is a real top-level module of MTnnUNet, but owns no tensor.
        with self.assertRaisesRegex(ValueError, "downsample"):
            validate_sharing(model, (*WIDE, "downsample"))

    def test_model_rejects_non_float_shared_tensors(self):
        model = _mtnnunet()
        model.encoder2.register_buffer("step_count", torch.zeros((), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "floating point"):
            validate_sharing(model, WIDE)
        with self.assertRaisesRegex(ValueError, "floating point"):
            validate_sharing(model, False)

    def test_string_spec_is_a_type_error(self):
        with self.assertRaises(TypeError):
            shared_keys(_mtnnunet(), "encoder2")


class LegacyEquivalenceTests(unittest.TestCase):
    def test_mtnnunet_keys_match_the_prefix_rule(self):
        for channels, n_classes in ((1, 3), (3, 7)):
            for share_stem in (True, False):
                for kendall in (False, True):
                    with self.subTest(channels=channels, share_stem=share_stem, kendall=kendall):
                        model = _mtnnunet(channels, n_classes, kendall)
                        sharing = shared_modules({"federated": {"share_stem": share_stem}})
                        keys = shared_keys(model, sharing)
                        self.assertEqual(keys, _legacy_keys(model, share_stem))
                        self.assertEqual(validate_sharing(model, sharing), keys)
                        personalized = get_personalized_state(model, sharing)
                        self.assertEqual(set(personalized) | set(keys), set(model.state_dict()))
                        self.assertFalse(set(personalized) & set(keys))
                        if not share_stem:
                            self.assertEqual(len(keys), 10)
                            self.assertEqual(_numel(model, keys), TRUNK_VALUES)
                        if kendall:
                            self.assertIn(LOG_VARS_ATTR, _top(personalized))

    def test_multi_btsunet_keeps_bottleneck2_under_the_prefix_rule(self):
        model = init_multitask_model("Multi_BTSUNet", sequences=1, regions=1, n_classes=3,
                                     width=24, deep_supervision=True)
        for share_stem, expected in ((True, 11), (False, 9)):
            keys = shared_keys(model, share_stem)
            self.assertEqual(keys, _legacy_keys(model, share_stem))
            self.assertEqual(len(keys), expected)
            self.assertIn("bottleneck2", _top(keys))
        # An exact-name list would drop it: why the legacy path keeps ``startswith``.
        exact = shared_keys(model, ("encoder2", "encoder3", "encoder4", "bottleneck"))
        self.assertNotIn("bottleneck2", _top(exact))

    def test_mtunetplusplus_shares_nothing_and_now_fails_loudly(self):
        model = init_multitask_model("MTUNetPlusPlus", sequences=1, regions=1, n_classes=3,
                                     width=24, deep_supervision=True)
        for share_stem in (True, False):
            self.assertEqual(shared_keys(model, share_stem), [])
            with self.assertRaisesRegex(ValueError, "no shared tensor"):
                validate_sharing(model, share_stem)


class ExactMatchTests(unittest.TestCase):
    def test_a_name_does_not_capture_a_longer_sibling(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = torch.nn.Linear(2, 2)
                self.enc2 = torch.nn.Linear(2, 2)
                self.head = torch.nn.Linear(2, 1)

        model = Tiny()
        self.assertEqual(shared_keys(model, ("enc",)), ["enc.weight", "enc.bias"])
        self.assertFalse(is_shared("enc2.weight", ("enc",)))
        self.assertEqual(set(get_personalized_state(model, ("enc",))),
                         {"enc2.weight", "enc2.bias", "head.weight", "head.bias"})


class WideSetTests(unittest.TestCase):
    def test_size_and_personalized_remainder(self):
        gray, rgb = _mtnnunet(1, 3), _mtnnunet(3, 7)
        keys = validate_sharing(gray, WIDE)
        self.assertEqual(len(keys), 32)
        self.assertEqual(_numel(gray, keys), WIDE_VALUES)
        self.assertEqual(sum(v.numel() for v in gray.state_dict().values()), MTNNUNET_TOTAL)
        self.assertEqual(keys, validate_sharing(rgb, WIDE))
        self.assertEqual([a.shape for a in get_shared_state(gray, WIDE)],
                         [a.shape for a in get_shared_state(rgb, WIDE)])
        self.assertEqual(resolved_shared_modules(gray, WIDE), list(WIDE))
        for model in (gray, rgb):
            personalized = get_personalized_state(model, WIDE)
            self.assertEqual(_top(personalized), PERSONAL)
            self.assertEqual(set(personalized) | set(keys), set(model.state_dict()))
            self.assertFalse(set(personalized) & set(keys))

    def test_kendall_log_vars_stay_personalized(self):
        model = _mtnnunet(kendall=True)
        self.assertEqual(shared_keys(model, WIDE), shared_keys(_mtnnunet(), WIDE))
        self.assertEqual(_top(get_personalized_state(model, WIDE)), PERSONAL | {LOG_VARS_ATTR})

    def test_round_trip_and_personalized_guard(self):
        model = _mtnnunet()
        arrays = [array + 1.0 for array in get_shared_state(model, WIDE)]
        set_shared_state(model, arrays, sharing=WIDE)
        for expected, actual in zip(arrays, get_shared_state(model, WIDE)):
            self.assertTrue((expected == actual).all())
        shared_decoder_key = next(k for k in shared_keys(model, WIDE) if k.startswith("decoder1."))
        with self.assertRaisesRegex(ValueError, "shared/unknown"):
            set_personalized_state(model, {shared_decoder_key: model.state_dict()[shared_decoder_key]},
                                   WIDE)

    def test_sharing_log_line(self):
        with self.assertLogs(level="INFO") as logs:
            _log_sharing(_mtnnunet(), WIDE)
        self.assertIn(f"[sharing] modules={','.join(WIDE)} tensors=32 params={WIDE_VALUES} "
                      f"(64.07% of {MTNNUNET_TOTAL})", logs.output[0])


class InitialSharedTests(unittest.TestCase):
    def test_four_dataset_wide_config_builds_32_identical_arrays(self):
        config = _repo_config(shared_modules=list(WIDE))
        validate_federated_config(config)
        datasets = active_datasets(config)
        self.assertGreater(len({config["datasets"][d]["channels"] for d in datasets}), 1)
        arrays = parameters_to_ndarrays(_initial_shared_parameters(config, datasets, "cpu", 0))
        self.assertEqual(len(arrays), 32)
        self.assertEqual(sum(array.size for array in arrays), WIDE_VALUES)

    def test_absent_key_keeps_the_trunk(self):
        config = _repo_config()
        arrays = parameters_to_ndarrays(
            _initial_shared_parameters(config, active_datasets(config), "cpu", 0))
        self.assertEqual(len(arrays), 10)
        self.assertEqual(sum(array.size for array in arrays), TRUNK_VALUES)

    def test_empty_shared_set_fails_before_flower(self):
        config = _repo_config()
        config["model"]["architecture"] = "MTUNetPlusPlus"
        with self.assertRaisesRegex(ValueError, "no shared tensor"):
            _initial_shared_parameters(config, active_datasets(config), "cpu", 0)


CLS_ONLY_UNREACHED = {"decoder1", "decoder2", "decoder3", "decoder4",
                      "upsample1", "upsample2", "upsample3", "upsample4"}
SEG_ONLY_UNREACHED = {"process_encoder_5", "process_decoder_5"}
MIXED = {"ISIC_2018_cls_0": ("ISIC_2018", ("cls",)), "ISIC_2018_seg_0": ("ISIC_2018", ("seg",)),
         "Curated_BUSI_mt_0": ("Curated_BUSI", ("seg", "cls"))}


def _probe_config(rounds=1):
    return {
        "federated": {"rounds": rounds, "standalone": False},
        "runtime": {"diagnostics": {"gradient_conflict": {
            "enabled": True, "every_n_rounds": 1, "probe_batches": 4, "sketch_dim": 4096,
            "lookahead": {"enabled": False, "val_batches": 2, "train_batches": 2},
        }}},
    }


def _batches(seed, n=4, size=64):
    generator = torch.Generator().manual_seed(seed)
    return [{"image": torch.rand(2, 1, size, size, generator=generator),
             "mask": (torch.rand(2, 1, size, size, generator=generator) > 0.7).float(),
             "label": torch.randint(0, 3, (2,), generator=generator),
             "has_mask": torch.ones(2, dtype=torch.bool),
             "has_label": torch.ones(2, dtype=torch.bool)} for _ in range(n)]


def _probe_client(directory, client_id, tasks, sharing, seed=0, model=None):
    """The attributes ``collect_fit_probes`` reads from a ``FederatedClient``."""
    torch.manual_seed(seed)
    model = model if model is not None else MTnnUNet(sequences=1, regions=1, n_classes=3)
    state_dir = Path(directory) / "fold_0" / f"client_{client_id}"
    state_dir.mkdir(parents=True, exist_ok=True)
    tasks = tuple(tasks)
    lambdas = {"seg": 0.8, "cls": 0.2}
    return SimpleNamespace(
        client_id=client_id, model=model, sharing=sharing, share_stem=False, tasks=tasks,
        task="+".join(tasks), task_lambdas={task: lambdas[task] for task in tasks},
        precision=PrecisionPolicy("fp32", "cpu"), device="cpu", num_classes=3,
        inversely_weighted=True, seg_criterion=init_criterion_segmentation("DiceBCE"),
        cls_criterion=init_criterion_classification(n_classes=3, device="cpu"),
        config=_probe_config(), state_dir=state_dir, train_loader=_batches(seed),
        val_loader=_batches(seed + 100, n=2),
    )


def _pre_fix_halves(client, batches):
    """``trunk_gradient_halves`` as it was before E6-C2, verbatim apart from the spec name."""
    model = client.model
    keys = shared_keys(model, client.sharing)
    params = dict(model.named_parameters())
    lambdas = resolve_task_lambdas(client.tasks, client.task_lambdas)
    half = len(batches) // 2
    sums = [{key: torch.zeros_like(params[key], dtype=torch.float32) for key in keys}
            for _ in range(2)]
    model.train()
    for position, data in enumerate(batches):
        model.zero_grad(set_to_none=True)
        inputs = client.precision.move(data["image"])
        loss, *_ = local_objective(
            model, data, inputs, client.tasks, lambdas, client.device, client.num_classes,
            client.seg_criterion, client.cls_criterion, client.inversely_weighted,
            client.precision,
        )
        client.precision.ensure_finite(loss, "gradient-conflict probe")
        loss.backward()
        target = sums[0 if position < half else 1]
        for key in keys:
            target[key] += params[key].grad.detach().float()
    model.zero_grad(set_to_none=True)
    counts = (half, len(batches) - half)
    return keys, [{key: value / count for key, value in group.items()}
                  for group, count in zip(sums, counts)]


def _block_exact(directory, client_id, server_round=1):
    path = cd.round_dir(Path(directory) / "fold_0", server_round) / f"{client_id}_m1.npz"
    with np.load(path) as stored:
        return {str(block): stored[f"{block}::exact"] for block in stored["blocks"]}


def _mixed_round(directory):
    """One sampled round probed by a cls-only, a seg-only and a multi-task client (wide set)."""
    for seed, (client_id, (dataset, tasks)) in enumerate(MIXED.items()):
        client = _probe_client(directory, client_id, tasks, WIDE, seed=seed)
        (client.state_dir / "metadata.yaml").write_text(yaml.safe_dump(
            {"client_id": client_id, "dataset": dataset, "task": client.task,
             "tasks": list(tasks)}), encoding="utf-8")
        cd.collect_fit_probes(client, 1)
    return Path(directory) / "fold_0"


class M1ProbeTests(unittest.TestCase):
    def test_trunk_only_is_bitwise_equal_to_the_pre_fix_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            for tasks in (("seg", "cls"), ("seg",), ("cls",)):
                with self.subTest(tasks=tasks):
                    client = _probe_client(directory, "c", tasks, False)
                    torch.manual_seed(7)
                    old_keys, old = _pre_fix_halves(client, client.train_loader)
                    torch.manual_seed(7)
                    new_keys, new = cd.trunk_gradient_halves(client, client.train_loader)
                    self.assertEqual(new_keys, old_keys)
                    self.assertEqual(len(new_keys), 10)
                    for old_half, new_half in zip(old, new):
                        for key in old_keys:
                            self.assertTrue(torch.equal(old_half[key], new_half[key]), key)

    def test_single_task_clients_probe_unreached_blocks_as_zeros(self):
        # Before E6-C2 both raised AttributeError on ``None.detach()`` inside observational().
        with tempfile.TemporaryDirectory() as directory:
            for client_id, tasks, unreached in (("ISIC_2018_cls_0", ("cls",), CLS_ONLY_UNREACHED),
                                                ("ISIC_2018_seg_0", ("seg",), SEG_ONLY_UNREACHED)):
                with self.subTest(tasks=tasks):
                    cd.collect_fit_probes(_probe_client(directory, client_id, tasks, WIDE), 1)
                    exact = _block_exact(directory, client_id)
                    self.assertEqual(list(exact), list(WIDE))
                    self.assertEqual({block for block, values in exact.items()
                                      if values[3] == 0}, unreached)
                    for block in unreached:
                        self.assertTrue((exact[block] == 0).all(), block)

    def test_shared_buffer_is_excluded_without_key_error(self):
        model = MTnnUNet(sequences=1, regions=1, n_classes=3)
        model.encoder2.register_buffer("running_stat", torch.ones(4))
        with tempfile.TemporaryDirectory() as directory:
            client = _probe_client(directory, "Curated_BUSI_mt_0", ("seg", "cls"), WIDE,
                                   model=model)
            self.assertIn("encoder2.running_stat", shared_keys(model, WIDE))
            keys, _ = cd.trunk_gradient_halves(client, client.train_loader)
            self.assertNotIn("encoder2.running_stat", keys)
            self.assertEqual(len(keys), 32)
            cd.collect_fit_probes(client, 1)
            self.assertEqual(list(_block_exact(directory, "Curated_BUSI_mt_0")), list(WIDE))

    def test_mixed_clients_pair_and_summarize(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = _mixed_round(directory)
            m1 = pd.DataFrame(cd.m1_pairs(cd.round_dir(fold_dir, 1),
                                          cd._client_metadata(fold_dir), 0, 1))
            self.assertEqual(len(m1), 3 * (len(WIDE) + 1))
            pair = m1[(m1.client_a == "ISIC_2018_cls_0")
                      & (m1.client_b == "ISIC_2018_seg_0")].set_index("block")
            self.assertEqual(set(pair.pair_class), {"intra_diff_tasks"})
            for block in CLS_ONLY_UNREACHED | SEG_ONLY_UNREACHED:
                self.assertTrue(np.isnan(pair.loc[block, "cos"]), block)
            self.assertTrue(np.isfinite(pair.loc["encoder2", "cos"]))
            totals = m1[m1.block == cd.TOTAL_BLOCK]
            self.assertEqual(len(totals), 3)
            self.assertTrue(np.isfinite(totals.cos).all())
            summary = cd.summarize(m1, pd.DataFrame())
            per_block = summary[summary.scope == "block"].set_index("group")["mean"]
            self.assertEqual(set(per_block.index), set(WIDE) | {cd.TOTAL_BLOCK})
            # NaN pairs are skipped: each block is reached by both clients of at least one pair.
            self.assertTrue(np.isfinite(per_block).all())
            self.assertIsNotNone(cd.report_run(directory))


class SmokeCoverageTests(unittest.TestCase):
    def _run(self, directory):
        fold_dir = _mixed_round(directory)
        (Path(directory) / "config.yaml").write_text(yaml.safe_dump(_probe_config(rounds=1)),
                                                     encoding="utf-8")
        model = MTnnUNet(sequences=1, regions=1, n_classes=3)
        torch.save({"round": 1, "keys": shared_keys(model, WIDE)}, fold_dir / "global_shared.pt")
        return fold_dir

    def test_complete_probes_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run(directory)
            with mock.patch("builtins.print") as printed:
                _verify_m1_coverage(directory)
            self.assertIn("M1_COVERAGE_OK", printed.call_args[0][0])

    def test_a_missing_probe_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = self._run(directory)
            (cd.round_dir(fold_dir, 1) / "ISIC_2018_seg_0_m1.npz").unlink()
            with self.assertRaisesRegex(RuntimeError, "ISIC_2018_seg_0"):
                _verify_m1_coverage(directory)

    def test_a_probe_over_other_blocks_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = self._run(directory)
            client = _probe_client(directory, "ISIC_2018_cls_0", ("cls",), False)
            cd.collect_fit_probes(client, 1)  # trunk-only probe overwrites the wide one
            with self.assertRaisesRegex(RuntimeError, "blocks"):
                _verify_m1_coverage(directory)
            self.assertTrue((cd.round_dir(fold_dir, 1) / "ISIC_2018_cls_0_m1.npz").exists())


RUN_CONFIG = ROOT / "studies" / "e6_sharing" / "wide_sharing.yaml"
E3_WINNER = ROOT / "runs" / "studies" / "e3_loss_ablation" / "resolved" / "seed_1993" / \
    "init_seed_dataset.yaml"
E6_EXPERIMENT = {"study_id": "e6_wider_sharing", "arm_id": "wide_sharing",
                 "method_id": "wide_sharing", "seed": 1993, "setup": "federated"}


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        flat = {}
        for key, child in value.items():
            flat.update(_flatten(child, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: value}


class RunConfigTests(unittest.TestCase):
    """The committed E6 run config (plano.md, E6.5)."""

    def setUp(self):
        self.config = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))

    def test_run_config_selects_the_wide_set(self):
        federated = self.config["federated"]
        self.assertEqual(shared_modules(self.config), WIDE)
        self.assertIs(federated["share_stem"], False)
        self.assertEqual(federated["rounds"], 400)
        self.assertEqual(self.config["experiment"], E6_EXPERIMENT)
        self.assertEqual(self.config["model"]["architecture"], "MTnnUNet")
        self.assertTrue(federated["partition_file"].endswith(
            "data/federated_studies/e3_loss_ablation/seed_1993/federated_mapping.csv"))
        validate_federated_config(self.config)

    @unittest.skipUnless(E3_WINNER.exists(), "needs the resolved E3 Phase B config under runs/")
    def test_run_config_differs_from_the_e3_winner_only_in_the_intervention(self):
        base = _flatten(yaml.safe_load(E3_WINNER.read_text(encoding="utf-8")))
        run = _flatten(self.config)
        differing = {key for key in set(base) | set(run)
                     if base.get(key, "<absent>") != run.get(key, "<absent>")}
        self.assertEqual(differing, {"federated.rounds", "federated.shared_modules",
                                     "experiment.study_id", "experiment.arm_id",
                                     "experiment.method_id"})
        self.assertEqual(base["federated.rounds"], 200)
        self.assertEqual(base["experiment.seed"], E6_EXPERIMENT["seed"])
        self.assertEqual(base["experiment.setup"], E6_EXPERIMENT["setup"])


class StudyHashGuardTests(unittest.TestCase):
    def test_shared_modules_is_not_in_the_working_config(self):
        # Defining it in src/config.yaml would change config_sha256 for every manifest built on it.
        text = (ROOT / "src" / "config.yaml").read_text(encoding="utf-8")
        self.assertNotIn("shared_modules", text)
        self.assertNotIn("shared_modules", yaml.safe_load(text)["federated"])


if __name__ == "__main__":
    unittest.main()
