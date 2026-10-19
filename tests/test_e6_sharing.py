"""E6: ``federated.shared_modules`` selects the modules FedAvg averages.

Absent (or null), the legacy ``share_stem`` prefix rule must reproduce today's shared keys byte for
byte on every architecture. Present, an explicit list of top-level modules is matched exactly and
validated twice before Flower starts: as config (``shared_modules``) and against the built model
(``validate_sharing``). The key stays out of ``src/config.yaml``: every study hashes its config.
"""
import unittest
from pathlib import Path

import torch
import yaml
from flwr.common import parameters_to_ndarrays

from src.federated.config import active_datasets, shared_modules, validate_federated_config
from src.federated.model_split import (get_personalized_state, get_shared_state, is_shared,
                                       resolved_shared_modules, set_personalized_state,
                                       set_shared_state, shared_keys, validate_sharing)
from src.models.multitask.MTnnUNet import MTnnUNet
from src.training_federated import _initial_shared_parameters, _log_sharing
from src.utils.experiment_init import init_multitask_model
from src.utils.supervision import LOG_VARS_ATTR, attach_task_uncertainty

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


class StudyHashGuardTests(unittest.TestCase):
    def test_shared_modules_is_not_in_the_working_config(self):
        # Defining it in src/config.yaml would change config_sha256 for every manifest built on it.
        text = (ROOT / "src" / "config.yaml").read_text(encoding="utf-8")
        self.assertNotIn("shared_modules", text)
        self.assertNotIn("shared_modules", yaml.safe_load(text)["federated"])


if __name__ == "__main__":
    unittest.main()
