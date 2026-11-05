"""E8: ``federated.layer_mixing`` blends the global shared state into each client's own copy.

Each shared module starts a round as ``beta * global + (1 - beta) * local``, ``beta`` decaying
linearly over ``order``. Absent or disabled it must leave the hard FedPer swap untouched, and
``beta = 1`` everywhere must equal ``set_shared_state``. The key stays out of ``src/config.yaml``:
every study hashes its config.
"""
import copy
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.federated.config import layer_betas, validate_federated_config
from src.federated.model_split import (get_shared_state, mix_shared_state, set_shared_state,
                                       shared_keys)
from src.models.multitask.MTResNet34UNet import MTResNet34UNet

ROOT = Path(__file__).resolve().parents[1]
E8 = ROOT / "studies" / "e8_layer_mixing"
MIXING = E8 / "resnet34_256_layer_mixing.yaml"


def _config(path=MIXING):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _sharing(config):
    return tuple(config["federated"]["shared_modules"])


def _shifted_global(model, sharing, offset=1.0):
    """A global state that differs from the model in every parameter but keeps its buffers."""
    params = {name for name, _ in model.named_parameters()}
    return [array + offset if key in params else array.copy()
            for key, array in zip(shared_keys(model, sharing), get_shared_state(model, sharing))]


class LayerBetasTests(unittest.TestCase):
    def test_linear_from_encoder_to_decoder(self):
        config = _config()
        betas = layer_betas(config)
        order = config["federated"]["layer_mixing"]["order"]
        self.assertEqual(list(betas), order)
        self.assertAlmostEqual(betas["encoder2"], 1.0)
        self.assertAlmostEqual(betas["decoder1"], 0.2)
        middle = order[len(order) // 2]
        self.assertAlmostEqual(betas[middle], 1.0 - 0.8 * (len(order) // 2) / (len(order) - 1))
        values = list(betas.values())
        self.assertEqual(values, sorted(values, reverse=True))

    def test_absent_or_disabled_is_none(self):
        config = _config()
        disabled = copy.deepcopy(config)
        disabled["federated"]["layer_mixing"]["enabled"] = False
        self.assertIsNone(layer_betas(disabled))
        del config["federated"]["layer_mixing"]
        self.assertIsNone(layer_betas(config))

    def test_order_must_match_shared_modules(self):
        for mutate in (lambda order: order.pop(), lambda order: order.append("encoder1"),
                       lambda order: order.append(order[0])):
            config = _config()
            mutate(config["federated"]["layer_mixing"]["order"])
            with self.assertRaisesRegex(ValueError, "order"):
                layer_betas(config)

    def test_bounds_must_be_in_unit_interval(self):
        for field, value in (("beta_max", 1.5), ("beta_min", -0.1), ("beta_min", True)):
            config = _config()
            config["federated"]["layer_mixing"][field] = value
            with self.assertRaisesRegex(ValueError, field):
                layer_betas(config)

    def test_refused_on_the_standalone_arm(self):
        config = _config()
        config["federated"]["standalone"] = True
        with self.assertRaisesRegex(ValueError, "standalone"):
            layer_betas(config)

    def test_validated_before_flower(self):
        config = _config()
        config["federated"]["layer_mixing"]["order"].pop()
        with self.assertRaisesRegex(ValueError, "order"):
            validate_federated_config(config)


class MixSharedStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = _config()
        self.sharing = _sharing(self.config)
        self.model = MTResNet34UNet(sequences=1, regions=1, n_classes=3)
        self.local = copy.deepcopy(self.model.state_dict())
        self.arrays = _shifted_global(self.model, self.sharing)

    def test_beta_one_equals_set_shared_state(self):
        reference = copy.deepcopy(self.model)
        set_shared_state(reference, self.arrays, self.sharing)
        mix_shared_state(self.model, self.arrays, {m: 1.0 for m in self.sharing}, self.sharing)
        for key, tensor in reference.state_dict().items():
            self.assertTrue(torch.equal(tensor, self.model.state_dict()[key]), key)

    def test_beta_zero_keeps_the_local_copy(self):
        mix_shared_state(self.model, self.arrays, {m: 0.0 for m in self.sharing}, self.sharing)
        for key, tensor in self.local.items():
            self.assertTrue(torch.equal(tensor, self.model.state_dict()[key]), key)

    def test_each_module_uses_its_own_beta(self):
        betas = layer_betas(self.config)
        mix_shared_state(self.model, self.arrays, betas, self.sharing)
        state = self.model.state_dict()
        params = {name for name, _ in self.model.named_parameters()}
        for key, array in zip(shared_keys(self.model, self.sharing), self.arrays):
            beta = betas[key.split(".", 1)[0]]
            expected = beta * torch.as_tensor(array) + (1.0 - beta) * self.local[key]
            self.assertTrue(torch.allclose(state[key], expected, atol=1e-6), key)
            if key in params:  # a parameter moved by exactly beta * offset
                shift = (state[key] - self.local[key]).flatten()[0].item()
                self.assertAlmostEqual(shift, beta, places=5)

    def test_frozen_batchnorm_buffers_are_unchanged(self):
        params = {name for name, _ in self.model.named_parameters()}
        buffers = [k for k in shared_keys(self.model, self.sharing) if k not in params]
        self.assertTrue(buffers, "MTResNet34UNet should share FrozenBN buffers")
        # Arbitrary values, like ImageNet statistics: fresh buffers are exact 0/1 and hide rounding.
        generator = torch.Generator().manual_seed(1)
        state = self.model.state_dict()
        for key in buffers:
            state[key].copy_(torch.rand(state[key].shape, generator=generator) * 3.7 + 0.01)
        self.local = copy.deepcopy(state)
        self.arrays = _shifted_global(self.model, self.sharing)
        mix_shared_state(self.model, self.arrays, layer_betas(self.config), self.sharing)
        for key in buffers:
            self.assertTrue(torch.equal(self.local[key], self.model.state_dict()[key]), key)

    def test_missing_beta_or_bad_shape_is_refused(self):
        betas = layer_betas(self.config)
        betas.pop("decoder1")
        with self.assertRaisesRegex(ValueError, "decoder1"):
            mix_shared_state(self.model, self.arrays, betas, self.sharing)
        with self.assertRaisesRegex(ValueError, "Expected"):
            mix_shared_state(self.model, self.arrays[:-1], layer_betas(self.config), self.sharing)
        bad = list(self.arrays)
        bad[0] = np.zeros((1,), dtype=bad[0].dtype)
        with self.assertRaisesRegex(ValueError, "shape"):
            mix_shared_state(self.model, bad, layer_betas(self.config), self.sharing)


class ManifestTests(unittest.TestCase):
    def test_arms_differ_only_in_identity_standalone_and_mixing(self):
        def flat(config):
            return {(section, key): value for section, body in config.items()
                    if isinstance(body, dict) for key, value in body.items()}
        arms = {path.stem: flat(_config(path)) for path in E8.glob("*.yaml")}
        self.assertEqual(set(arms), {"resnet34_256_layer_mixing", "resnet34_256_control",
                                     "resnet34_256_local"})
        allowed = {("federated", "standalone"), ("federated", "layer_mixing"),
                   ("experiment", "arm_id"), ("experiment", "method_id"),
                   ("experiment", "setup")}
        control = arms["resnet34_256_control"]
        for name, arm in arms.items():
            differing = {key for key in set(arm) | set(control)
                         if arm.get(key) != control.get(key)}
            self.assertLessEqual(differing, allowed, name)

    def test_resolution_and_master_follow_e4(self):
        config = _config()
        variants = {name: cfg["variant"] for name, cfg in config["datasets"].items()}
        self.assertEqual(variants, {"Curated_BUSI": "processed_128", "ISIC_2018": "processed_128",
                                    "TCGA_LGG": "processed_256", "SIIM_ACR": "processed_256"})
        e4 = _config(ROOT / "studies" / "e4_preprocessing" / "resolution_256.yaml")
        self.assertEqual(config["federated"]["partition_file"], e4["federated"]["partition_file"])
        self.assertEqual(config["model"]["architecture"], "MTResNet34UNet")

    def test_control_and_local_have_no_mixing(self):
        for arm in ("resnet34_256_control", "resnet34_256_local"):
            self.assertIsNone(layer_betas(_config(E8 / f"{arm}.yaml")), arm)


class StudyHashGuardTests(unittest.TestCase):
    def test_layer_mixing_is_not_in_the_working_config(self):
        # Defining it in src/config.yaml would change config_sha256 for every manifest built on it.
        text = (ROOT / "src" / "config.yaml").read_text(encoding="utf-8")
        self.assertNotIn("layer_mixing", text)


if __name__ == "__main__":
    unittest.main()
