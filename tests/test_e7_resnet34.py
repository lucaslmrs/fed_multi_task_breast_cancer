"""E7: ``MTResNet34UNet``, an ImageNet ResNet-34 encoder under MTnnUNet's module names.

The model must plug into the FedPer split unchanged (same names, same shared/personalized
contract as E6's wide sharing), keep its BatchNorm frozen (buffers only, never stepped) and return
raw logits at full resolution. The dispatcher must refuse the paths that were not ported to a
pretrained encoder instead of silently training one from scratch.

The ImageNet weights are read from the local torch hub cache only, verified, and loaded once, into
the reference model of the initial global parameters; tests use a fake checkpoint in a temporary
hub directory (with the expected sha256 prefix patched to the fake's), never the network.
"""
import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torchvision
import yaml
from flwr.common import parameters_to_ndarrays

from scripts.smoke_federated import _verify_frozen_pretrained
from src.federated import conflict_diagnostics as cd
from src.federated.config import active_datasets, validate_federated_config
from src.federated.local_trainer import local_objective
from src.federated.model_split import (assert_pretrained_shared, get_personalized_state,
                                       get_shared_state, resolved_shared_modules,
                                       set_personalized_state, set_shared_state, shared_keys,
                                       validate_sharing)
from src.federated.server import _weighted_arrays
from src.models.multitask import MTResNet34UNet as resnet_module
from src.models.multitask.MTnnUNet import MTnnUNet
from src.models.multitask.MTResNet34UNet import (IMAGENET_CHECKPOINT, MTResNet34UNet,
                                                 imagenet_checkpoint_path, load_imagenet_state,
                                                 map_imagenet_state)
from src.training_federated import _initial_shared_parameters
from src.utils.experiment_init import (init_criterion_classification,
                                       init_criterion_segmentation, init_multitask_model)
from src.utils.training_runtime import PrecisionPolicy
from torchvision.ops.misc import FrozenBatchNorm2d

ROOT = Path(__file__).resolve().parents[1]

WIDE = ("encoder2", "encoder3", "encoder4", "encoder5", "bottleneck",
        "decoder5", "decoder4", "decoder3", "decoder2", "decoder1",
        "upsample5", "upsample4", "upsample3", "upsample2", "upsample1",
        "process_encoder_5", "process_decoder_5")
PERSONAL = {"encoder1", "output1", "output2", "output3", "output4", "classifier"}
PRETRAINED = ("encoder2", "encoder3", "encoder4", "encoder5", "bottleneck")
TOP_LEVEL = PERSONAL | set(WIDE)
# 202 shared state_dict tensors (parameters + FrozenBN buffers), identical for 1 and 3 channels.
WIDE_TENSORS = 202
WIDE_VALUES = 23_914_976


def _model(channels=1, n_classes=3, seed=0):
    torch.manual_seed(seed)
    return MTResNet34UNet(sequences=channels, regions=1, n_classes=n_classes)


def _image(channels, size, batch=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, channels, size, size, generator=generator) * 255.0


def _top(keys):
    return {str(key).split(".", 1)[0] for key in keys}


def _frozen_bn_buffers(model):
    """Every FrozenBatchNorm2d buffer, by state_dict key."""
    names = {}
    for module_name, module in model.named_modules():
        if isinstance(module, FrozenBatchNorm2d):
            for buffer_name, _ in module.named_buffers(recurse=False):
                names[f"{module_name}.{buffer_name}"] = module
    return names


class ForwardTests(unittest.TestCase):
    def test_shapes_for_every_modality_class_count_and_resolution(self):
        for channels in (1, 3):
            for n_classes, logits in ((2, 1), (7, 7)):
                for size in (128, 256):
                    with self.subTest(channels=channels, n_classes=n_classes, size=size):
                        model = _model(channels, n_classes)
                        with torch.no_grad():
                            cls, seg = model(_image(channels, size))
                        self.assertEqual(len(cls), 1)
                        self.assertEqual(tuple(cls[0].shape), (2, logits))
                        self.assertEqual(len(seg), 4)
                        for output in seg:
                            self.assertEqual(tuple(output.shape), (2, 1, size, size))

    def test_outputs_are_the_raw_head_outputs(self):
        # No sigmoid/softmax: the DiceBCE criterion applies the sigmoid itself.
        model = _model(3, 7)
        captured = {}
        names = ("classifier", "output4", "output3", "output2", "output1")
        for name in names:
            getattr(model, name).register_forward_hook(
                lambda module, args, output, name=name: captured.__setitem__(name, output))
        with torch.no_grad():
            cls, seg = model(_image(3, 128))
        self.assertIs(cls[0], captured["classifier"])
        for output, name in zip(seg, names[1:]):
            self.assertIs(output, captured[name])
        self.assertLess(float(torch.cat([o.flatten() for o in seg]).min()), 0.0)

    def test_input_contract_is_checked(self):
        with self.assertRaisesRegex(ValueError, "1 or 3 input channels"):
            MTResNet34UNet(sequences=2, regions=1)
        model = _model(1)
        with self.assertRaisesRegex(ValueError, "1-channel"):
            model(_image(3, 128))
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            model(torch.rand(1, 1, 112, 128))

    def test_gray_input_is_repeated_and_imagenet_normalized(self):
        model = _model(1)
        x = _image(1, 64)
        z = model.normalize(x)
        self.assertEqual(tuple(z.shape), (2, 3, 64, 64))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.assertTrue(torch.allclose(z, (x.repeat(1, 3, 1, 1) / 255 - mean) / std))


class StateDictTests(unittest.TestCase):
    def test_top_level_modules_and_tensor_hygiene(self):
        model = _model()
        state = model.state_dict()
        self.assertEqual(_top(state), TOP_LEVEL)  # no backbone.*, fc, avgpool
        self.assertFalse([key for key in state if "num_batches_tracked" in key])
        self.assertTrue(all(tensor.is_floating_point() for tensor in state.values()))
        # The input-normalization constants are buffers of the model but never state.
        self.assertNotIn("imagenet_mean", state)
        self.assertNotIn("imagenet_std", state)
        self.assertIn("imagenet_mean", dict(model.named_buffers()))

    def test_frozen_bn_buffers_are_state_but_not_parameters(self):
        model = _model()
        state, params = model.state_dict(), dict(model.named_parameters())
        frozen = _frozen_bn_buffers(model)
        # 36 FrozenBatchNorm2d layers in ResNet-34 (bn1, 2 per BasicBlock, 3 downsamples) x 4.
        self.assertEqual(len(frozen), 36 * 4)
        self.assertTrue(set(frozen) <= set(state))
        self.assertFalse(set(frozen) & set(params))
        self.assertEqual(_top(frozen), set(PRETRAINED))
        for key in ("encoder2.bn1.running_var", "encoder4.0.downsample.1.weight",
                    "bottleneck.2.bn2.bias"):
            self.assertIn(key, frozen)

    def test_pretrained_modules_attribute(self):
        self.assertEqual(MTResNet34UNet.pretrained_modules, PRETRAINED)


class WideSharingTests(unittest.TestCase):
    def test_gray_and_rgb_expose_identical_shared_states(self):
        gray, rgb = _model(1, 3), _model(3, 7)
        keys = validate_sharing(gray, WIDE)
        self.assertEqual(keys, validate_sharing(rgb, WIDE))
        self.assertEqual(len(keys), WIDE_TENSORS)
        state = gray.state_dict()
        self.assertEqual(sum(state[key].numel() for key in keys), WIDE_VALUES)
        self.assertEqual([a.shape for a in get_shared_state(gray, WIDE)],
                         [a.shape for a in get_shared_state(rgb, WIDE)])
        self.assertEqual(resolved_shared_modules(gray, WIDE), list(WIDE))
        for model in (gray, rgb):
            personalized = get_personalized_state(model, WIDE)
            self.assertEqual(_top(personalized), PERSONAL)
            self.assertEqual(set(personalized) | set(keys), set(model.state_dict()))
            self.assertFalse(set(personalized) & set(keys))

    def test_shared_plus_personalized_reproduce_the_forward(self):
        for channels, n_classes in ((1, 3), (3, 7)):
            with self.subTest(channels=channels):
                source, target = _model(channels, n_classes, 0), _model(channels, n_classes, 1)
                x = _image(channels, 128)
                with torch.no_grad():
                    expected_cls, expected_seg = source(x)
                    different, _ = target(x)
                self.assertFalse(torch.equal(expected_cls[0], different[0]))
                set_shared_state(target, get_shared_state(source, WIDE), sharing=WIDE)
                set_personalized_state(target, get_personalized_state(source, WIDE), WIDE)
                with torch.no_grad():
                    cls, seg = target(x)
                self.assertTrue(torch.equal(cls[0], expected_cls[0]))
                for actual, expected in zip(seg, expected_seg):
                    self.assertTrue(torch.equal(actual, expected))

    def test_legacy_trunk_spec_shares_exactly_the_pretrained_encoder(self):
        model = _model()
        self.assertEqual(_top(shared_keys(model, False)), set(PRETRAINED))


class DispatchTests(unittest.TestCase):
    def _build(self, architecture, **kwargs):
        return init_multitask_model(architecture, sequences=1, regions=1, n_classes=3, width=24,
                                    deep_supervision=True, **kwargs)

    def test_resnet_needs_an_explicit_pretrained_choice(self):
        with self.assertRaisesRegex(ValueError, "pretrained"):
            self._build("MTResNet34UNet")
        self.assertIsInstance(self._build("MTResNet34UNet", pretrained=False), MTResNet34UNet)

    def test_unknown_architecture_raises(self):
        with self.assertRaisesRegex(ValueError, "MTnnUNett"):
            self._build("MTnnUNett")

    def test_pretrained_true_is_refused_on_other_architectures(self):
        for architecture in ("MTnnUNet", "Multi_BTSUNet", "MTUNetPlusPlus"):
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(ValueError, "pretrained=True"):
                    self._build(architecture, pretrained=True)

    def test_mtnnunet_is_unchanged(self):
        for pretrained in (None, False):
            with self.subTest(pretrained=pretrained):
                torch.manual_seed(5)
                built = self._build("MTnnUNet", pretrained=pretrained)
                torch.manual_seed(5)
                direct = MTnnUNet(sequences=1, regions=1, n_classes=3)
                self.assertIs(type(built), MTnnUNet)
                for (key, a), (other, b) in zip(built.state_dict().items(),
                                                direct.state_dict().items()):
                    self.assertEqual(key, other)
                    self.assertTrue(torch.equal(a, b), key)


class FreezeTests(unittest.TestCase):
    def test_an_adam_step_leaves_every_frozen_bn_buffer_untouched(self):
        model = _model(1, 3)
        frozen = _frozen_bn_buffers(model)
        before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        parameter_ids = {id(p) for p in model.parameters()}
        for module in set(frozen.values()):
            self.assertEqual(list(module.parameters(recurse=False)), [])
            self.assertFalse({id(b) for b in module.buffers(recurse=False)} & parameter_ids)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        cls, seg = model(_image(1, 64))
        loss = sum(output.mean() for output in seg) + cls[0].logsumexp(dim=1).mean()
        loss.backward()
        optimizer.step()

        after = model.state_dict()
        for key in frozen:
            self.assertTrue(torch.equal(before[key], after[key]), key)
        # The step happened: the pretrained convolutions themselves are trainable.
        self.assertFalse(torch.equal(before["encoder2.conv1.weight"],
                                     after["encoder2.conv1.weight"]))


# ---- ImageNet loading ------------------------------------------------------------------------
def _fake_imagenet(hub_dir, seed=0):
    """A torchvision-layout ResNet-34 checkpoint (plain BatchNorm with ``num_batches_tracked``, and
    ``fc``) with non-default BN statistics, written where ``imagenet_checkpoint_path`` looks."""
    torch.manual_seed(seed)
    state = torchvision.models.resnet34(weights=None).state_dict()
    generator = torch.Generator().manual_seed(seed)
    for key, tensor in state.items():
        if tensor.is_floating_point() and tensor.ndim == 1:  # BN weight/bias/mean/var, fc.bias
            state[key] = torch.rand(tensor.shape, generator=generator) + 0.5
    path = Path(hub_dir) / "checkpoints" / IMAGENET_CHECKPOINT
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state, hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def _hub(hub_dir, sha_prefix=None):
    """Point ``torch.hub.get_dir`` at ``hub_dir``; optionally accept the fake's sha256 prefix."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("torch.hub.get_dir", return_value=str(hub_dir)))
        if sha_prefix is not None:
            stack.enter_context(
                mock.patch.object(resnet_module, "IMAGENET_SHA256_PREFIX", sha_prefix))
        yield


class _FakeHubCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.hub = Path(cls._directory.name)
        cls.fake, cls.sha = _fake_imagenet(cls.hub)
        cls.mapped = map_imagenet_state(cls.fake)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def fake_hub(self):
        return _hub(self.hub, self.sha[:8])


class ImageNetLoaderTests(_FakeHubCase):
    def test_prefix_map_covers_the_pretrained_modules_exactly(self):
        model_state = _model().state_dict()
        expected = {key for key in model_state if key.split(".", 1)[0] in PRETRAINED}
        self.assertEqual(set(self.mapped), expected)
        self.assertEqual(len(self.mapped), 36 + 36 * 4)  # 36 conv weights + FrozenBN buffers
        for model_key, checkpoint_key in (
                ("encoder2.conv1.weight", "conv1.weight"),
                ("encoder2.bn1.running_var", "bn1.running_var"),
                ("encoder3.layer1.2.bn2.bias", "layer1.2.bn2.bias"),
                ("encoder4.0.downsample.1.running_mean", "layer2.0.downsample.1.running_mean"),
                ("encoder5.5.conv2.weight", "layer3.5.conv2.weight"),
                ("bottleneck.0.downsample.0.weight", "layer4.0.downsample.0.weight")):
            self.assertIs(self.mapped[model_key], self.fake[checkpoint_key])
            self.assertEqual(self.mapped[model_key].shape, model_state[model_key].shape)

    def test_an_unmapped_key_is_refused(self):
        with self.assertRaisesRegex(ValueError, "prefix map"):
            map_imagenet_state({**self.fake, "layer5.0.conv1.weight": torch.zeros(1)})

    def test_pretrained_true_loads_the_encoder_and_nothing_else(self):
        with self.fake_hub(), self.assertLogs(level="INFO") as logs:
            torch.manual_seed(3)
            loaded = init_multitask_model("MTResNet34UNet", sequences=3, regions=1, n_classes=7,
                                          pretrained=True).state_dict()
        torch.manual_seed(3)
        scratch = init_multitask_model("MTResNet34UNet", sequences=3, regions=1, n_classes=7,
                                       pretrained=False).state_dict()
        for key, tensor in loaded.items():
            expected = self.mapped[key] if key in self.mapped else scratch[key]
            self.assertTrue(torch.equal(tensor, expected), key)
        line = next(entry for entry in logs.output if "[pretrained]" in entry)
        self.assertIn("180 tensors", line)
        self.assertIn(self.sha[:8], line)

    def test_a_hash_mismatch_is_refused(self):
        with _hub(self.hub), self.assertRaisesRegex(ValueError, "sha256"):
            load_imagenet_state()

    def test_missing_weights_raise_without_touching_the_network(self):
        network = AssertionError("the loader must never download")
        with contextlib.ExitStack() as stack:
            empty = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(_hub(empty))
            patched = [stack.enter_context(mock.patch(target, side_effect=network))
                       for target in ("torch.hub.load_state_dict_from_url",
                                      "torch.hub.download_url_to_file",
                                      "torchvision.models._api.load_state_dict_from_url")]
            self.assertEqual(imagenet_checkpoint_path(),
                             Path(empty) / "checkpoints" / IMAGENET_CHECKPOINT)
            with self.assertRaises(FileNotFoundError) as caught:
                init_multitask_model("MTResNet34UNet", sequences=1, regions=1, n_classes=3,
                                     pretrained=True)
        message = str(caught.exception)
        self.assertIn(IMAGENET_CHECKPOINT, message)
        self.assertIn("ResNet34_Weights.IMAGENET1K_V1", message)
        for function in patched:
            function.assert_not_called()

    @unittest.skipUnless(imagenet_checkpoint_path().is_file(),
                         "real ImageNet ResNet-34 weights are not cached")
    def test_the_cached_checkpoint_is_the_expected_one(self):
        mapped, _, digest = load_imagenet_state()
        self.assertTrue(digest.startswith("b627a593"))
        model_state = _model().state_dict()
        self.assertEqual(len(mapped), 180)
        for key, tensor in mapped.items():
            self.assertEqual(tensor.shape, model_state[key].shape, key)


class AssertPretrainedSharedTests(unittest.TestCase):
    def test_wide_and_legacy_trunk_specs_pass(self):
        model = _model()
        for sharing in (WIDE, False, True, PRETRAINED):
            with self.subTest(sharing=sharing):
                assert_pretrained_shared(model, sharing)

    def test_a_personalized_pretrained_module_is_refused(self):
        model = _model()
        with self.assertRaisesRegex(ValueError, r"\['bottleneck'\]"):
            assert_pretrained_shared(model, tuple(m for m in WIDE if m != "bottleneck"))
        with self.assertRaisesRegex(ValueError, "encoder3"):
            assert_pretrained_shared(model, ("encoder2",))

    def test_models_without_pretrained_modules_pass(self):
        assert_pretrained_shared(MTnnUNet(sequences=1, regions=1, n_classes=3), ("decoder1",))


class AggregationIdentityTests(unittest.TestCase):
    def test_weighted_mean_of_identical_float32_arrays_is_bitwise_identity(self):
        # FrozenBN buffers come back identical from every client; FedAvg must not move them.
        rng = np.random.default_rng(0)
        arrays = [rng.standard_normal(4096).astype(np.float32) * np.float32(scale)
                  for scale in (1e-6, 1.0, 1e4)]
        arrays.append(np.array([0.0, 1e-38, 3.0e38, 1.0 / 3.0, 1e-5], dtype=np.float32))
        arrays.append((rng.random(512) + 0.5).astype(np.float32))  # running_var-like
        for weights in (np.array([0.7, 0.2, 0.1]), rng.dirichlet(np.ones(16)),
                        np.full(7, 1.0 / 7.0)):
            with self.subTest(n=len(weights)):
                aggregated = _weighted_arrays([[a.copy() for a in arrays] for _ in weights],
                                              weights)
                for expected, actual in zip(arrays, aggregated):
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertTrue(np.array_equal(expected.view(np.uint32),
                                                   actual.view(np.uint32)))


def _probe_batches(seed, n=4, size=64):
    generator = torch.Generator().manual_seed(seed)
    return [{"image": torch.rand(2, 1, size, size, generator=generator) * 255.0,
             "mask": (torch.rand(2, 1, size, size, generator=generator) > 0.7).float(),
             "label": torch.randint(0, 3, (2,), generator=generator),
             "has_mask": torch.ones(2, dtype=torch.bool),
             "has_label": torch.ones(2, dtype=torch.bool)} for _ in range(n)]


def _probe_client(directory, client_id, tasks, seed=0):
    """The attributes ``collect_fit_probes`` reads from a ``FederatedClient``."""
    state_dir = Path(directory) / "fold_0" / f"client_{client_id}"
    state_dir.mkdir(parents=True, exist_ok=True)
    lambdas = {"seg": 0.8, "cls": 0.2}
    return SimpleNamespace(
        client_id=client_id, model=_model(1, 3, seed), sharing=WIDE, tasks=tuple(tasks),
        task="+".join(tasks), task_lambdas={task: lambdas[task] for task in tasks},
        precision=PrecisionPolicy("fp32", "cpu"), device="cpu", num_classes=3,
        inversely_weighted=True, seg_criterion=init_criterion_segmentation("DiceBCE"),
        cls_criterion=init_criterion_classification(n_classes=3, device="cpu"),
        config={"federated": {"rounds": 1, "standalone": False},
                "runtime": {"diagnostics": {"gradient_conflict": {
                    "enabled": True, "every_n_rounds": 1, "probe_batches": 4,
                    "sketch_dim": 4096,
                    "lookahead": {"enabled": False, "val_batches": 2, "train_batches": 2}}}}},
        state_dir=state_dir, train_loader=_probe_batches(seed),
    )


class M1ProbeTests(unittest.TestCase):
    def test_probes_cover_parameters_only_and_every_wide_block(self):
        unreached = {("seg", "cls"): set(),
                     ("cls",): {"decoder1", "decoder2", "decoder3", "decoder4",
                                "upsample1", "upsample2", "upsample3", "upsample4"},
                     ("seg",): {"process_encoder_5", "process_decoder_5"}}
        with tempfile.TemporaryDirectory() as directory:
            for tasks, expected_zero in unreached.items():
                with self.subTest(tasks=tasks):
                    client_id = "_".join(tasks)
                    client = _probe_client(directory, client_id, tasks)
                    params = dict(client.model.named_parameters())
                    keys, _ = cd.trunk_gradient_halves(client, client.train_loader)
                    self.assertEqual(keys, [k for k in shared_keys(client.model, WIDE)
                                            if k in params])
                    self.assertFalse(set(keys) & set(_frozen_bn_buffers(client.model)))
                    cd.collect_fit_probes(client, 1)
                    path = cd.round_dir(Path(directory) / "fold_0", 1) / f"{client_id}_m1.npz"
                    with np.load(path) as stored:
                        blocks = [str(block) for block in stored["blocks"]]
                        norms = {block: float(stored[f"{block}::exact"][3]) for block in blocks}
                    self.assertEqual(blocks, list(WIDE))
                    self.assertEqual({b for b, value in norms.items() if value == 0},
                                     expected_zero)
                    self.assertTrue(all(np.isfinite(value) for value in norms.values()))


def _repo_config(**federated):
    config = yaml.safe_load((ROOT / "src" / "config.yaml").read_text(encoding="utf-8"))
    config["model"]["architecture"] = "MTResNet34UNet"
    config["federated"].update(share_stem=False, **federated)
    return config


class InitialSharedTests(_FakeHubCase):
    def test_reference_model_carries_the_imagenet_encoder(self):
        config = _repo_config(shared_modules=list(WIDE))
        validate_federated_config(config)
        datasets = active_datasets(config)
        self.assertGreater(len({config["datasets"][d]["channels"] for d in datasets}), 1)
        with self.fake_hub():
            arrays = parameters_to_ndarrays(_initial_shared_parameters(config, datasets, "cpu", 0))
        keys = shared_keys(_model(), WIDE)
        self.assertEqual(len(arrays), len(keys))
        compared = 0
        for key, array in zip(keys, arrays):
            if key in self.mapped:
                self.assertTrue(np.array_equal(array, self.mapped[key].numpy()), key)
                compared += 1
        self.assertEqual(compared, len(self.mapped))

    def test_a_personalized_pretrained_module_fails_before_flower(self):
        config = _repo_config(shared_modules=[m for m in WIDE if m != "encoder5"])
        with self.fake_hub(), self.assertRaisesRegex(ValueError, "encoder5"):
            _initial_shared_parameters(config, active_datasets(config), "cpu", 0)


class SmokeFrozenCheckTests(_FakeHubCase):
    def _run_dir(self, directory, architecture="MTResNet34UNet", standalone=False):
        run = Path(directory)
        (run / "config.yaml").write_text(yaml.safe_dump({"model": {"architecture": architecture}}),
                                         encoding="utf-8")
        with self.fake_hub():
            model = init_multitask_model("MTResNet34UNet", sequences=1, regions=1, n_classes=3,
                                         pretrained=True)
        fold = run / "fold_0"
        if standalone:
            (fold / "client_x").mkdir(parents=True)
            torch.save({"round": 2, "model": model.state_dict()}, fold / "client_x" / "state.pt")
            return run
        fold.mkdir(parents=True)
        torch.save({"round": 2, "keys": shared_keys(model, WIDE),
                    "arrays": [torch.as_tensor(a) for a in get_shared_state(model, WIDE)]},
                   fold / "global_shared.pt")
        return run

    def _verify(self, run, **kwargs):
        with self.fake_hub(), contextlib.redirect_stdout(io.StringIO()) as out:
            _verify_frozen_pretrained(run, **kwargs)
        return out.getvalue()

    def _replace(self, run, key, tensor):
        path = run / "fold_0" / "global_shared.pt"
        stored = torch.load(path)
        stored["arrays"][stored["keys"].index(key)] = tensor(stored["arrays"][
            stored["keys"].index(key)])
        torch.save(stored, path)

    def test_an_intact_trunk_passes(self):
        for standalone in (False, True):
            with self.subTest(standalone=standalone), tempfile.TemporaryDirectory() as directory:
                output = self._verify(self._run_dir(directory, standalone=standalone))
                self.assertIn("FROZEN_PRETRAINED_OK", output)
                self.assertIn("buffers=144", output)

    def test_a_moved_buffer_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_dir(directory)
            self._replace(run, "encoder5.0.bn1.running_var",
                          lambda old: torch.nextafter(old, torch.full_like(old, 10.0)))
            with self.assertRaisesRegex(RuntimeError, "FrozenBN"):
                self._verify(run)

    def test_a_random_encoder_fails_the_drift_check_only(self):
        random_conv1 = _model(seed=9).state_dict()["encoder2.conv1.weight"]
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_dir(directory)
            self._replace(run, "encoder2.conv1.weight", lambda old: random_conv1)
            with self.assertRaisesRegex(RuntimeError, "relative L2"):
                self._verify(run)
            self.assertIn("FROZEN_PRETRAINED_OK", self._verify(run, max_conv1_drift=None))

    def test_a_scratch_architecture_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "no pretrained encoder"):
                self._verify(self._run_dir(directory, architecture="MTnnUNet"))


RUN_CONFIG = ROOT / "studies" / "e7_transfer" / "resnet34_imagenet.yaml"
BASE_CONFIG = (ROOT / "runs" / "studies" / "e3_loss_ablation" / "resolved" / "seed_1993"
               / "init_seed_dataset.yaml")


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        flat = {}
        for key, inner in value.items():
            flat.update(_flatten(inner, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: value}


class RunConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))

    def test_run_config_is_a_valid_wide_resnet_run(self):
        from src.federated.config import shared_modules
        from src.training_federated import _build_model

        validate_federated_config(self.config)
        self.assertEqual(self.config["model"]["architecture"], "MTResNet34UNet")
        self.assertEqual(shared_modules(self.config), WIDE)
        self.assertEqual(self.config["federated"]["rounds"], 400)
        self.assertEqual(self.config["experiment"], {
            "study_id": "e7_transfer_learning", "arm_id": "resnet34_imagenet",
            "method_id": "resnet34_imagenet", "seed": 1993, "setup": "federated"})
        self.assertTrue(str(self.config["federated"]["partition_file"]).endswith(
            "federated_studies/e3_loss_ablation/seed_1993/federated_mapping.csv"))
        shapes = set()
        for dataset in active_datasets(self.config):
            model = _build_model(self.config, dataset, "cpu")
            assert_pretrained_shared(model, WIDE)
            shapes.add(tuple(tuple(a.shape) for a in get_shared_state(model, WIDE)))
        self.assertEqual(len(shapes), 1)

    @unittest.skipUnless(BASE_CONFIG.exists(), "E3 resolved config not available")
    def test_differs_from_the_e3_winner_only_where_declared(self):
        base = _flatten(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")))
        run = _flatten(self.config)
        changed = {key for key in set(base) | set(run) if base.get(key) != run.get(key)}
        self.assertEqual(changed, {
            "model.architecture", "federated.shared_modules", "federated.rounds",
            "experiment.study_id", "experiment.arm_id", "experiment.method_id"})


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                     "BF16 needs a CUDA GPU (the orchestrator runs this once, GPU free)")
class BF16CudaTests(unittest.TestCase):
    def test_bf16_forward_backward_is_finite_and_keeps_buffers(self):
        model = _model(3, 7).to("cuda")
        frozen = _frozen_bn_buffers(model)
        before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        precision = PrecisionPolicy("bf16", "cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)
        generator = torch.Generator().manual_seed(0)
        data = {"image": torch.rand(4, 3, 128, 128, generator=generator) * 255.0,
                "mask": (torch.rand(4, 1, 128, 128, generator=generator) > 0.8).float(),
                "label": torch.randint(0, 7, (4,), generator=generator),
                "has_mask": torch.ones(4, dtype=torch.bool),
                "has_label": torch.ones(4, dtype=torch.bool)}
        model.train()
        loss, *_ = local_objective(
            model, data, precision.move(data["image"]), ("seg", "cls"), {"seg": 0.8, "cls": 0.2},
            "cuda", 7, init_criterion_segmentation("DiceBCE", dice_batch=True),
            init_criterion_classification(n_classes=7, device="cuda"), True, precision,
        )
        precision.ensure_finite(loss, "E7 BF16 test")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        after = model.state_dict()
        for key in frozen:
            self.assertTrue(torch.equal(before[key], after[key]), key)


if __name__ == "__main__":
    unittest.main()
