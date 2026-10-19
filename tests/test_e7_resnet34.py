"""E7: ``MTResNet34UNet``, an ImageNet ResNet-34 encoder under MTnnUNet's module names.

The model must plug into the FedPer split unchanged (same names, same shared/personalized
contract as E6's wide sharing), keep its BatchNorm frozen (buffers only, never stepped) and return
raw logits at full resolution. The dispatcher must refuse the paths that were not ported to a
pretrained encoder instead of silently training one from scratch.
"""
import unittest

import torch

from src.federated.model_split import (get_personalized_state, get_shared_state,
                                       resolved_shared_modules, set_personalized_state,
                                       set_shared_state, shared_keys, validate_sharing)
from src.models.multitask.MTnnUNet import MTnnUNet
from src.models.multitask.MTResNet34UNet import MTResNet34UNet
from src.utils.experiment_init import init_multitask_model
from torchvision.ops.misc import FrozenBatchNorm2d

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


if __name__ == "__main__":
    unittest.main()
