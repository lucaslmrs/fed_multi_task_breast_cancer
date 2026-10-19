"""Multi-task U-Net on an ImageNet ResNet-34 encoder (stage E7, ``plano.md``).

The encoder blocks are exposed under MTnnUNet's module names, so ``model_split`` (the FedAvg
shared/personalized split), ``validate_sharing`` and the server work unchanged. The geometry is
MTnnUNet's too: ``encoder2`` at H/2 down to ``bottleneck`` at H/32.

==================  ================================================  ========  ============
module              content                                           @128 px   pretrained
==================  ================================================  ========  ============
encoder1            LevelBlock(C, 32, 32) on the raw input (0-255)    128       no
encoder2            conv1, bn1, relu on the ImageNet-normalized input 64        yes
encoder3            maxpool, layer1                                   32        yes
encoder4            layer2                                            16        yes
encoder5            layer3                                            8         yes
bottleneck          layer4                                            4         yes
upsample5/decoder5  ConvT(512->256) / LevelBlock(512, 128, 128)       8         no
upsample4/decoder4  ConvT(128->128) / LevelBlock(256, 64, 64)         16        no
upsample3/decoder3  ConvT(64->64)   / LevelBlock(128, 64, 64)         32        no
upsample2/decoder2  ConvT(64->64)   / LevelBlock(128, 32, 32)         64        no
upsample1/decoder1  ConvT(32->32)   / LevelBlock(64, 32, 16)          128       no
output4..output1    deep supervision, upsampled to full resolution    128       no
process_*           ConvInNormLeReLU(256->256) / (128->256)           8         no
classifier          ConvInNormLeReLU(768->512), GAP, FC 512-256-n'              no
==================  ================================================  ========  ============

Which of these modules are federated is the run's ``federated.shared_modules``; E7 shares every
module except ``encoder1``, ``output1..4`` and ``classifier``. Every module in
``pretrained_modules`` must be shared: the ImageNet weights reach the clients only through the
server's initial global parameters (``model_split.assert_pretrained_shared``).

The ResNet BatchNorm layers are torchvision's ``FrozenBatchNorm2d``: statistics, scale and shift are
buffers, never trained and never updated in ``train()`` mode, so they cannot diverge between
clients (``docs/DECISION_SOURCES.md``, D27). They are float32 buffers inside shared modules, so they
travel with the shared state and FedAvg returns them unchanged (a weighted mean of identical arrays).

The pretrained path normalizes its input inside the model: ``x / 255``, a grayscale input repeated
to three channels, then ImageNet mean/std. The loader already delivers RGB for 3-channel datasets
(``BUSI_dataset.py`` converts OpenCV's BGR), so no channel swap happens here. ``encoder1`` sees the
raw input, as MTnnUNet's stem does, and feeds only ``decoder1``'s skip connection.

``forward`` returns ``([cls_logits], [out4, out3, out2, out1])`` as raw logits: the segmentation
criterion applies the sigmoid itself.
"""

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet34
from torchvision.ops.misc import FrozenBatchNorm2d

from src.models.multitask.MTnnUNet import ConvInNormLeReLU, LevelBlock, conv1x1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MTResNet34UNet(nn.Module):
    """MTnnUNet-compatible multi-task U-Net whose encoder is a frozen-BN ResNet-34."""

    # The modules whose weights come from the ImageNet checkpoint; they must all be shared.
    pretrained_modules = ("encoder2", "encoder3", "encoder4", "encoder5", "bottleneck")
    # Total downsampling factor: H and W must be multiples of it.
    stride = 32

    def __init__(self, sequences, regions, n_classes=3):
        super().__init__()
        if sequences not in (1, 3):
            raise ValueError(
                f"MTResNet34UNet takes 1 or 3 input channels (an ImageNet encoder), got "
                f"{sequences}; extra augmentation channels are not supported"
            )
        self.sequences = int(sequences)
        self.n_classes = 1 if n_classes == 2 else n_classes

        # A LOCAL variable, never assigned to ``self``: only the blocks below are registered, so
        # ``fc``/``avgpool`` never enter the state_dict and no tensor is registered twice.
        backbone = resnet34(weights=None, norm_layer=FrozenBatchNorm2d)

        # Personalized modality stem on the raw input; its only consumer is decoder1's skip.
        self.encoder1 = LevelBlock(sequences, 32, 32)
        # Pretrained encoder.
        self.encoder2 = nn.Sequential(OrderedDict([
            ("conv1", backbone.conv1), ("bn1", backbone.bn1), ("relu", backbone.relu),
        ]))
        self.encoder3 = nn.Sequential(OrderedDict([
            ("maxpool", backbone.maxpool), ("layer1", backbone.layer1),
        ]))
        self.encoder4 = backbone.layer2
        self.encoder5 = backbone.layer3
        self.bottleneck = backbone.layer4

        # Decoders: input = concat(skip, upsampled), in the order MTnnUNet registers them.
        self.decoder5 = LevelBlock(256 + 256, 128, 128)
        self.decoder4 = LevelBlock(128 + 128, 64, 64)
        self.decoder3 = LevelBlock(64 + 64, 64, 64)
        self.decoder2 = LevelBlock(64 + 64, 32, 32)
        self.decoder1 = LevelBlock(32 + 32, 32, 16)

        self.upsample5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        # Deep supervision outputs, every one at full resolution (as in MTnnUNet).
        self.output4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8), conv1x1(64, regions)
        )
        self.output3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4), conv1x1(64, regions)
        )
        self.output2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2), conv1x1(32, regions)
        )
        self.output1 = conv1x1(16, regions)

        # Classification head: encoder5, upsampled bottleneck and decoder5 features, as MTnnUNet.
        self.process_encoder_5 = ConvInNormLeReLU(256, 256)
        self.process_decoder_5 = ConvInNormLeReLU(128, 256)
        self.classifier = nn.Sequential(
            ConvInNormLeReLU(256 * 3, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.n_classes),
        )

        # Not persistent: constants of the input contract, not state to federate or checkpoint.
        self.register_buffer(
            "imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        self.weights_initialization()

    def weights_initialization(self):
        """MTnnUNet's Kaiming init on the convolutions this model trains from scratch."""
        scratch = (self.encoder1, self.decoder5, self.decoder4, self.decoder3, self.decoder2,
                   self.decoder1, self.output4, self.output3, self.output2, self.output1)
        for block in scratch:
            for module in block.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def _check_input(self, x):
        if x.ndim != 4 or x.shape[1] != self.sequences:
            raise ValueError(
                f"MTResNet34UNet was built for {self.sequences}-channel input, got shape "
                f"{tuple(x.shape)}"
            )
        height, width = x.shape[-2:]
        if height % self.stride or width % self.stride:
            raise ValueError(
                f"MTResNet34UNet needs H and W divisible by {self.stride}, got {height}x{width}"
            )

    def normalize(self, x):
        """Raw 0-255 input -> the ImageNet input contract (3 channels, mean/std normalized)."""
        z = x / 255.0
        if z.shape[1] == 1:
            z = z.expand(-1, 3, -1, -1)
        return (z - self.imagenet_mean) / self.imagenet_std

    def forward(self, x):
        self._check_input(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.normalize(x))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        bottleneck = self.bottleneck(e5)

        up5 = self.upsample5(bottleneck)
        d5 = self.decoder5(torch.cat([e5, up5], dim=1))
        up4 = self.upsample4(d5)
        d4 = self.decoder4(torch.cat([e4, up4], dim=1))
        up3 = self.upsample3(d4)
        d3 = self.decoder3(torch.cat([e3, up3], dim=1))
        up2 = self.upsample2(d3)
        d2 = self.decoder2(torch.cat([e2, up2], dim=1))
        up1 = self.upsample1(d2)
        d1 = self.decoder1(torch.cat([e1, up1], dim=1))

        features = torch.cat([self.process_encoder_5(e5), up5, self.process_decoder_5(d5)], dim=1)
        predicted_class = self.classifier(features)

        return [predicted_class], [self.output4(d4), self.output3(d3), self.output2(d2),
                                   self.output1(d1)]
