import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F

# source:
# https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-4-the-finale-24dc7e3d9d19
from homework.model.efficient_net.conv_bn_act_block import ConvBNActBlock
from homework.model.efficient_net.mb_conv import MBConv1, MBConv6


def create_stage(mb_conv_type,
                 num_layers,
                 filters_in,
                 filters_out,
                 kernel_size=3,
                 se_ratio=24,
                 stride=1,
                 dropout_rate=0
                 ):
    """Creates a Sequential consisting of [num_layers] layer_type"""
    assert num_layers >= 1, "num layers should be at least 1"
    layers = [mb_conv_type(filters_in if i == 0 else filters_out,
                           filters_out,
                           kernel_size=kernel_size,
                           stride=stride,
                           se_ratio=se_ratio,
                           dropout_rate=dropout_rate) for i in range(num_layers)]
    layers = nn.Sequential(*layers)
    return layers


def scale_width(w, w_factor):
    """Scales width given a scale factor"""
    w *= w_factor
    new_w = (int(w + 4) // 8) * 8
    new_w = max(8, new_w)
    if new_w < 0.9 * w:
        new_w += 8
    return int(new_w)


class EfficientNetMini(pl.LightningModule):
    """
    Generic EfficientNet that takes in the width and depth scale factors and scales accordingly
    Minified version for CIFAR 100
    """

    def __init__(self,
                 lr,
                 w_factor=1,
                 d_factor=1,
                 output_size=100):
        super(EfficientNetMini, self).__init__()
        self.save_hyperparameters()
        base_widths = [(32, 16), (16, 24), (24, 40),
                       (40, 80), (80, 112), (112, 192),
                       (192, 1280)]
        base_depths = [1, 2, 2, 3, 3, 1]
        scaled_widths = [(scale_width(w[0], w_factor), scale_width(w[1], w_factor))
                         for w in base_widths]
        scaled_depths = [math.ceil(d_factor * d) for d in base_depths]
        kernel_sizes = [3, 3, 5, 3, 5, 3]
        strides = [1, 1, 1, 1, 1, 1]
        ps = [0, 0.029, 0.057, 0.086, 0.114, 0.171]
        # TODO: stride might be reduced to 1
        self.stem = ConvBNActBlock(3,
                                   scaled_widths[0][0],
                                   stride=2,
                                   padding=1,
                                   kernel_size=3)

        stages = []
        for i in range(6):
            layer_type = MBConv1 if (i == 0) else MBConv6
            r = 4 if (i == 0) else 24
            stage = create_stage(layer_type,
                                 scaled_depths[i],
                                 *scaled_widths[i],
                                 kernel_size=kernel_sizes[i],
                                 stride=strides[i],
                                 se_ratio=r,
                                 dropout_rate=ps[i])
            stages.append(stage)
        self.stages = nn.Sequential(*stages)

        self.pre_head = ConvBNActBlock(*scaled_widths[-1],
                                       kernel_size=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(scaled_widths[-1][1], output_size)
        )

    def feature_extractor(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pre_head(x)
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=-1) == labels).float().mean()
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc, on_step=False, on_epoch=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


if __name__ == '__main__':
    model = EfficientNetMini()
    x = torch.rand(16, 3, 32, 32)
    res = model(x)
    print(res)
