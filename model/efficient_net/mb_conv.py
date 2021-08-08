import torch
import torch.nn as nn

from homework.model.efficient_net.inverted_bottleneck import InvertedBottleneck


class MBConv1(InvertedBottleneck):
    def __init__(self,
                 filters_in,
                 filters_out,
                 kernel_size=3,
                 se_ratio=24,
                 stride=1,
                 dropout_rate=0):
        super().__init__(filters_in,
                         filters_out,
                         1,
                         kernel_size=kernel_size,
                         stride=stride,
                         dropout_rate=dropout_rate,
                         se_ratio=se_ratio)


class MBConv6(InvertedBottleneck):
    def __init__(self,
                 filters_in,
                 filters_out,
                 kernel_size=3,
                 se_ratio=24,
                 stride=1,
                 dropout_rate=0):
        super().__init__(filters_in,
                         filters_out,
                         6,
                         kernel_size=kernel_size,
                         stride=stride,
                         dropout_rate=dropout_rate,
                         se_ratio=se_ratio)
