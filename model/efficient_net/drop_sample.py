import torch
import torch.nn as nn


# source: https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-3-mbconv-squeeze-and-excitation-and-more-4ca9fd62d302
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
class DropSample(nn.Module):
    """Drops each sample in x with probability p during training"""

    def __init__(self, p=0):
        super().__init__()
        assert 0 <= p <= 1, 'p must be in range of [0,1]'
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        batch_size = x.shape[0]
        keep_prob = 1 - self.p
        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype)
        binary_tensor = torch.floor(random_tensor)
        output = x / keep_prob * binary_tensor.to(x.device)
        return output
