import torch
import torch.nn as nn


# sources:
# https://amaarora.github.io/2020/07/24/SeNet.html
# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
# https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-3-mbconv-squeeze-and-excitation-and-more-4ca9fd62d302

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, c_in, r=16):
        """
        SE Block
        :param c: in channel size
        :param r: reduction ratio
        """
        super(SqueezeExcitationBlock, self).__init__()
        reduction = max(1, c_in // r)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(c_in, reduction, kernel_size=1),
                                        nn.SiLU(),
                                        nn.Conv2d(reduction, c_in, kernel_size=1),
                                        nn.Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


if __name__ == '__main__':
    se_block = SqueezeExcitationBlock(3, r=16)
    x = torch.rand(16, 3, 32, 32)
    res = se_block(x)
    print(res)
