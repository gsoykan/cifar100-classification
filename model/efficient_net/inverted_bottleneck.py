import torch
import torch.nn as nn

# sources:
# https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-3-mbconv-squeeze-and-excitation-and-more-4ca9fd62d302
# https://amaarora.github.io/2020/08/13/efficientnet.html
# https://github.com/3outeille/Research-Paper-Summary/blob/master/src/architecture/efficientnet/tensorflow_2/utils.py
# https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
# https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/9

from homework.model.efficient_net.conv_bn_act_block import ConvBNActBlock
from homework.model.efficient_net.drop_sample import DropSample
from homework.model.efficient_net.squeeze_excitation import SqueezeExcitationBlock


class InvertedBottleneck(nn.Module):
    """
         Basic bottleneck structure.

         Parameters:
         -inputs: Tensor, input tensor of conv layer.
         -filters_in: Integer, dimension of the input space.
         -filters_out: Integer, dimension of the output space.
         -kernel_size: Integer or tuple of 2 integers, width and height of filters.
         -expansion_coef: Integer, expansion coefficient.
         -se_ratio: Float, ratio use to squeeze the input filters.
         -stride: Integer or tuple of 2 integers, conv stride.
     """

    def __init__(self,
                 filters_in,
                 filters_out,
                 expansion_coefficient,
                 kernel_size=3,
                 se_ratio=24,
                 stride=1,
                 dropout_rate=0):
        super().__init__()
        # dim after expansion
        expanded = filters_in * expansion_coefficient
        padding = (kernel_size - 1) // 2
        self.skip_connection = (filters_in == filters_out) and (stride == 1)
        self.expand_pw = nn.Identity() \
            if (expansion_coefficient == 1) \
            else ConvBNActBlock(filters_in,
                                expanded,
                                kernel_size=1)
        self.depth_wise = ConvBNActBlock(expanded,
                                         expanded,
                                         kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         groups=expanded)
        self.se = SqueezeExcitationBlock(expanded,
                                         r=se_ratio)
        self.reduce_pw = ConvBNActBlock(expanded,
                                        filters_out,
                                        kernel_size=1,
                                        use_act=False)
        # This can be replaced with dropout but one of the sources claims that original paper users drop_sample method
        self.drop_sample = DropSample(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.expand_pw(x)
        x = self.depth_wise(x)
        x = self.se(x)
        x = self.reduce_pw(x)
        if self.skip_connection:
            x = self.drop_sample(x)
            x = x + residual
        return x


if __name__ == '__main__':
    model = InvertedBottleneck(3, 3, 2, stride=2)
    x = torch.rand(16, 3, 32, 32)
    res = model(x)
    print(res)
