from typing import Tuple

import torch
from torch import nn, Tensor
from tsl.nn.layers import TemporalConv


class TemporalPartialConv(nn.Module):

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 stride: int = 1,
                 bias: bool = True,
                 padding: int = 0,
                 causal_pad: bool = True,
                 weight_norm: bool = False,
                 channel_last: bool = False):
        super().__init__()
        self.input_conv = TemporalConv(input_channels=input_channels,
                                       output_channels=output_channels,
                                       kernel_size=kernel_size,
                                       dilation=dilation,
                                       stride=stride,
                                       padding=padding,
                                       causal_pad=causal_pad,
                                       weight_norm=weight_norm,
                                       channel_last=channel_last,
                                       bias=False)
        self.mask_conv = TemporalConv(input_channels=1,
                                      output_channels=1,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      stride=stride,
                                      padding=padding,
                                      causal_pad=causal_pad,
                                      weight_norm=weight_norm,
                                      channel_last=channel_last,
                                      bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_channels))
        else:
            self.register_parameter('bias', None)

        self.mask_conv.conv.weight.requires_grad = False
        torch.nn.init.constant_(self.mask_conv.conv.weight, 1.0)

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.input_conv(x * mask)

        output_mask = self.mask_conv(mask.float())

        invalid_mask = output_mask == 0
        output_mask.masked_fill_(invalid_mask, 1.0)

        output = output / output_mask
        if self.bias is not None:
            output = output + self.bias

        output.masked_fill_(invalid_mask, 0.0)

        return output, ~invalid_mask
