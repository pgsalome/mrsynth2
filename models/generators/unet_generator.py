"""U-Net generator implementation."""

import torch
import torch.nn as nn
import functools
from typing import Dict, Any, Optional, Type, List, Union, Callable

from ..components.blocks import UnetSkipConnectionBlock


class UnetGenerator(nn.Module):
    """U-Net Generator with skip connections"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        """
        Construct a Unet generator

        Parameters:
            input_nc (int) -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, if num_downs=7,
                              image of size 128x128 will become of size 1x1 at the bottleneck
            ngf (int) -- the number of filters in the last conv layer
            norm_layer -- normalization layer
            use_dropout (bool) -- if use dropout layers
        """
        super(UnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # Construct the Unet structure
        # Add the innermost layer
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)

        # Add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)

        # Gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)