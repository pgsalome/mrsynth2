"""Attention-enhanced ResNet generator implementation."""

import torch
import torch.nn as nn
import functools
from typing import Dict, Any, Optional, Type, List, Union, Callable

from ..components.blocks import ResnetBlock
from ..components.attention import SelfAttention


class AttentionResnetGenerator(nn.Module):
    """Resnet-based generator with self-attention layers"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, padding_type='reflect', use_attention=True, attention_layers=[5, 6, 7]):
        """
        Initialize with attention mechanism

        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            ngf: Number of filters in the last conv layer
            norm_layer: Normalization layer
            use_dropout: Whether to use dropout in the generator
            n_blocks: Number of ResNet blocks
            padding_type: Type of padding ('reflect', 'replicate', or 'zero')
            use_attention: Whether to use self-attention layers
            attention_layers: Indices of ResNet blocks to add attention after
        """
        super(AttentionResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_attention = use_attention
        self.attention_layers = attention_layers

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # Resnet blocks with attention
        mult = 2 ** n_downsampling
        model_with_attention = []
        for i in range(n_blocks):
            model_with_attention.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                                    use_dropout=use_dropout, use_bias=use_bias))
            # Add self-attention after specified layers
            if use_attention and i in attention_layers:
                model_with_attention.append(SelfAttention(ngf * mult))

        # Upsampling
        upsampling = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            upsampling += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=use_bias),
                           norm_layer(int(ngf * mult / 2)),
                           nn.ReLU(True)]
        upsampling += [nn.ReflectionPad2d(3)]
        upsampling += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        upsampling += [nn.Tanh()]

        # Combine the model parts
        self.initial_layers = nn.Sequential(*model)
        self.resnet_attention_blocks = nn.Sequential(*model_with_attention)
        self.upsampling_layers = nn.Sequential(*upsampling)

    def forward(self, input):
        """Forward with attention blocks"""
        out = self.initial_layers(input)
        out = self.resnet_attention_blocks(out)
        out = self.upsampling_layers(out)
        return out