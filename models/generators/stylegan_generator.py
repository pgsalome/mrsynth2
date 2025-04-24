"""StyleGAN2-based generator implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Type, List, Union, Callable

from ..components.normalization import AdaIN


class StyleGAN2SynthesisNetwork(nn.Module):
    """StyleGAN2 synthesis network"""

    def __init__(self, style_dim, output_nc, ngf=64, n_layers=8, use_bias=True):
        super(StyleGAN2SynthesisNetwork, self).__init__()

        self.n_layers = n_layers
        self.style_dim = style_dim

        # Initial constant input
        self.constant_input = nn.Parameter(torch.randn(1, ngf, 4, 4))

        # AdaIN layers and convs
        self.style_convs = nn.ModuleList()
        self.adain_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        current_size = 4
        current_channels = ngf

        for i in range(n_layers):
            # Add upsampling for all but first layer
            if i > 0:
                self.upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear'))

            # First conv after upsampling
            self.style_convs.append(
                nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1, bias=use_bias)
            )

            # AdaIN after first conv
            self.adain_layers.append(AdaIN(style_dim, current_channels))

            # Second conv
            next_channels = current_channels // 2 if i > n_layers // 2 else current_channels
            self.style_convs.append(
                nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1, bias=use_bias)
            )

            # AdaIN after second conv
            self.adain_layers.append(AdaIN(style_dim, next_channels))

            # Update current state
            current_channels = next_channels
            current_size *= 2

        # Final conv to get to output_nc
        self.final_conv = nn.Conv2d(current_channels, output_nc, kernel_size=1, bias=use_bias)
        self.activation = nn.Tanh()

    def forward(self, style):
        # Start from the constant input
        batch_size = style.shape[0]
        x = self.constant_input.repeat(batch_size, 1, 1, 1)

        # Process through the synthesis network
        style_idx = 0
        for i in range(self.n_layers):
            # Upsample if not the first layer
            if i > 0:
                x = self.upsamples[i - 1](x)

            # First conv and AdaIN
            x = self.style_convs[2 * i](x)
            x = self.adain_layers[2 * i](x, style)
            x = F.leaky_relu(x, 0.2)

            # Second conv and AdaIN
            x = self.style_convs[2 * i + 1](x)
            x = self.adain_layers[2 * i + 1](x, style)
            x = F.leaky_relu(x, 0.2)

        # Final processing
        x = self.final_conv(x)
        x = self.activation(x)

        return x


class StyleGAN2Generator(nn.Module):
    """StyleGAN2-based generator for high-quality image synthesis"""

    def __init__(self, input_nc, output_nc, ngf=64, n_layers=8, style_dim=512,
                 use_mapping_network=True, mapping_layers=8, use_bias=True):
        super(StyleGAN2Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.style_dim = style_dim
        self.use_mapping_network = use_mapping_network

        # Create mapping network if needed
        if use_mapping_network:
            mapping_layers_list = []
            mapping_layers_list.append(nn.Linear(input_nc, style_dim))
            mapping_layers_list.append(nn.LeakyReLU(0.2))

            for i in range(mapping_layers - 1):
                mapping_layers_list.append(nn.Linear(style_dim, style_dim))
                mapping_layers_list.append(nn.LeakyReLU(0.2))

            self.mapping_network = nn.Sequential(*mapping_layers_list)

        # Create synthesis network
        self.synthesis_network = StyleGAN2SynthesisNetwork(
            style_dim=style_dim,
            output_nc=output_nc,
            ngf=ngf,
            n_layers=n_layers,
            use_bias=use_bias
        )

    def forward(self, input):
        if self.use_mapping_network:
            # Process input to extract style vector
            if len(input.shape) == 4:  # If input is an image
                b, c, h, w = input.shape
                input_flat = input.view(b, c * h * w)
                style = self.mapping_network(input_flat)
            else:  # If input is already a vector
                style = self.mapping_network(input)
        else:
            # Treat input directly as style
            style = input

        # Generate image using synthesis network
        output = self.synthesis_network(style)
        return output