"""PatchGAN discriminator implementation."""

import torch
import torch.nn as nn
from typing import Type, Optional


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for image-to-image translation"""

    def __init__(
            self,
            input_nc: int,
            ndf: int = 64,
            n_layers: int = 3,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            use_sigmoid: bool = False
    ):
        """
        Initialize the PatchGAN discriminator.

        Args:
            input_nc: Number of input channels
            ndf: Number of filters in the first conv layer
            n_layers: Number of conv layers
            norm_layer: Type of normalization layer
            use_sigmoid: Whether to use sigmoid activation
        """
        super(PatchDiscriminator, self).__init__()

        kw = 4  # kernel width
        padw = 1  # padding width
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output 1-channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(input)