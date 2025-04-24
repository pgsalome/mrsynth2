"""Spectral Normalization discriminator implementation."""

import torch
import torch.nn as nn
from typing import Optional


class SpectralNormDiscriminator(nn.Module):
    """Discriminator with spectral normalization for more stable training"""

    def __init__(
            self,
            input_nc: int,
            ndf: int = 64,
            n_layers: int = 3,
            use_sigmoid: bool = False
    ):
        """
        Initialize the spectral norm discriminator.

        Args:
            input_nc: Number of input channels
            ndf: Number of filters in the first conv layer
            n_layers: Number of conv layers
            use_sigmoid: Whether to use sigmoid activation
        """
        super(SpectralNormDiscriminator, self).__init__()

        kw = 4  # kernel width
        padw = 1  # padding width

        # First layer
        sequence = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        # Intermediate layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                                 kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                             kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        # Output 1-channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        # Add sigmoid if needed
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(input)