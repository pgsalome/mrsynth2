"""Multi-scale discriminator implementation."""

import torch
import torch.nn as nn
from typing import Type, List, Optional

from .patch_discriminator import PatchDiscriminator


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator that operates at multiple image resolutions"""

    def __init__(
            self,
            input_nc: int,
            ndf: int = 64,
            n_layers: int = 3,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            use_sigmoid: bool = False,
            num_D: int = 3,
            getIntermFeat: bool = False
    ):
        """
        Initialize multi-scale discriminator.

        Args:
            input_nc: Number of input channels
            ndf: Number of filters in the first conv layer
            n_layers: Number of conv layers
            norm_layer: Type of normalization layer
            use_sigmoid: Whether to use sigmoid activation
            num_D: Number of discriminators at different scales
            getIntermFeat: Whether to return intermediate features
        """
        super(MultiScaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.getIntermFeat = getIntermFeat

        # Create multiple discriminators
        for i in range(num_D):
            # Create discriminator at this scale
            netD = PatchDiscriminator(
                input_nc=input_nc,
                ndf=ndf,
                n_layers=n_layers,
                norm_layer=norm_layer,
                use_sigmoid=use_sigmoid
            )

            # Add to module
            setattr(self, f'layer{i}', netD)

        # Downsample layer for multi-scale operation
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass

        Args:
            input: Input tensor

        Returns:
            List of outputs from each discriminator
        """
        results = []

        # Pass input through each discriminator with downsampling
        for i in range(self.num_D):
            if i > 0:
                input = self.downsample(input)

            # Get discriminator and run forward pass
            netD = getattr(self, f'layer{i}')
            output = netD(input)
            results.append(output)

        return results