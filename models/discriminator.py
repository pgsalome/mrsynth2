import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type, List, Union, Callable
import functools

from models.components.blocks import get_norm_layer
from models.components.initialization import init_weights


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

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # First layer doesn't use normalization
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        ]

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output 1-channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        # Add sigmoid if needed
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(input)


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

        # First layer
        sequence = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.LeakyReLU(0.2, True)
        ]

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                                 kernel_size=4, stride=2, padding=1, bias=True)),
                nn.LeakyReLU(0.2, True)
            ]

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                             kernel_size=4, stride=1, padding=1, bias=True)),
            nn.LeakyReLU(0.2, True)
        ]

        # Output 1-channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        # Add sigmoid if needed
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(input)


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


def get_discriminator(config: Dict[str, Any], input_nc: int) -> nn.Module:
    """
    Create discriminator based on configuration.

    Args:
        config: Discriminator configuration
        input_nc: Number of input channels

    Returns:
        Discriminator model
    """
    # Get parameters from config
    name = config.get("name", "basic").lower()
    ndf = config.get("ndf", 64)
    n_layers = config.get("n_layers", 3)
    norm_type = config.get("norm", "instance")
    use_sigmoid = config.get("gan_mode", "lsgan") == "vanilla"
    init_type = config.get("init_type", "normal")
    init_gain = config.get("init_gain", 0.02)

    # Get normalization layer
    norm_layer = get_norm_layer(norm_type)

    # Create discriminator based on type
    if name == "basic":
        # PatchGAN discriminator
        netD = PatchDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid
        )
    elif name == "n_layers":
        # PatchGAN with configurable number of layers
        netD = PatchDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid
        )
    elif name == "spectral":
        # Discriminator with spectral normalization
        netD = SpectralNormDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers,
            use_sigmoid=use_sigmoid
        )
    elif name == "multiscale":
        # Multi-scale discriminator
        num_D = config.get("num_D", 3)
        getIntermFeat = config.get("getIntermFeat", False)
        netD = MultiScaleDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            num_D=num_D,
            getIntermFeat=getIntermFeat
        )
    else:
        raise ValueError(f"Discriminator model [{name}] not recognized")

    # Initialize weights
    init_weights(netD, init_type, init_gain)

    return netD