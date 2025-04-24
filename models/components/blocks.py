"""Common building blocks for model architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Optional, Type, Union, Callable


class ResnetBlock(nn.Module):
    """Residual block with configurable padding, normalization, and dropout."""

    def __init__(
            self,
            dim: int,
            padding_type: str = 'reflect',
            norm_layer: Optional[Type[nn.Module]] = nn.BatchNorm2d,
            use_dropout: bool = False,
            use_bias: bool = True
    ):
        """
        Initialize the Resnet block.

        Args:
            dim: Number of channels
            padding_type: Type of padding ('reflect', 'replicate', or 'zero')
            norm_layer: Normalization layer
            use_dropout: Whether to use dropout
            use_bias: Whether to use bias in conv layers
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(
            self,
            dim: int,
            padding_type: str,
            norm_layer: Optional[Type[nn.Module]],
            use_dropout: bool,
            use_bias: bool
    ) -> nn.Sequential:
        """
        Construct a convolutional block.

        Args:
            dim: Number of channels
            padding_type: Type of padding
            norm_layer: Normalization layer
            use_dropout: Whether to use dropout
            use_bias: Whether to use bias in conv layers

        Returns:
            Sequential convolutional block
        """
        conv_block = []
        p = 0

        # Add padding
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding type {padding_type} is not implemented')

        # Add conv-norm-relu-dropout
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # Add padding again
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding type {padding_type} is not implemented')

        # Add conv-norm (without activation)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection
        """
        return x + self.conv_block(x)


class UnetSkipConnectionBlock(nn.Module):
    """Unet submodule with skip connection."""

    def __init__(
            self,
            outer_nc: int,
            inner_nc: int,
            input_nc: Optional[int] = None,
            submodule: Optional[nn.Module] = None,
            outermost: bool = False,
            innermost: bool = False,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            use_dropout: bool = False
    ):
        """
        Initialize the Unet submodule.

        Args:
            outer_nc: Number of channels in outer conv layer
            inner_nc: Number of channels in inner conv layer
            input_nc: Number of input channels (if None, same as outer_nc)
            submodule: Previously defined Unet submodule
            outermost: Whether this is the outermost module
            innermost: Whether this is the innermost module
            norm_layer: Normalization layer
            use_dropout: Whether to use dropout
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        # Define downsampling layers
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        # Define upsampling layers
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # Build the model based on position in Unet
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.outermost:
            return self.model(x)
        else:
            # Add skip connection
            return torch.cat([x, self.model(x)], 1)


def get_norm_layer(norm_type: str = 'instance') -> Callable:
    """
    Get normalization layer.

    Args:
        norm_type: Type of normalization ('batch', 'instance', or 'none')

    Returns:
        Normalization layer constructor
    """
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        return lambda x: nn.Identity()
    else:
        raise NotImplementedError(f'normalization layer {norm_type} is not implemented')