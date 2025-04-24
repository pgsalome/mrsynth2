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


class SelfAttention(nn.Module):
    """Self-attention module for GANs."""

    def __init__(self, in_channels: int):
        """
        Initialize the self-attention module.

        Args:
            in_channels: Number of input channels
        """
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable parameter
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Attention output tensor of same shape
        """
        batch_size, C, width, height = x.size()

        # Project into query, key, value spaces
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x (H*W) x C'
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C' x (H*W)

        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H*W)
        attention = self.softmax(energy)  # B x (H*W) x (H*W)

        # Apply attention to values
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, width, height)  # B x C x H x W

        # Apply residual connection with learnable weight
        out = self.gamma * out + x

        return out


class AdaIN(nn.Module):
    """Adaptive Instance Normalization layer."""

    def __init__(self, style_dim: int, channel_dim: int):
        """
        Initialize AdaIN layer.

        Args:
            style_dim: Dimension of style input
            channel_dim: Number of channels to normalize
        """
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channel_dim, affine=False)
        self.style_scale = nn.Linear(style_dim, channel_dim)
        self.style_bias = nn.Linear(style_dim, channel_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input feature map [B, C, H, W]
            style: Style vector [B, style_dim]

        Returns:
            Normalized and modulated feature map
        """
        # Apply instance norm
        x = self.instance_norm(x)

        # Extract scale and bias from style
        style_scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        style_bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]

        # Apply scale and bias
        return x * style_scale + style_bias


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