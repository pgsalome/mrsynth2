"""Normalization layers for model architectures."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


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


class SPADE(nn.Module):
    """Spatially-Adaptive Normalization layer."""

    def __init__(self, norm_nc: int, label_nc: int):
        """
        Initialize SPADE layer.

        Args:
            norm_nc: Number of channels in normalized activation
            label_nc: Number of channels in input semantic map
        """
        super(SPADE, self).__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space
        nhidden = 128

        # Network to generate modulation parameters
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input feature map [B, C, H, W]
            segmap: Input semantic map [B, C', H', W']

        Returns:
            Normalized and modulated feature map
        """
        # Part 1. Generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. Resize input semantic map to match input activation size
        if x.size(2) != segmap.size(2) or x.size(3) != segmap.size(3):
            segmap = nn.functional.interpolate(segmap, size=x.size()[2:], mode='nearest')

        # Part 3. Get modulation parameters
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Apply modulation
        out = normalized * (1 + gamma) + beta

        return out


class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D inputs."""

    def __init__(self, channels: int, eps: float = 1e-5):
        """
        Initialize LayerNorm2d.

        Args:
            channels: Number of channels
            eps: Small value added for numerical stability
        """
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Normalized tensor
        """
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply per-channel scale and bias
        scale = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * scale + bias


class GroupNorm2d(nn.Module):
    """Group Normalization wrapper."""

    def __init__(self, channels: int, num_groups: int = 32, eps: float = 1e-5):
        """
        Initialize GroupNorm2d.

        Args:
            channels: Number of channels
            num_groups: Number of groups to separate the channels into
            eps: Small value added for numerical stability
        """
        super(GroupNorm2d, self).__init__()
        self.gn = nn.GroupNorm(
            num_groups=min(num_groups, channels),
            num_channels=channels,
            eps=eps,
            affine=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Normalized tensor
        """
        return self.gn(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, feature_dim: int, conditioning_dim: int):
        """
        Initialize FiLM layer.

        Args:
            feature_dim: Number of feature channels
            conditioning_dim: Dimension of conditioning input
        """
        super(FiLM, self).__init__()
        self.modulation = nn.Linear(conditioning_dim, feature_dim * 2)  # For scale and bias

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input feature map [B, C, H, W]
            condition: Conditioning vector [B, conditioning_dim]

        Returns:
            Modulated feature map
        """
        modulation = self.modulation(condition)

        # Split into scale and bias
        B, C = x.shape[0], x.shape[1]
        gamma, beta = torch.split(modulation, C, dim=1)

        # Reshape for broadcasting
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        # Apply modulation
        return gamma * x + beta