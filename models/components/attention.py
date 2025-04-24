"""Attention mechanism implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class CrossAttention(nn.Module):
    """Cross-attention module for conditional GANs."""

    def __init__(self, in_channels: int, condition_channels: int):
        """
        Initialize the cross-attention module.

        Args:
            in_channels: Number of input channels
            condition_channels: Number of channels in the conditioning input
        """
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(condition_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(condition_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable parameter
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]
            condition: Conditioning tensor of shape [B, C', H', W']

        Returns:
            Attention output tensor of same shape as x
        """
        batch_size, C, width, height = x.size()
        _, C_cond, width_cond, height_cond = condition.size()

        # Project into query, key, value spaces
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x (H*W) x C'
        proj_key = self.key_conv(condition).view(batch_size, -1, width_cond * height_cond)  # B x C' x (H'*W')

        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H'*W')
        attention = self.softmax(energy)  # B x (H*W) x (H'*W')

        # Apply attention to values from condition
        proj_value = self.value_conv(condition).view(batch_size, -1, width_cond * height_cond)  # B x C x (H'*W')
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, width, height)  # B x C x H x W

        # Apply residual connection with learnable weight
        out = self.gamma * out + x

        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, in_channels: int, num_heads: int = 8):
        """
        Initialize multi-head self-attention.

        Args:
            in_channels: Number of input channels
            num_heads: Number of attention heads
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"

        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Attention output tensor of same shape
        """
        batch_size, C, height, width = x.size()

        # Project to queries, keys, values
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, 3, self.num_heads, self.head_dim, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Reshape for attention computation
        q = q.permute(0, 1, 3, 2)  # B, num_heads, height*width, head_dim
        k = k.permute(0, 1, 2, 3)  # B, num_heads, head_dim, height*width
        v = v.permute(0, 1, 3, 2)  # B, num_heads, height*width, head_dim

        # Compute attention scores
        attn = torch.matmul(q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # B, num_heads, height*width, head_dim
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, C, height, width)

        # Project output and apply residual connection
        out = self.out_proj(out)
        out = self.gamma * out + x

        return out