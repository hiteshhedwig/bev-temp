"""
Tiny BEV Transformer for educational attention visualization.

A minimal transformer architecture designed for:
- Fast CPU training (<5 minutes)
- Clear attention pattern visualization
- Understanding temperature effects on softmax
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from detection import DetectionHead


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for BEV patches.

    We use learnable embeddings rather than sinusoidal for simplicity.
    """

    def __init__(self, num_patches: int, embed_dim: int) -> None:
        """
        Initialize positional encoding.

        Args:
            num_patches: Number of patches (e.g., 64 for 8x8 grid)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pos_embed


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Architecture: Multi-Head Attention -> Add & Norm -> FFN -> Add & Norm
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize transformer layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ffn_dim: Feedforward network hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, seq_len, embed_dim)
            temperature: Temperature for attention softmax
            return_attention: Whether to return attention weights

        Returns:
            output: Transformed features
            attention_weights: (Optional) Attention weights
        """
        # Self-attention with residual connection
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            temperature=temperature,
            return_attention=return_attention,
        )
        x = self.norm1(x + attn_out)

        # Feedforward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


class BEVTransformer(nn.Module):
    """
    Tiny BEV Transformer for object detection.

    Architecture:
        Input (32x32 BEV) -> Patch Embedding (8x8 patches)
        -> Positional Encoding -> Transformer Layers (2)
        -> Detection Head -> Object predictions
    """

    def __init__(
        self,
        grid_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        ffn_dim: int = 128,
        num_queries: int = 3,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize BEV Transformer.

        Args:
            grid_size: Size of input BEV grid (assumed square)
            patch_size: Size of each patch
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ffn_dim: Feedforward network hidden dimension
            num_queries: Number of object queries (max detections)
            num_classes: Number of object classes
            dropout: Dropout probability
        """
        super().__init__()

        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches = (grid_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Patch embedding: Convert patches to embeddings
        # Input: (batch, 1, grid_size, grid_size)
        # Output: (batch, num_patches, embed_dim)
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.num_patches, embed_dim)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Detection head (now needs grid_size and patch_size for patch coords)
        self.detection_head = DetectionHead(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            grid_size=grid_size,
            patch_size=patch_size,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        return_attention: bool = False,
    ) -> Tuple[dict, Optional[list]]:
        """
        Forward pass.

        Args:
            x: Input BEV grid (batch_size, 1, grid_size, grid_size)
            temperature: Temperature for attention softmax
            return_attention: Whether to return attention weights from all layers

        Returns:
            predictions: Dictionary with 'pred_centers' and 'pred_logits'
            attention_weights: (Optional) List of attention weights from each layer
        """
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # (B, E, H', W')

        # Reshape to sequence: (B, E, H', W') -> (B, num_patches, E)
        x = x.flatten(2).transpose(1, 2)

        # Step 2: Add positional encoding
        x = self.pos_encoding(x)

        # Step 3: Pass through transformer layers
        attention_weights_list = []

        for layer in self.layers:
            x, attn_weights = layer(x, temperature, return_attention)
            if return_attention and attn_weights is not None:
                attention_weights_list.append(attn_weights)

        # Step 4: Detection head
        predictions = self.detection_head(x)

        if return_attention:
            return predictions, attention_weights_list
        else:
            return predictions, None

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    """Test BEV Transformer."""
    print("Testing BEV Transformer...")

    # Set seed
    torch.manual_seed(42)

    # Create model
    model = BEVTransformer(
        grid_size=32,
        patch_size=4,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        ffn_dim=128,
        num_queries=3,
        num_classes=2,
    )

    print(f"\nModel architecture:")
    print(f"  Grid size: 32x32")
    print(f"  Patch size: 4x4")
    print(f"  Number of patches: {model.num_patches}")
    print(f"  Embedding dimension: {model.embed_dim}")
    print(f"  Number of layers: {model.num_layers}")
    print(f"  Total parameters: {model.get_num_parameters():,}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 32, 32)

    print(f"\nInput shape: {x.shape}")

    # Forward pass without attention
    predictions, _ = model(x, temperature=1.0, return_attention=False)

    print(f"\nPredictions:")
    print(f"  Centers: {predictions['pred_centers'].shape}")
    print(f"  Logits: {predictions['pred_logits'].shape}")

    # Forward pass with attention
    predictions, attn_weights = model(x, temperature=1.0, return_attention=True)

    print(f"\nAttention weights:")
    for i, attn in enumerate(attn_weights):
        print(f"  Layer {i}: {attn.shape}")

    print("\nBEV Transformer test completed successfully!")
