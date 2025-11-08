"""
Multi-head attention with temperature control for educational visualization.

This module implements scaled dot-product attention from scratch with explicit
temperature parameter to demonstrate how it affects attention sharpness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with temperature control.

    Key features:
    - Returns both output AND attention weights (for visualization)
    - Supports temperature parameter to control attention sharpness
    - Clear, educational implementation of the attention formula:
        Attention(Q, K, V) = softmax(Q @ K^T / (sqrt(d_k) * temperature)) @ V
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)  # Scaling factor for dot products

        # Linear transformations for Q, K, V
        # We use separate linear layers for clarity (could combine into one)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temperature: float = 1.0,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor (batch_size, seq_len, embed_dim)
            key: Key tensor (batch_size, seq_len, embed_dim)
            value: Value tensor (batch_size, seq_len, embed_dim)
            temperature: Temperature parameter for softmax (default 1.0)
                - T < 1.0: Sharpens attention (more focused)
                - T = 1.0: Standard attention
                - T > 1.0: Smooths attention (more distributed)
            return_attention: Whether to return attention weights

        Returns:
            output: Attention output (batch_size, seq_len, embed_dim)
            attention_weights: (Optional) Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape

        # Step 1: Linear projections and reshape for multi-head attention
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, head_dim)
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 2: Compute scaled dot-product attention scores
        # Q @ K^T: (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        #       -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Step 3: Scale by sqrt(d_k) to prevent gradients from vanishing
        attention_scores = attention_scores / self.scale

        # Step 4: Apply temperature scaling
        # Lower temperature (T < 1) makes distribution sharper (more peaked)
        # Higher temperature (T > 1) makes distribution smoother (more uniform)
        attention_scores = attention_scores / temperature

        # Step 5: Apply softmax to get attention weights
        # Softmax converts scores to probabilities that sum to 1
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Optional: Check for NaN (can happen with extreme temperatures)
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            print(f"Warning: NaN/Inf detected in attention weights with temperature={temperature}")
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=1.0, neginf=0.0)

        # Step 6: Apply dropout (regularization)
        attention_weights_dropped = self.dropout(attention_weights)

        # Step 7: Apply attention weights to values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights_dropped, V)

        # Step 8: Reshape and apply output projection
        # Transpose back and reshape: (batch_size, num_heads, seq_len, head_dim)
        #                           -> (batch_size, seq_len, num_heads, head_dim)
        #                           -> (batch_size, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        if return_attention:
            return output, attention_weights  # Return weights WITHOUT dropout for visualization
        else:
            return output, None


def compute_attention_statistics(attention_weights: torch.Tensor) -> dict:
    """
    Compute statistics about attention distribution.

    Useful for understanding how temperature affects attention patterns.

    Args:
        attention_weights: (batch_size, num_heads, seq_len, seq_len)

    Returns:
        Dictionary with statistics:
            - max_weight: Maximum attention weight
            - min_weight: Minimum attention weight
            - entropy: Average entropy across all positions (higher = more uniform)
            - sparsity: Fraction of weights below threshold (0.1)
    """
    # Compute entropy for each query position
    # Entropy = -sum(p * log(p)), higher entropy = more uniform distribution
    eps = 1e-8
    log_weights = torch.log(attention_weights + eps)
    entropy = -(attention_weights * log_weights).sum(dim=-1)  # Sum over key dimension
    avg_entropy = entropy.mean().item()

    # Compute sparsity (fraction of small weights)
    sparsity = (attention_weights < 0.1).float().mean().item()

    return {
        'max_weight': attention_weights.max().item(),
        'min_weight': attention_weights.min().item(),
        'mean_weight': attention_weights.mean().item(),
        'entropy': avg_entropy,
        'sparsity': sparsity
    }


if __name__ == "__main__":
    """Test attention module with toy inputs."""
    print("Testing Multi-Head Attention...")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create toy input
    batch_size = 2
    seq_len = 8
    embed_dim = 64
    num_heads = 2

    # Random query, key, value
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # Initialize attention module
    attn = MultiHeadAttention(embed_dim, num_heads)

    print(f"\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Value: {value.shape}")

    # Test with different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print(f"\nTesting temperature effects:")
    print("-" * 80)

    for temp in temperatures:
        output, attn_weights = attn(query, key, value, temperature=temp, return_attention=True)

        stats = compute_attention_statistics(attn_weights)

        print(f"\nTemperature = {temp:.1f}")
        print(f"  Output shape: {output.shape}")
        print(f"  Attention weights shape: {attn_weights.shape}")
        print(f"  Max weight: {stats['max_weight']:.4f}")
        print(f"  Min weight: {stats['min_weight']:.4f}")
        print(f"  Entropy: {stats['entropy']:.4f}")
        print(f"  Sparsity: {stats['sparsity']:.2%}")

    print("\n" + "=" * 80)
    print("Key observations:")
    print("  - Lower T → Higher max_weight, lower entropy (sharper, more focused)")
    print("  - Higher T → Lower max_weight, higher entropy (smoother, more uniform)")
    print("=" * 80)

    print("\nAttention module test completed successfully!")
