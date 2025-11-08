"""
Detection head for BEV object detection using query-to-patch attention.

Key ideas:
- Each learnable object query attends over the BEV patch features.
- The attention weights over patches are used to compute a *soft-argmax*
  center coordinate for each query.
- We still predict class logits via an MLP on the attended context.

This makes the predicted center explicitly tied to "where the query looks",
which is great for educational visualization.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Cross-attention-based detection head with soft-argmax centers.

    Takes transformer output features and predicts:
        - Object centers (x, y) in normalized coordinates [0, 1]
        - Object classes (car vs pedestrian)
    """

    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_classes: int,
        grid_size: int,
        patch_size: int,
        hidden_dim: int = 128,
    ) -> None:
        """
        Initialize detection head.

        Args:
            embed_dim: Dimension of transformer features
            num_queries: Number of object queries (max objects to detect)
            num_classes: Number of object classes
            grid_size: Size of the original BEV grid (e.g., 32)
            patch_size: Patch size used in the transformer (e.g., 4)
            hidden_dim: Hidden dimension for the classification MLP
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Learnable object queries (like DETR)
        # Shape: (num_queries, embed_dim)
        self.object_queries = nn.Parameter(
            torch.randn(num_queries, embed_dim) * 0.02
        )

        # Single-head query-to-patch attention:
        #   Query = object_queries
        #   Key/Value = patch features
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Register patch coordinates for soft-argmax centers
        num_patches_side = grid_size // patch_size
        xs = torch.linspace(
            0.5 / num_patches_side,
            1.0 - 0.5 / num_patches_side,
            steps=num_patches_side,
        )
        ys = torch.linspace(
            0.5 / num_patches_side,
            1.0 - 0.5 / num_patches_side,
            steps=num_patches_side,
        )
        # ys = rows (y), xs = cols (x)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # Shape: (num_patches, 2) with columns (x, y)
        patch_coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        # Not a parameter, but moved with the module to the right device
        self.register_buffer("patch_coords", patch_coords)

        # MLP for classification (uses attended context vectors)
        self.class_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Transformer output
                      Shape: (batch_size, num_patches, embed_dim)

        Returns:
            Dictionary with:
                - 'pred_centers': (batch_size, num_queries, 2)
                - 'pred_logits': (batch_size, num_queries, num_classes)
        """
        batch_size, num_patches, embed_dim = features.shape
        assert (
            embed_dim == self.embed_dim
        ), f"Expected embed_dim={self.embed_dim}, got {embed_dim}"

        # Expand learnable object queries to batch:
        # (num_queries, E) -> (B, num_queries, E)
        queries = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Project queries, keys, values
        Q = self.query_proj(queries)          # (B, Q, E)
        K = self.key_proj(features)           # (B, P, E)
        V = self.value_proj(features)         # (B, P, E)

        # Scaled dot-product attention between queries and patches
        # scores: (B, Q, P)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # Attention weights over patches (for each query)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, Q, P)

        # Context vectors: summary of patch features for each query
        # context: (B, Q, E)
        context = torch.matmul(attn_weights, V)

        # --- Soft-argmax centers ------------------------------------------
        # patch_coords: (P, 2) with normalized (x, y) in [0, 1]
        # centers: (B, Q, 2)
        centers = torch.matmul(attn_weights, self.patch_coords)

        # Classification logits from context
        pred_logits = self.class_head(context)  # (B, Q, C)

        return {
            "pred_centers": centers,
            "pred_logits": pred_logits,
        }


if __name__ == "__main__":
    """Quick smoke test for the detection head."""
    torch.manual_seed(42)

    batch_size = 4
    num_patches = 64   # 8x8 patches
    embed_dim = 64
    num_queries = 3
    num_classes = 2
    grid_size = 32
    patch_size = 4

    # Fake transformer features
    features = torch.randn(batch_size, num_patches, embed_dim)

    head = DetectionHead(
        embed_dim=embed_dim,
        num_queries=num_queries,
        num_classes=num_classes,
        grid_size=grid_size,
        patch_size=patch_size,
        hidden_dim=128,
    )

    outputs = head(features)
    print("Pred centers shape:", outputs["pred_centers"].shape)
    print("Pred logits shape:", outputs["pred_logits"].shape)
