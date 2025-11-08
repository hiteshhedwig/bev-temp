"""
Attention visualization script for BEV Transformer.

Generates high-quality visualizations of attention patterns overlaid on BEV grids.
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # optional but fine to keep
import torch

sys.path.append("data")
sys.path.append("model")

from data.generate_bev import generate_synthetic_bev  # type: ignore
from model.transformer import BEVTransformer  # type: ignore


def plot_attention_map(
    model: torch.nn.Module,
    grid: torch.Tensor,
    temperature: float = 1.0,
    layer_idx: int = 0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot attention maps for all heads in a specific layer.

    Args:
        model: Trained BEV Transformer
        grid: Input BEV grid (1, 1, H, W)
        temperature: Temperature for attention softmax
        layer_idx: Which transformer layer to visualize
        save_path: Path to save the figure
    """
    from scipy.ndimage import zoom  # local import so script works even if not needed

    model.eval()

    with torch.no_grad():
        # Get predictions and attention weights
        predictions, attention_weights = model(
            grid,
            temperature=temperature,
            return_attention=True,
        )

    # Safety check on layer index
    if layer_idx < 0 or layer_idx >= len(attention_weights):
        raise ValueError(
            f"layer_idx={layer_idx} is out of range for "
            f"{len(attention_weights)} transformer layers."
        )

    # Get attention weights for specified layer
    # Shape: (1, num_heads, num_patches, num_patches)
    attn = attention_weights[layer_idx].squeeze(0)  # (num_heads, num_patches, num_patches)

    num_heads = attn.shape[0]
    num_patches = attn.shape[1]
    patch_per_side = int(np.sqrt(num_patches))

    # Get input grid for visualization
    input_grid = grid.squeeze().cpu().numpy()  # (H, W)

    # Create figure
    plt.figure(figsize=(16, 4 * num_heads))

    for head_idx in range(num_heads):
        # Get attention weights for this head
        head_attn = attn[head_idx].cpu().numpy()  # (num_patches, num_patches)

        # Average attention across all query positions to get overall pattern
        avg_attn = head_attn.mean(axis=0)  # (num_patches,)

        # Reshape to 2D grid
        attn_grid = avg_attn.reshape(patch_per_side, patch_per_side)

        # Plot 1: Input BEV grid
        ax1 = plt.subplot(num_heads, 3, head_idx * 3 + 1)
        ax1.imshow(input_grid, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax1.set_title(
            f"Head {head_idx} - Input BEV (T={temperature:.1f})",
            fontsize=14,
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Attention heatmap
        ax2 = plt.subplot(num_heads, 3, head_idx * 3 + 2)
        im = ax2.imshow(attn_grid, cmap="viridis", origin="lower")
        ax2.set_title(f"Head {head_idx} - Attention Map", fontsize=14)
        ax2.set_xlabel("Patch X")
        ax2.set_ylabel("Patch Y")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        # Plot 3: Overlay attention on input
        ax3 = plt.subplot(num_heads, 3, head_idx * 3 + 3)
        ax3.imshow(input_grid, cmap="gray", origin="lower", vmin=0, vmax=1, alpha=0.6)

        # Resize attention map to match input grid size
        scale = input_grid.shape[0] / attn_grid.shape[0]
        attn_resized = zoom(attn_grid, scale, order=1)

        # Overlay attention
        ax3.imshow(attn_resized, cmap="viridis", origin="lower", alpha=0.5)
        ax3.set_title(f"Head {head_idx} - Overlay", fontsize=14)
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved attention visualization to {save_path}")
        # Also save as PDF for vector graphics
        pdf_path = save_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved PDF version to {pdf_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_distribution(
    model: torch.nn.Module,
    grid: torch.Tensor,
    temperature: float = 1.0,
    layer_idx: int = 0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot distribution of attention weights (histogram).

    Shows how temperature affects the concentration of attention.
    """
    model.eval()

    with torch.no_grad():
        _, attention_weights = model(
            grid,
            temperature=temperature,
            return_attention=True,
        )

    if layer_idx < 0 or layer_idx >= len(attention_weights):
        raise ValueError(
            f"layer_idx={layer_idx} is out of range for "
            f"{len(attention_weights)} transformer layers."
        )

    attn = attention_weights[layer_idx].squeeze(0)  # (num_heads, num_patches, num_patches)
    num_heads = attn.shape[0]

    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for head_idx, ax in enumerate(axes):
        head_attn = attn[head_idx].cpu().numpy().flatten()

        # Plot histogram
        ax.hist(head_attn, bins=50, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Attention Weight", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Head {head_idx} - Distribution (T={temperature:.1f})",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = head_attn.mean()
        max_val = head_attn.max()
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.4f}",
        )
        ax.axvline(
            max_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Max: {max_val:.4f}",
        )
        ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved distribution plot to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_sample_predictions(
    model: torch.nn.Module,
    grid: torch.Tensor,
    temperature: float = 1.0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Visualize model predictions overlaid on input BEV.
    """
    model.eval()

    with torch.no_grad():
        predictions, _ = model(grid, temperature=temperature, return_attention=False)

    # Get predictions
    pred_centers = predictions["pred_centers"].squeeze(0).cpu().numpy()  # (num_queries, 2)
    pred_logits = predictions["pred_logits"].squeeze(0).cpu().numpy()  # (num_queries, num_classes)

    # Get class predictions
    pred_classes = pred_logits.argmax(axis=1)
    pred_probs = np.exp(pred_logits) / np.exp(pred_logits).sum(axis=1, keepdims=True)

    # Get input grid
    input_grid = grid.squeeze().cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(input_grid, cmap="gray", origin="lower", vmin=0, vmax=1)
    plt.title(f"Model Predictions (T={temperature:.1f})", fontsize=16)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.grid(True, alpha=0.3)

    class_names = ["Car", "Pedestrian"]
    colors = ["red", "blue"]

    for i, (center, cls, prob) in enumerate(
        zip(pred_centers, pred_classes, pred_probs)
    ):
        x, y = center * input_grid.shape[0]  # Denormalize
        confidence = prob[cls]

        plt.plot(
            x,
            y,
            "o",
            color=colors[cls],
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=2,
            label=f"{class_names[cls]} ({confidence:.2f})" if i < 3 else "",
        )

    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved prediction visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def main() -> None:
    """Main visualization function."""
    print("BEV Transformer Attention Visualizer")
    print("=" * 60)

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load trained model  ---> use BEST model
    checkpoint_path = output_dir / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please run train.py first to train the model.")
        return

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    # Create model (match training config, including dropout)
    model = BEVTransformer(
        grid_size=config["grid_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ffn_dim=config["ffn_dim"],
        num_queries=config["num_queries"],
        num_classes=config["num_classes"],
        dropout=config.get("dropout", 0.1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")

    # Generate test sample
    print("\nGenerating test sample...")
    grids, labels = generate_synthetic_bev(
        n_samples=5,
        grid_size=config["grid_size"],
        seed=123,
    )

    # Select a sample with multiple objects (if possible)
    sample_idx = 0
    for i, label_list in enumerate(labels):
        if len(label_list) >= 2:
            sample_idx = i
            break

    test_grid = (
        torch.from_numpy(grids[sample_idx : sample_idx + 1])
        .float()
        .unsqueeze(1)  # (1,1,H,W)
    )
    print(f"Using sample {sample_idx} with {len(labels[sample_idx])} objects")

    # Visualize with standard temperature
    print("\nGenerating attention visualizations...")

    print("  - Layer 0 attention maps...")
    plot_attention_map(
        model,
        test_grid,
        temperature=1.0,
        layer_idx=0,
        save_path=output_dir / "attention_layer0.png",
    )

    print("  - Layer 1 attention maps...")
    plot_attention_map(
        model,
        test_grid,
        temperature=1.0,
        layer_idx=1,
        save_path=output_dir / "attention_layer1.png",
    )

    print("  - Attention distribution...")
    plot_attention_distribution(
        model,
        test_grid,
        temperature=1.0,
        layer_idx=0,
        save_path=output_dir / "attention_distribution.png",
    )

    print("  - Model predictions...")
    visualize_sample_predictions(
        model,
        test_grid,
        temperature=1.0,
        save_path=output_dir / "model_predictions.png",
    )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Outputs saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    # Optional: try to ensure scipy is available
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("Warning: scipy not found. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])

    main()
