"""
Temperature study for BEV Transformer attention.

This script:
- Loads the trained BEV Transformer (best_model.pt if available, else final_model.pt)
- Runs inference for different temperatures T in [0.1, 0.5, 1.0, 2.0, 5.0]
- Computes:
    * Attention entropy, max weight, mean weight, sparsity (from layer 0)
    * Classification accuracy (per object)
    * Detection accuracy (correct class AND center within a distance threshold)
    * Mean center error
- Produces:
    * A comparison figure showing input + attention overlay for each T
    * A metrics figure: entropy vs T, accuracy vs T, center error vs T

Designed to run quickly on CPU or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Make local packages importable
sys.path.append("data")
sys.path.append("model")

from data.generate_bev import generate_synthetic_bev  # type: ignore
from data.dataset import BEVDataset  # type: ignore
from model.transformer import BEVTransformer  # type: ignore
from model.attention import compute_attention_statistics  # type: ignore


TEMPERATURES: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_temperature(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float,
    layer_idx: int = 0,
    loc_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Evaluate model at a specific temperature.

    Args:
        model: Trained BEV Transformer
        dataloader: DataLoader over BEVDataset (max_objects=3)
        device: CPU or CUDA device
        temperature: Softmax temperature
        layer_idx: Which transformer layer's attention to analyze (default: 0)
        loc_threshold: Max L2 distance in normalized coords for a "correct" detection

    Returns:
        Dictionary with aggregated metrics:
            - 'entropy'
            - 'max_weight'
            - 'mean_weight'
            - 'sparsity'
            - 'class_acc'
            - 'det_acc'
            - 'center_mae'
    """
    model.eval()

    entropies: List[float] = []
    max_weights: List[float] = []
    mean_weights: List[float] = []
    sparsities: List[float] = []

    total_valid_objects = 0
    correct_classes = 0
    correct_detections = 0
    center_error_sum = 0.0

    with torch.no_grad():
        for grids, targets in dataloader:
            grids = grids.to(device)
            for key in targets:
                if isinstance(targets[key], torch.Tensor):
                    targets[key] = targets[key].to(device)

            predictions, attention_weights = model(
                grids,
                temperature=temperature,
                return_attention=True,
            )

            # -------------------------
            # Detection metrics
            # -------------------------
            pred_centers = predictions["pred_centers"]  # (B, Q, 2)
            pred_logits = predictions["pred_logits"]    # (B, Q, C)

            target_centers = targets["centers"]         # (B, Q, 2)
            target_classes = targets["classes"]         # (B, Q)
            valid_mask = targets["valid_mask"]          # (B, Q)

            pred_classes = pred_logits.argmax(dim=-1)   # (B, Q)

            valid_mask_float = valid_mask.float()
            num_valid = int(valid_mask.sum().item())
            total_valid_objects += num_valid

            # Classification accuracy (per valid object)
            correct_classes_batch = (
                (pred_classes == target_classes) & valid_mask
            ).sum().item()
            correct_classes += correct_classes_batch

            # Center error (L2 in normalized coords)
            dists = torch.norm(pred_centers - target_centers, dim=-1)  # (B, Q)
            center_error_sum += float((dists * valid_mask_float).sum().item())

            # Detection accuracy: correct class AND within distance threshold
            correct_det_batch = (
                (dists <= loc_threshold)
                & (pred_classes == target_classes)
                & valid_mask
            ).sum().item()
            correct_detections += correct_det_batch

            # -------------------------
            # Attention statistics
            # -------------------------
            # Use attention from a specific layer (e.g., layer 0)
            # Shape: (B, num_heads, num_patches, num_patches)
            layer_attn = attention_weights[layer_idx]

            stats = compute_attention_statistics(layer_attn)
            entropies.append(stats["entropy"])
            max_weights.append(stats["max_weight"])
            mean_weights.append(stats["mean_weight"])
            sparsities.append(stats["sparsity"])

    # Avoid division by zero
    total_valid_objects = max(total_valid_objects, 1)

    metrics = {
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "max_weight": float(np.mean(max_weights)) if max_weights else 0.0,
        "mean_weight": float(np.mean(mean_weights)) if mean_weights else 0.0,
        "sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
        "class_acc": correct_classes / total_valid_objects,
        "det_acc": correct_detections / total_valid_objects,
        "center_mae": center_error_sum / total_valid_objects,
    }

    return metrics


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------

def plot_temperature_metrics(
    temperatures: List[float],
    metrics_per_temp: Dict[float, Dict[str, float]],
    save_path: Path,
) -> None:
    """
    Plot entropy vs T, accuracy vs T, and center error vs T.

    Args:
        temperatures: List of temperatures
        metrics_per_temp: Dict mapping T -> metrics dict
        save_path: Where to save PNG (PDF will also be saved)
    """
    entropies = [metrics_per_temp[T]["entropy"] for T in temperatures]
    class_accs = [metrics_per_temp[T]["class_acc"] for T in temperatures]
    det_accs = [metrics_per_temp[T]["det_acc"] for T in temperatures]
    center_mae = [metrics_per_temp[T]["center_mae"] for T in temperatures]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Entropy vs T
    axes[0].plot(temperatures, entropies, marker="o")
    axes[0].set_xlabel("Temperature T")
    axes[0].set_ylabel("Attention Entropy")
    axes[0].set_title("Entropy vs Temperature")
    axes[0].grid(True, alpha=0.3)

    # Accuracy vs T
    axes[1].plot(temperatures, class_accs, marker="o", label="Class Acc")
    axes[1].plot(temperatures, det_accs, marker="s", label="Det. Acc")
    axes[1].set_xlabel("Temperature T")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Detection Accuracy vs Temperature")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Center error vs T
    axes[2].plot(temperatures, center_mae, marker="o")
    axes[2].set_xlabel("Temperature T")
    axes[2].set_ylabel("Mean Center Error (L2, norm.)")
    axes[2].set_title("Localization Error vs Temperature")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved temperature metrics figure to {save_path}")

    pdf_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved PDF version to {pdf_path}")

    plt.close()


def plot_temperature_comparison(
    model: torch.nn.Module,
    device: torch.device,
    grid: np.ndarray,
    temperatures: List[float],
    layer_idx: int,
    head_idx: int,
    save_path: Path,
) -> None:
    """
    Create a qualitative comparison figure:
    For each T, show (input BEV, attention overlay) in a 5×2 grid.

    Args:
        model: Trained BEV Transformer
        device: Device
        grid: Single BEV grid (H, W) as numpy array
        temperatures: List of temperatures
        layer_idx: Transformer layer index whose attention to visualize
        head_idx: Attention head index
        save_path: Path to save PNG (PDF also saved)
    """
    from scipy.ndimage import zoom  # local import

    model.eval()

    H, W = grid.shape
    input_grid = grid

    fig, axes = plt.subplots(
        nrows=len(temperatures),
        ncols=2,
        figsize=(10, 4 * len(temperatures)),
    )

    if len(temperatures) == 1:
        axes = np.array([axes])  # make indexable as [row, col]

    grid_tensor = (
        torch.from_numpy(input_grid[None, None, ...])
        .float()
        .to(device)
    )  # (1, 1, H, W)

    with torch.no_grad():
        for row, T in enumerate(temperatures):
            preds, attn_list = model(
                grid_tensor,
                temperature=T,
                return_attention=True,
            )

            # Attention: (1, num_heads, num_patches, num_patches)
            attn_layer = attn_list[layer_idx]
            attn_head = attn_layer[0, head_idx]  # (num_patches, num_patches)

            num_patches = attn_head.shape[-1]
            patch_per_side = int(np.sqrt(num_patches))

            # Average over query positions to get a single map over keys
            avg_attn = attn_head.mean(dim=0).cpu().numpy()  # (num_patches,)
            attn_grid = avg_attn.reshape(patch_per_side, patch_per_side)

            # --- Left column: input BEV ---
            ax_in = axes[row, 0]
            ax_in.imshow(input_grid, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax_in.set_title(f"T={T:.1f} - Input BEV", fontsize=14)
            ax_in.set_xlabel("X")
            ax_in.set_ylabel("Y")
            ax_in.grid(True, alpha=0.3)

            # --- Right column: overlay ---
            ax_ov = axes[row, 1]
            ax_ov.imshow(input_grid, cmap="gray", origin="lower", vmin=0, vmax=1, alpha=0.6)

            scale = H / attn_grid.shape[0]
            attn_resized = zoom(attn_grid, scale, order=1)
            im = ax_ov.imshow(attn_resized, cmap="viridis", origin="lower", alpha=0.5)

            ax_ov.set_title(f"T={T:.1f} - Attention Overlay", fontsize=14)
            ax_ov.set_xlabel("X")
            ax_ov.set_ylabel("Y")
            ax_ov.grid(True, alpha=0.3)

    # Add a single colorbar on the right for the last overlay
    fig.colorbar(im, ax=axes[:, 1], fraction=0.02, pad=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved temperature comparison figure to {save_path}")

    pdf_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved PDF version to {pdf_path}")

    plt.close()


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def load_model_and_config(output_dir: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """
    Load BEVTransformer and config from best_model.pt (if available) or final_model.pt.
    """
    best_path = output_dir / "best_model.pt"
    final_path = output_dir / "final_model.pt"

    if best_path.exists():
        checkpoint_path = best_path
        print(f"Using best model: {checkpoint_path}")
    elif final_path.exists():
        checkpoint_path = final_path
        print(
            f"Warning: best_model.pt not found. "
            f"Falling back to final model: {final_path}"
        )
    else:
        raise FileNotFoundError(
            f"No model checkpoint found in {output_dir} "
            f"(expected best_model.pt or final_model.pt)."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

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
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint['epoch']})")

    return model, config


def main() -> None:
    print("BEV Transformer Temperature Study")
    print("=" * 60)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, config = load_model_and_config(output_dir, device)

    # ----------------------------------------------------------------------
    # Build evaluation dataset (new synthetic data to probe temperature)
    # ----------------------------------------------------------------------
    print("\nGenerating evaluation dataset...")
    eval_grids, eval_labels = generate_synthetic_bev(
        n_samples=200,
        grid_size=config["grid_size"],
        seed=999,
    )

    eval_dataset = BEVDataset(eval_grids, eval_labels, max_objects=config["num_queries"])
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Eval dataset size: {len(eval_dataset)} samples")

    # ----------------------------------------------------------------------
    # Quantitative study across temperatures
    # ----------------------------------------------------------------------
    print("\nRunning temperature sweep...")
    metrics_per_temp: Dict[float, Dict[str, float]] = {}

    for T in TEMPERATURES:
        print(f"  Evaluating T={T} ...")
        metrics = evaluate_temperature(
            model=model,
            dataloader=eval_loader,
            device=device,
            temperature=T,
            layer_idx=0,
            loc_threshold=0.1,
        )
        metrics_per_temp[T] = metrics
        print(
            f"    Entropy={metrics['entropy']:.4f}, "
            f"MaxW={metrics['max_weight']:.4f}, "
            f"ClassAcc={metrics['class_acc']:.3f}, "
            f"DetAcc={metrics['det_acc']:.3f}, "
            f"CenterMAE={metrics['center_mae']:.3f}"
        )

    # Plot metrics vs temperature
    metrics_path = output_dir / "temperature_metrics.png"
    plot_temperature_metrics(TEMPERATURES, metrics_per_temp, metrics_path)

    # ----------------------------------------------------------------------
    # Qualitative comparison: single sample with multiple temperatures
    # ----------------------------------------------------------------------
    print("\nGenerating qualitative comparison figure...")
    # Use a sample with at least 2 objects if possible
    sample_idx = 0
    for i, sample_labels in enumerate(eval_labels):
        if len(sample_labels) >= 2:
            sample_idx = i
            break

    sample_grid = eval_grids[sample_idx]  # (H, W)
    print(f"Using eval sample {sample_idx} with {len(eval_labels[sample_idx])} objects")

    comparison_path = output_dir / "temperature_comparison.png"
    plot_temperature_comparison(
        model=model,
        device=device,
        grid=sample_grid,
        temperatures=TEMPERATURES,
        layer_idx=0,
        head_idx=0,
        save_path=comparison_path,
    )

    print("\n" + "=" * 60)
    print("Temperature study complete!")
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    # Ensure scipy is available for zoom in comparison plots
    try:
        import scipy  # type: ignore  # noqa: F401
    except ImportError:
        print("Warning: scipy not found. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])

    main()
