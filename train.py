"""
Training script for BEV Transformer.

Trains a tiny transformer model on synthetic BEV data for object detection.
Designed to run quickly on CPU (<5 minutes).
"""

import sys
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths for imports
sys.path.append("data")
sys.path.append("model")

from data.generate_bev import generate_synthetic_bev  # type: ignore
from data.dataset import create_dataloaders  # type: ignore
from model.transformer import BEVTransformer  # type: ignore


class DetectionLoss(nn.Module):
    """
    Combined loss for object detection.

    Combines:
        - MSE loss for center coordinate regression
        - Cross-entropy loss for classification
    """

    def __init__(self, center_weight: float = 1.0, class_weight: float = 1.0) -> None:
        """
        Initialize detection loss.

        Args:
            center_weight: Weight for center prediction loss
            class_weight: Weight for classification loss
        """
        super().__init__()
        self.center_weight = center_weight
        self.class_weight = class_weight
        self.mse_loss = nn.MSELoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            predictions: Model predictions with 'pred_centers' and 'pred_logits'
            targets: Ground truth with 'centers', 'classes', 'valid_mask'

        Returns:
            Dictionary with total loss and individual components
        """
        pred_centers = predictions["pred_centers"]  # (batch, num_queries, 2)
        pred_logits = predictions["pred_logits"]  # (batch, num_queries, num_classes)

        target_centers = targets["centers"]  # (batch, num_queries, 2)
        target_classes = targets["classes"]  # (batch, num_queries)
        valid_mask = targets["valid_mask"]  # (batch, num_queries)

        # Center loss (only for valid objects)
        center_loss = self.mse_loss(pred_centers, target_centers)  # (B, Q, 2)
        center_loss = center_loss.mean(dim=-1)  # (B, Q)
        center_loss = (center_loss * valid_mask.float()).sum() / (
            valid_mask.sum() + 1e-6
        )

        # Classification loss (only for valid objects)
        batch_size, num_queries = pred_logits.shape[:2]
        pred_logits_flat = pred_logits.view(-1, pred_logits.size(-1))
        target_classes_flat = target_classes.view(-1)

        class_loss = self.ce_loss(pred_logits_flat, target_classes_flat)
        class_loss = class_loss.view(batch_size, num_queries)
        class_loss = (class_loss * valid_mask.float()).sum() / (
            valid_mask.sum() + 1e-6
        )

        # Total loss
        total_loss = self.center_weight * center_loss + self.class_weight * class_loss

        return {
            "total": total_loss,
            "center": center_loss,
            "class": class_loss,
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    center_loss_sum = 0.0
    class_loss_sum = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for grids, targets in pbar:
        grids = grids.to(device)
        for key in targets:
            if isinstance(targets[key], torch.Tensor):
                targets[key] = targets[key].to(device)

        optimizer.zero_grad()
        predictions, _ = model(grids, temperature=1.0, return_attention=False)

        loss_dict = criterion(predictions, targets)
        loss = loss_dict["total"]

        loss.backward()
        # optional: gradient clipping (helps with spikes)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        center_loss_sum += loss_dict["center"].item()
        class_loss_sum += loss_dict["class"].item()

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "center": f"{loss_dict['center'].item():.4f}",
                "class": f"{loss_dict['class'].item():.4f}",
            }
        )

    n_batches = len(dataloader)
    return {
        "total": total_loss / n_batches,
        "center": center_loss_sum / n_batches,
        "class": class_loss_sum / n_batches,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    center_loss_sum = 0.0
    class_loss_sum = 0.0

    with torch.no_grad():
        for grids, targets in dataloader:
            grids = grids.to(device)
            for key in targets:
                if isinstance(targets[key], torch.Tensor):
                    targets[key] = targets[key].to(device)

            predictions, _ = model(grids, temperature=1.0, return_attention=False)

            loss_dict = criterion(predictions, targets)

            total_loss += loss_dict["total"].item()
            center_loss_sum += loss_dict["center"].item()
            class_loss_sum += loss_dict["class"].item()

    n_batches = len(dataloader)
    return {
        "total": total_loss / n_batches,
        "center": center_loss_sum / n_batches,
        "class": class_loss_sum / n_batches,
    }


def plot_training_curves(history: Dict[str, list], save_path: Path) -> None:
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history["train_total"]) + 1)

    axes[0].plot(epochs, history["train_total"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_total"], "r-", label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_center"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_center"], "r-", label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Center Loss")
    axes[1].set_title("Center Prediction Loss (MSE)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_class"], "b-", label="Train", linewidth=2)
    axes[2].plot(epochs, history["val_class"], "r-", label="Val", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Class Loss")
    axes[2].set_title("Classification Loss (CE)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved training curves to {save_path}")
    plt.close()


def main() -> None:
    """Main training function."""
    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "n_samples": 1000,
        "grid_size": 32,
        "patch_size": 4,
        "embed_dim": 64,
        "num_heads": 2,
        "num_layers": 2,
        "ffn_dim": 128,
        "num_queries": 3,
        "num_classes": 2,
        "batch_size": 8,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "train_ratio": 0.8,
        "dropout": 0.3,
    }

    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # Data
    print("Generating synthetic BEV data...")
    grids, labels = generate_synthetic_bev(
        n_samples=config["n_samples"],
        grid_size=config["grid_size"],
        seed=42,
    )

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        grids,
        labels,
        train_ratio=config["train_ratio"],
        batch_size=config["batch_size"],
        seed=42,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Model
    print("\nInitializing model...")
    model = BEVTransformer(
        grid_size=config["grid_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ffn_dim=config["ffn_dim"],
        num_queries=config["num_queries"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    print(f"Model parameters: {model.get_num_parameters():,}")

    criterion = DetectionLoss(center_weight=1.0, class_weight=1.0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4,
    )

    history: Dict[str, list] = {
        "train_total": [],
        "train_center": [],
        "train_class": [],
        "val_total": [],
        "val_center": [],
        "val_class": [],
    }

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    start_time = time.time()
    best_val_loss = float("inf")
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, config["num_epochs"] + 1):
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_losses = validate(model, val_loader, criterion, device)

        history["train_total"].append(train_losses["total"])
        history["train_center"].append(train_losses["center"])
        history["train_class"].append(train_losses["class"])
        history["val_total"].append(val_losses["total"])
        history["val_center"].append(val_losses["center"])
        history["val_class"].append(val_losses["class"])

        print(
            f"Epoch {epoch:3d}/{config['num_epochs']} | "
            f"Train Loss: {train_losses['total']:.4f} | "
            f"Val Loss: {val_losses['total']:.4f} | "
            f"Center: {val_losses['center']:.4f} | "
            f"Class: {val_losses['class']:.4f}"
        )

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_losses["total"],
                    "config": config,
                },
                best_model_path,
            )
            print(
                f"> Updated best model at epoch {epoch} "
                f"(val_loss={best_val_loss:.4f}) -> {best_model_path}"
            )

        # Periodic checkpoints
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "config": config,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    training_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(
        f"Total training time: {training_time:.2f} seconds "
        f"({training_time / 60:.2f} minutes)"
    )
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {history['train_total'][-1]:.4f}")
    print(f"Final val loss: {history['val_total'][-1]:.4f}")
    print("=" * 60 + "\n")

    print("Plotting training curves...")
    plot_training_curves(history, output_dir / "training_curves.png")

    # Final model (last epoch)
    torch.save(
        {
            "epoch": config["num_epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        output_dir / "final_model.pt",
    )

    if best_model_path.exists():
        print(f"Best model saved at: {best_model_path.resolve()}")
    else:
        print(
            "Warning: best_model.pt does not exist. "
            "Only final_model.pt and checkpoints are available."
        )

    print(f"\nModels saved to {output_dir}/")
    print("Training script completed successfully!")


if __name__ == "__main__":
    main()
