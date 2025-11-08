"""
PyTorch Dataset class for BEV synthetic data.

Provides a standard PyTorch Dataset interface for the synthetic BEV grids.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BEVDataset(Dataset):
    """
    PyTorch Dataset for synthetic BEV grids.

    Each sample contains:
        - BEV grid: (1, H, W) tensor
        - Target: Dict with 'centers' (N, 2) and 'classes' (N,)
    """

    def __init__(
        self,
        grids: np.ndarray,
        labels: List[List[Dict]],
        max_objects: int = 3,
    ) -> None:
        """
        Initialize dataset.

        Args:
            grids: Array of BEV grids (N, H, W)
            labels: List of label dicts for each sample
            max_objects: Maximum number of objects (for padding)
        """
        self.grids = torch.from_numpy(grids).float()
        self.labels = labels
        self.max_objects = max_objects

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            grid: (1, H, W) tensor
            target: Dict with:
                - 'centers': (max_objects, 2) tensor, padded with zeros
                - 'classes': (max_objects,) tensor, padded with -1
                - 'valid_mask': (max_objects,) boolean tensor indicating valid objects
        """
        # Get grid and add channel dimension
        grid = self.grids[idx].unsqueeze(0)  # (1, H, W)

        # --- Canonical ordering of objects ---------------------------------
        # IMPORTANT: we sort by x (then y) so that:
        #   - slot 0 = leftmost object
        #   - slot 1 = next, etc.
        # This makes the fixed "query slots" compatible with the loss.
        sample_labels = sorted(
            self.labels[idx],
            key=lambda obj: (obj["x"], obj["y"]),
        )
        n_objects = len(sample_labels)

        # Initialize padded tensors
        centers = torch.zeros((self.max_objects, 2), dtype=torch.float32)
        classes = torch.full((self.max_objects,), -1, dtype=torch.long)
        valid_mask = torch.zeros(self.max_objects, dtype=torch.bool)

        # Fill in actual object data
        for i, obj in enumerate(sample_labels):
            if i >= self.max_objects:
                break
            centers[i, 0] = obj["x"]
            centers[i, 1] = obj["y"]
            classes[i] = obj["class"]
            valid_mask[i] = True

        target = {
            "centers": centers,
            "classes": classes,
            "valid_mask": valid_mask,
            "n_objects": n_objects,
        }

        return grid, target


def create_dataloaders(
    grids: np.ndarray,
    labels: List[List[Dict]],
    train_ratio: float = 0.8,
    batch_size: int = 8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        grids: Array of BEV grids
        labels: List of label dicts
        train_ratio: Ratio of data to use for training
        batch_size: Batch size for dataloaders
        seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Shuffle indices
    n_samples = len(grids)
    indices = np.random.permutation(n_samples)

    # Split indices
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create datasets
    train_dataset = BEVDataset(
        grids[train_indices],
        [labels[i] for i in train_indices],
        max_objects=3,
    )

    val_dataset = BEVDataset(
        grids[val_indices],
        [labels[i] for i in val_indices],
        max_objects=3,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for CPU-only systems
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    """Test dataset and dataloader."""
    from generate_bev import generate_synthetic_bev

    print("Testing BEV Dataset...")

    # Generate data
    grids, labels = generate_synthetic_bev(n_samples=20, seed=42)

    # Create dataset
    dataset = BEVDataset(grids, labels, max_objects=3)

    print(f"Dataset size: {len(dataset)}")

    # Test single sample
    grid, target = dataset[0]
    print(f"\nSample 0:")
    print(f"  Grid shape: {grid.shape}")
    print(f"  Centers shape: {target['centers'].shape}")
    print(f"  Classes shape: {target['classes'].shape}")
    print(f"  Valid mask: {target['valid_mask']}")
    print(f"  N objects: {target['n_objects']}")
    print(f"  Centers: {target['centers'][target['valid_mask']]}")
    print(f"  Classes: {target['classes'][target['valid_mask']]}")

    # Test dataloader
    train_loader, val_loader = create_dataloaders(grids, labels, batch_size=4)

    print(f"\nDataLoader test:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Test batch
    for batch_grids, batch_targets in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Grids: {batch_grids.shape}")
        print(f"  Centers: {batch_targets['centers'].shape}")
        print(f"  Classes: {batch_targets['classes'].shape}")
        break

    print("\nDataset test completed successfully!")
