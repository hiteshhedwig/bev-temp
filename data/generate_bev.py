"""
Synthetic BEV (Bird's Eye View) data generator for transformer attention visualization.

This module generates simple synthetic BEV grids with rectangular objects representing
cars and pedestrians. Used for educational purposes to demonstrate attention mechanisms.
"""

from typing import Tuple, List, Dict
import random

import matplotlib.pyplot as plt
import numpy as np


def generate_synthetic_bev(
    n_samples: int = 100,
    grid_size: int = 32,
    min_objects: int = 1,
    max_objects: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, List[List[Dict]]]:
    """
    Generate synthetic BEV grids with random objects.

    Args:
        n_samples: Number of samples to generate
        grid_size: Size of the square BEV grid (default 32x32)
        min_objects: Minimum number of objects per grid
        max_objects: Maximum number of objects per grid
        seed: Random seed for reproducibility

    Returns:
        grids:
            Array of shape (n_samples, grid_size, grid_size) with values in [0, 1]
        labels:
            List of lists, where each inner list contains dicts with keys:
                - 'x', 'y': center coordinates normalized to [0, 1]
                - 'class': 0 = car, 1 = pedestrian

            Objects in each sample are sorted by x-coordinate (left to right)
            so that label slot 0 corresponds to the leftmost object, etc.
    """
    np.random.seed(seed)
    random.seed(seed)

    grids = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    labels: List[List[Dict]] = []

    # Object dimensions: (height, width)
    object_specs = {
        "car": {"size": (5, 3), "class_id": 0, "intensity": 1.0},
        "pedestrian": {"size": (2, 2), "class_id": 1, "intensity": 0.8},
    }

    for i in range(n_samples):
        sample_labels: List[Dict] = []
        n_objects = np.random.randint(min_objects, max_objects + 1)

        for _ in range(n_objects):
            # Randomly choose object type
            obj_type = random.choice(["car", "pedestrian"])
            spec = object_specs[obj_type]
            h, w = spec["size"]

            # Random position ensuring object fits in grid
            # Leave margin to avoid edge artifacts
            margin = 2
            max_y = grid_size - h - margin
            max_x = grid_size - w - margin

            if max_y <= margin or max_x <= margin:
                continue

            top = np.random.randint(margin, max_y)
            left = np.random.randint(margin, max_x)

            # Calculate center coordinates (grid indices)
            center_y = top + h // 2
            center_x = left + w // 2

            # Place object on grid with some intensity variation
            intensity = spec["intensity"] + np.random.uniform(-0.1, 0.1)
            intensity = np.clip(intensity, 0.5, 1.0)

            # Avoid heavy overlap with existing objects
            patch = grids[i, top : top + h, left : left + w]
            if np.mean(patch) < 0.3:
                grids[i, top : top + h, left : left + w] = intensity

                sample_labels.append(
                    {
                        "x": center_x / grid_size,  # Normalize to [0, 1]
                        "y": center_y / grid_size,
                        "class": spec["class_id"],
                    }
                )

        # --- NEW: sort objects left-to-right for consistent slot ordering ---
        if sample_labels:
            sample_labels.sort(key=lambda obj: obj["x"])

        labels.append(sample_labels)

    return grids, labels


def visualize_bev_samples(
    grids,
    labels,
    n_samples = 3,
    save_path = None) :
    """
    Visualize BEV grid samples with labeled object centers.

    Args:
        grids: Array of BEV grids
        labels: List of label dictionaries
        n_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    n_samples = min(n_samples, len(grids))
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))

    if n_samples == 1:
        axes = [axes]

    class_names = ["Car", "Pedestrian"]
    colors = ["red", "blue"]

    for idx, ax in enumerate(axes):
        # Display grid
        ax.imshow(grids[idx], cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_title(f"Sample {idx + 1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Plot object centers
        for obj in labels[idx]:
            x = obj["x"] * grids.shape[1]
            y = obj["y"] * grids.shape[2]
            class_id = obj["class"]

            ax.plot(
                x,
                y,
                "o",
                color=colors[class_id],
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=2,
                label=class_names[class_id],
            )

        # Remove duplicate labels
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def print_dataset_statistics(grids: np.ndarray, labels: List[List[Dict]]) -> None:
    """Print statistics about the generated dataset."""
    n_objects_per_sample = [len(sample_labels) for sample_labels in labels]
    class_counts = {0: 0, 1: 0}

    for sample_labels in labels:
        for obj in sample_labels:
            class_counts[obj["class"]] += 1

    total_objects = sum(n_objects_per_sample)

    print("=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total samples: {len(grids)}")
    print(f"Grid size: {grids.shape[1]}x{grids.shape[2]}")
    print(
        "Objects per sample: "
        f"{np.mean(n_objects_per_sample):.2f} ± {np.std(n_objects_per_sample):.2f}"
    )
    print(f"Total objects: {total_objects}")
    if total_objects > 0:
        print(
            f"  - Cars: {class_counts[0]} "
            f"({class_counts[0] / total_objects * 100:.1f}%)"
        )
        print(
            f"  - Pedestrians: {class_counts[1]} "
            f"({class_counts[1] / total_objects * 100:.1f}%)"
        )
    print(f"Grid value range: [{grids.min():.3f}, {grids.max():.3f}]")
    print("=" * 60)


if __name__ == "__main__":
    """Test data generation with visualization."""
    print("Generating synthetic BEV data...")

    # Generate samples
    grids, labels = generate_synthetic_bev(
        n_samples=10,
        grid_size=32,
        min_objects=1,
        max_objects=3,
        seed=42,
    )

    # Print statistics
    print_dataset_statistics(grids, labels)

    # Visualize examples
    print("\nVisualizing 3 samples...")
    visualize_bev_samples(
        grids,
        labels,
        n_samples=3,
        save_path="../outputs/bev_samples.png",
    )

    print("\nData generation test completed successfully!")
