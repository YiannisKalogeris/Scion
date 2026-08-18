"""
Dataset generation and loading for diffeomorphism network training.

This module provides:
    - DiffeomorphismDataset: Generates (image, points, targets) tuples
    - DataLoader: PyTorch-compatible batching iterator
    - Dataset generation and caching utilities
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Iterator, Dict, Any
from dataclasses import dataclass
import pickle
from tqdm import tqdm


from scion.geometry.shapes import ShapeGenerator, ClosedCurve, SHAPE_CLASSES
from scion.geometry.sdf import SDFRenderer, sample_points_in_shape
from scion.geometry.harmonic_maps import HarmonicMapSolver, interpolate_harmonic_map


@dataclass
class DataSample:
    """A single training sample."""
    image: np.ndarray           # (H, W, 1) SDF image
    interior_points: np.ndarray  # (N, 2) points inside domain
    interior_targets: np.ndarray # (N, 2) ground truth map values
    boundary_points: np.ndarray  # (M, 2) points on boundary
    boundary_targets: np.ndarray # (M, 2) should map to square boundary


class DiffeomorphismDataset:
    """
    Dataset for training diffeomorphism networks.
    
    Each sample contains:
        - SDF image of the shape
        - Interior points and their ground truth mapped locations
        - Boundary points and their ground truth mapped locations
    """
    
    def __init__(
        self,
        n_shapes: int = 1000,
        image_size: int = 64,
        n_interior_points: int = 256,
        n_boundary_points: int = 128,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        harmonic_grid_size: int = 64,
        shape_distribution: Optional[dict] = None,
        seed: int = 42,
    ):
        """
        Initialize dataset.
        
        Args:
            n_shapes: Number of shapes to generate
            image_size: SDF image resolution
            n_interior_points: Points sampled inside each shape
            n_boundary_points: Points sampled on boundary
            bounds: Domain bounds
            harmonic_grid_size: Grid size for harmonic map solver
            shape_distribution: Dict mapping shape names to weights.
                Example: {"Ellipse": 0.5, "Superellipse": 0.5}
                Example: {"Ellipse": 1.0}  # Only ellipses
                If None, uniform distribution over all shapes.
            seed: Random seed
        """
        self.n_shapes = n_shapes
        self.image_size = image_size
        self.n_interior_points = n_interior_points
        self.n_boundary_points = n_boundary_points
        self.bounds = bounds
        self.harmonic_grid_size = harmonic_grid_size
        self.shape_distribution = shape_distribution
        self.seed = seed
        
        # Initialize components
        self.shape_generator = ShapeGenerator(
            seed=seed,
            shape_distribution=shape_distribution,
        )
        self.sdf_renderer = SDFRenderer(
            grid_size=image_size,
            bounds=bounds,
            normalize=True,
        )
        self.harmonic_solver = HarmonicMapSolver(
            grid_size=harmonic_grid_size,
            bounds=bounds,
        )
        
        # Storage
        self.samples = []
        self.shapes = []
    
    def _compute_boundary_targets(self, t: np.ndarray) -> np.ndarray:
        """
        Compute boundary targets directly from parameter t.
        
        Maps t ∈ [0, 2π) to points on square boundary [-1, 1]².
        Traverses counterclockwise to match orientation-preserving harmonic maps.
        
        Args:
            t: (N,) array of parameters in [0, 2π)
        Returns:
            targets: (N, 2) points on square boundary
        """
        # Normalize to [0, 1)
        t_normalized = (t / (2 * np.pi)) % 1.0
        
        # Map to square perimeter (total length = 8)
        s = t_normalized * 8
        
        targets = np.zeros((len(t), 2), dtype=np.float32)
        
        for i, si in enumerate(s):
            # Counterclockwise from (1, -1):
            # Right edge UP, Top edge LEFT, Left edge DOWN, Bottom edge RIGHT
            if si < 2:      # Right edge: (1, -1) to (1, 1)
                targets[i] = [1.0, -1.0 + si]
            elif si < 4:    # Top edge: (1, 1) to (-1, 1)
                targets[i] = [1.0 - (si - 2), 1.0]
            elif si < 6:    # Left edge: (-1, 1) to (-1, -1)
                targets[i] = [-1.0, 1.0 - (si - 4)]
            else:           # Bottom edge: (-1, -1) to (1, -1)
                targets[i] = [-1.0 + (si - 6), -1.0]
        
        return targets
    
    def generate(self, show_progress: bool = True):
        """Generate all samples."""
        self.samples = []
        self.shapes = []
        
        rng = np.random.default_rng(self.seed)
        
        # Print distribution info
        print(f"Shape distribution: {self.shape_generator.get_distribution_summary()}")
        
        iterator = range(self.n_shapes)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating dataset")
        
        for _ in iterator:
            try:
                sample, shape = self._generate_single_sample(rng)
                self.samples.append(sample)
                self.shapes.append(shape)
            except Exception as e:
                # Skip problematic shapes (e.g., too small)
                print(f"Warning: Skipping shape due to error: {e}")
                continue
    
    def _generate_single_sample(
        self, 
        rng: np.random.Generator
    ) -> Tuple[DataSample, ClosedCurve]:
        """Generate a single training sample."""
        # Generate random shape
        shape = self.shape_generator.generate()
        
        # Render to SDF image
        image = self.sdf_renderer.render(shape)
        
        # Compute harmonic map (ground truth)
        self.harmonic_solver.solve(shape)
        
        # Sample interior points — oversample and retry to guarantee
        # exactly n_interior_points after NaN filtering from harmonic
        # map interpolation (points near the boundary often fail).
        n_target = self.n_interior_points
        interior_points = np.empty((0, 2), dtype=np.float32)
        interior_targets = np.empty((0, 2), dtype=np.float32)
        attempts = 0
        max_attempts = 10

        while len(interior_points) < n_target and attempts < max_attempts:
            n_needed = n_target - len(interior_points)
            # Request extra to compensate for NaN losses (~30% margin)
            n_request = int(n_needed * 1.3) + 10

            candidates = sample_points_in_shape(
                shape, n_request, self.bounds, rng
            )
            candidate_targets = self.harmonic_solver.evaluate(candidates)

            # Keep only valid (non-NaN) points
            valid = ~np.any(np.isnan(candidate_targets), axis=1)
            good_pts = candidates[valid]
            good_tgt = candidate_targets[valid]

            # Append only as many as we still need
            n_take = min(len(good_pts), n_needed)
            interior_points = np.concatenate([interior_points, good_pts[:n_take]])
            interior_targets = np.concatenate([interior_targets, good_tgt[:n_take]])
            attempts += 1

        if len(interior_points) < n_target // 4:
            raise ValueError(
                f"Could not get enough valid interior points: "
                f"{len(interior_points)}/{n_target} after {max_attempts} attempts"
            )
        
        # Sample boundary points
        t = np.linspace(0, 2*np.pi, self.n_boundary_points, endpoint=False)
        boundary_points = shape(t)
        
        # Boundary targets: compute directly from parametric correspondence
        # Instead of interpolating (which fails at boundaries), we use the
        # fact that t ∈ [0, 2π) corresponds to a point on the square boundary
        boundary_targets = self._compute_boundary_targets(t)
        
        # No need to filter NaN since we're computing directly
        # But filter any invalid points just in case
        valid_mask = ~np.any(np.isnan(boundary_points), axis=1)
        if not np.all(valid_mask):
            boundary_points = boundary_points[valid_mask]
            boundary_targets = boundary_targets[valid_mask]
        
        return DataSample(
            image=image.astype(np.float32),
            interior_points=interior_points.astype(np.float32),
            interior_targets=interior_targets.astype(np.float32),
            boundary_points=boundary_points.astype(np.float32),
            boundary_targets=boundary_targets.astype(np.float32),
        ), shape
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> DataSample:
        return self.samples[idx]
    
    def save(self, path: str | Path):
        """Save dataset to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'samples': self.samples,
            'config': {
                'n_shapes': self.n_shapes,
                'image_size': self.image_size,
                'n_interior_points': self.n_interior_points,
                'n_boundary_points': self.n_boundary_points,
                'bounds': self.bounds,
                'harmonic_grid_size': self.harmonic_grid_size,
                'shape_distribution': self.shape_distribution,
                'seed': self.seed,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved dataset with {len(self.samples)} samples to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> 'DiffeomorphismDataset':
        """Load dataset from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        config = data['config']
        dataset = cls(**config)
        dataset.samples = data['samples']
        
        print(f"Loaded dataset with {len(dataset.samples)} samples from {path}")
        return dataset


class DataLoader:
    """
    PyTorch-compatible data loader with batching and shuffling.
    Returns numpy arrays (convert to tensors in training loop).
    """
    
    def __init__(
        self,
        dataset: DiffeomorphismDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        
        self._indices = np.arange(len(dataset))
    
    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over batches."""
        indices = self._indices.copy()
        
        if self.shuffle:
            self.rng.shuffle(indices)
        
        n_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            n_batches += 1
        
        for i in range(n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(indices))
            batch_indices = indices[start:end]
            
            yield self._collate_batch(batch_indices)
    
    def _collate_batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Collate samples into a batch."""
        samples = [self.dataset[i] for i in indices]
        
        # Find minimum sizes for this batch (due to filtering)
        min_interior = min(len(s.interior_points) for s in samples)
        min_boundary = min(len(s.boundary_points) for s in samples)
        
        # Stack with truncation to minimum size (using numpy)
        batch = {
            'image': np.stack([np.array(s.image, dtype=np.float32) for s in samples]),
            'interior_points': np.stack([
                np.array(s.interior_points[:min_interior], dtype=np.float32) for s in samples
            ]),
            'interior_targets': np.stack([
                np.array(s.interior_targets[:min_interior], dtype=np.float32) for s in samples
            ]),
            'boundary_points': np.stack([
                np.array(s.boundary_points[:min_boundary], dtype=np.float32) for s in samples
            ]),
            'boundary_targets': np.stack([
                np.array(s.boundary_targets[:min_boundary], dtype=np.float32) for s in samples
            ]),
        }
        
        return batch
    
    def __len__(self):
        """Number of batches per epoch."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def generate_datasets(
    data_dir: str | Path = "./data",
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 500,
    shape_distribution: Optional[dict] = None,
    **kwargs,
):
    """
    Generate and save train/val/test datasets.
    
    Args:
        data_dir: Directory to save datasets
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        shape_distribution: Dict mapping shape names to weights.
            Example: {"Ellipse": 0.5, "Superellipse": 0.5}
            Example: {"Ellipse": 1.0}  # Only ellipses
            If None, uniform distribution over all shapes.
        **kwargs: Additional arguments passed to DiffeomorphismDataset
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Training set
    print("Generating training set...")
    train_dataset = DiffeomorphismDataset(
        n_shapes=n_train, 
        seed=42, 
        shape_distribution=shape_distribution,
        **kwargs
    )
    train_dataset.generate()
    train_dataset.save(data_dir / "train.pkl")
    
    # Validation set
    print("\nGenerating validation set...")
    val_dataset = DiffeomorphismDataset(
        n_shapes=n_val, 
        seed=123, 
        shape_distribution=shape_distribution,
        **kwargs
    )
    val_dataset.generate()
    val_dataset.save(data_dir / "val.pkl")
    
    # Test set
    print("\nGenerating test set...")
    test_dataset = DiffeomorphismDataset(
        n_shapes=n_test, 
        seed=456, 
        shape_distribution=shape_distribution,
        **kwargs
    )
    test_dataset.generate()
    test_dataset.save(data_dir / "test.pkl")
    
    print(f"\nDatasets saved to {data_dir}")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Generate small test dataset
    print("Generating test datasets...")
    
    train_ds, val_ds, test_ds = generate_datasets(
        data_dir="./data/test_run",
        n_train=100,
        n_val=20,
        n_test=20,
        image_size=64,
        n_interior_points=128,
        n_boundary_points=64,
    )
    
    # Test data loader
    loader = DataLoader(train_ds, batch_size=8)
    
    print(f"\nDataLoader test:")
    print(f"  Number of batches: {len(loader)}")
    
    for batch in loader:
        print(f"  Batch shapes:")
        for key, val in batch.items():
            print(f"    {key}: {val.shape}")
        break
    
    print("\nDataset generation complete!")