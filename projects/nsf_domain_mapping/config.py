"""
Configuration and hyperparameters for diffeomorphism network training.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DataConfig:
    """Data generation configuration."""
    # Image settings
    image_size: int = 64
    domain_bounds: tuple = (-2.0, 2.0)  # shapes live in [-2, 2]²
    
    # Dataset sizes
    n_train_shapes: int = 5000
    n_val_shapes: int = 500
    n_test_shapes: int = 500
    
    # Points per shape
    n_interior_points: int = 256
    n_boundary_points: int = 128
    
    # Harmonic map solver settings
    harmonic_grid_size: int = 64  # Grid for solving Laplace equation
    harmonic_tol: float = 1e-6
    harmonic_max_iter: int = 5000
    
    # Shape distribution: Dict[shape_name, weight] or None for uniform
    # Example: {"Ellipse": 0.5, "Superellipse": 0.5}
    # Example: {"Ellipse": 1.0}  # Only ellipses
    # Available shapes: Ellipse, Superellipse, FourierBlob, StarPolygon, SmoothPolygon
    shape_distribution: Dict[str, float] = field(default_factory=lambda: None)
    
    # Random seed
    seed: int = 42


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    # Encoder
    latent_dim: int = 256
    
    # Velocity field
    hidden_dim: int = 256
    n_layers: int = 4
    n_coord_frequencies: int = 8
    n_time_frequencies: int = 4
    
    # Lipschitz constraint (critical for diffeomorphism guarantee)
    lipschitz_const: float = 5.0
    
    # ODE solver
    dt0: float = 0.1
    rtol: float = 1e-5
    atol: float = 1e-5


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    n_epochs: int = 200
    
    # Learning rate schedule
    warmup_epochs: int = 5
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 20
    
    # Loss weights (these need tuning!)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "supervised": 1.0,      # MSE on interior points
        "boundary": 10.0,       # Boundary maps to boundary
        "jacobian": 0.1,        # Positive Jacobian determinant
        "cycle": 1.0,           # Forward-backward consistency
        "coverage": 0.5,        # Points stay in [-1,1]²
    })
    
    # Jacobian regularization
    min_jacobian_det: float = 0.01
    
    # Checkpointing
    checkpoint_every: int = 10
    validate_every: int = 5
    
    # Early stopping
    early_stopping_patience: int = 30
    
    # Random seed
    seed: int = 42


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


def get_config() -> Config:
    """Get default configuration."""
    return Config()


# Hyperparameter tuning suggestions
TUNING_GUIDE = """
HYPERPARAMETER TUNING GUIDE
===========================

1. LIPSCHITZ CONSTANT (model.lipschitz_const)
   - Controls how fast the flow can change
   - Too low (< 2): transformations too constrained, poor fit
   - Too high (> 10): numerical instability, potential folding
   - Start with 5.0, increase if underfitting, decrease if unstable
   
2. LOSS WEIGHTS
   - boundary: Most critical for correct topology mapping
     Start high (10-50), decrease if interior fit suffers
   
   - supervised: Balance with boundary loss
     If you have good ground truth, can be 1.0-10.0
   
   - jacobian: Prevents folding during training
     Usually 0.01-0.1 is enough, increase if seeing negative determinants
   
   - cycle: Improves inverse quality
     Set to 0 initially to speed up training, add later for refinement
   
   - coverage: Keeps outputs in [-1,1]²
     Usually 0.1-1.0, increase if points escape bounds

3. LEARNING RATE
   - Start with 1e-4 for Adam/AdamW
   - If loss oscillates: reduce to 1e-5
   - If converges too slowly: try 3e-4

4. NUMBER OF ODE LAYERS (model.n_layers)
   - More layers = more expressive velocity field
   - 4 layers usually sufficient
   - Try 6-8 for very complex shapes

5. FREQUENCY EMBEDDINGS (model.n_coord_frequencies)
   - More frequencies = finer spatial resolution
   - 8-16 usually good
   - Increase for shapes with sharp features
"""


if __name__ == "__main__":
    # Print default config
    config = get_config()
    print("Default Configuration:")
    print(f"  Data: {config.data}")
    print(f"  Model: {config.model}")
    print(f"  Training: {config.training}")
    print("\n" + TUNING_GUIDE)
