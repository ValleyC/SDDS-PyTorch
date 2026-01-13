"""
Integration tests for SDDS-PyTorch with permutation matrix formulation.

Tests the full pipeline: model + noise + energy + trainer.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Networks import DiffusionModelDense
from NoiseDistributions import CategoricalNoise
from EnergyFunctions import TSPEnergyClass
from Trainers import PPOTrainer
from Data import get_tsp_dataloader


def test_full_pipeline_initialization():
    """Test that all components can be initialized together."""
    print("Testing full pipeline initialization...")

    n_nodes = 10

    config = {
        "n_nodes": n_nodes,
        "n_diffusion_steps": 5,
        "diff_schedule": "DiffUCO",
        "n_bernoulli_features": n_nodes,  # Categorical over N positions
        "n_layers": 2,
        "hidden_dim": 32,
        "node_dim": 32,
        "edge_dim": 32,
        "time_dim": 32,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "n_basis_states": 2,
        "clip_value": 0.2,
        "value_coef": 0.5,
        "gae_lambda": 0.95,
        "n_inner_steps": 2,
        "penalty_weight": 1.0,
        "seed": 42,
    }

    device = torch.device("cpu")

    # Create model - now requires n_nodes for categorical output
    model = DiffusionModelDense(
        n_nodes=config["n_nodes"],
        n_layers=config["n_layers"],
        hidden_dim=config["hidden_dim"],
        node_dim=config["node_dim"],
        edge_dim=config["edge_dim"],
        time_dim=config["time_dim"],
    )

    # Create noise distribution
    noise_class = CategoricalNoise(config)

    # Create energy function
    energy_class = TSPEnergyClass(config)

    # Create trainer
    trainer = PPOTrainer(
        config=config,
        model=model,
        energy_class=energy_class,
        noise_class=noise_class,
        device=device
    )

    print("  All components initialized successfully [PASS]")
    return True


def test_forward_pass():
    """Test a forward pass through the model."""
    print("\nTesting forward pass...")

    n_nodes = 10

    config = {
        "n_nodes": n_nodes,
        "n_diffusion_steps": 5,
        "diff_schedule": "DiffUCO",
        "n_bernoulli_features": n_nodes,
        "n_layers": 2,
        "hidden_dim": 32,
        "node_dim": 32,
        "edge_dim": 32,
        "time_dim": 32,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "n_basis_states": 2,
        "clip_value": 0.2,
        "value_coef": 0.5,
        "gae_lambda": 0.95,
        "n_inner_steps": 2,
        "penalty_weight": 1.0,
        "seed": 42,
    }

    device = torch.device("cpu")
    batch_size = 4

    # Create model
    model = DiffusionModelDense(
        n_nodes=config["n_nodes"],
        n_layers=config["n_layers"],
        hidden_dim=config["hidden_dim"],
        node_dim=config["node_dim"],
        edge_dim=config["edge_dim"],
        time_dim=config["time_dim"],
    )

    # Create inputs - positions instead of adjacency matrix
    coords = torch.rand(batch_size, n_nodes, 2)
    positions = torch.randint(0, n_nodes, (batch_size, n_nodes))  # Position indices
    timesteps = torch.randint(0, 5, (batch_size,))

    # Forward pass
    logits = model(coords, positions, timesteps)

    # Model outputs (batch, n_nodes, n_positions) for categorical position classification
    expected_shape = (batch_size, n_nodes, n_nodes)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

    print(f"  Output shape: {logits.shape} [PASS]")
    return True


def test_dataloader():
    """Test data loading."""
    print("\nTesting data loader...")

    loader = get_tsp_dataloader(
        n_nodes=10,
        batch_size=4,
        n_instances=16,
        seed=42
    )

    batch = next(iter(loader))

    assert "coords" in batch, "Batch should contain 'coords'"
    assert batch["coords"].shape == (4, 10, 2), f"Wrong shape: {batch['coords'].shape}"

    print(f"  Batch coords shape: {batch['coords'].shape} [PASS]")
    return True


def test_single_train_step():
    """Test a single training step."""
    print("\nTesting single train step...")

    n_nodes = 10

    config = {
        "n_nodes": n_nodes,
        "n_diffusion_steps": 3,
        "diff_schedule": "DiffUCO",
        "n_bernoulli_features": n_nodes,
        "n_layers": 2,
        "hidden_dim": 32,
        "node_dim": 32,
        "edge_dim": 32,
        "time_dim": 32,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "n_basis_states": 2,
        "clip_value": 0.2,
        "value_coef": 0.5,
        "gae_lambda": 0.95,
        "n_inner_steps": 2,
        "penalty_weight": 1.0,
        "seed": 42,
    }

    device = torch.device("cpu")

    # Create components
    model = DiffusionModelDense(
        n_nodes=config["n_nodes"],
        n_layers=config["n_layers"],
        hidden_dim=config["hidden_dim"],
        node_dim=config["node_dim"],
        edge_dim=config["edge_dim"],
        time_dim=config["time_dim"],
    )

    noise_class = CategoricalNoise(config)
    energy_class = TSPEnergyClass(config)

    trainer = PPOTrainer(
        config=config,
        model=model,
        energy_class=energy_class,
        noise_class=noise_class,
        device=device
    )

    # Create batch
    batch = {"coords": torch.rand(4, n_nodes, 2)}

    # Run train step
    loss, metrics = trainer.train_step(batch)

    assert loss is not None, "Loss should not be None"
    assert isinstance(metrics, dict), "Metrics should be a dict"

    print(f"  Loss: {loss.item():.4f} [PASS]")
    print(f"  Metrics keys: {list(metrics.keys())} [PASS]")
    return True


def test_sampling():
    """Test sampling from the model."""
    print("\nTesting sampling...")

    n_nodes = 10

    config = {
        "n_nodes": n_nodes,
        "n_diffusion_steps": 3,
        "diff_schedule": "DiffUCO",
        "n_bernoulli_features": n_nodes,
        "n_layers": 2,
        "hidden_dim": 32,
        "node_dim": 32,
        "edge_dim": 32,
        "time_dim": 32,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "n_basis_states": 2,
        "clip_value": 0.2,
        "value_coef": 0.5,
        "gae_lambda": 0.95,
        "n_inner_steps": 2,
        "penalty_weight": 1.0,
        "seed": 42,
    }

    device = torch.device("cpu")

    # Create components
    model = DiffusionModelDense(
        n_nodes=config["n_nodes"],
        n_layers=config["n_layers"],
        hidden_dim=config["hidden_dim"],
        node_dim=config["node_dim"],
        edge_dim=config["edge_dim"],
        time_dim=config["time_dim"],
    )

    noise_class = CategoricalNoise(config)
    energy_class = TSPEnergyClass(config)

    trainer = PPOTrainer(
        config=config,
        model=model,
        energy_class=energy_class,
        noise_class=noise_class,
        device=device
    )

    # Sample
    coords = torch.rand(2, n_nodes, 2)

    trainer.model.eval()
    with torch.no_grad():
        result = trainer.sample(coords, n_samples=2)

    assert "x_0" in result, "Result should contain 'x_0'"
    assert "energies" in result, "Result should contain 'energies'"
    assert "best_energy" in result, "Result should contain 'best_energy'"

    # x_0 should be position indices: (batch, n_samples, n_nodes)
    print(f"  Samples shape: {result['x_0'].shape} [PASS]")
    print(f"  Best energies: {result['best_energy'].tolist()} [PASS]")
    return True


def test_checkpoint_save_load():
    """Test checkpoint save and load."""
    print("\nTesting checkpoint save/load...")

    import tempfile

    n_nodes = 10

    config = {
        "n_nodes": n_nodes,
        "n_diffusion_steps": 3,
        "diff_schedule": "DiffUCO",
        "n_bernoulli_features": n_nodes,
        "n_layers": 2,
        "hidden_dim": 32,
        "node_dim": 32,
        "edge_dim": 32,
        "time_dim": 32,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "n_basis_states": 2,
        "clip_value": 0.2,
        "value_coef": 0.5,
        "gae_lambda": 0.95,
        "n_inner_steps": 2,
        "penalty_weight": 1.0,
        "seed": 42,
    }

    device = torch.device("cpu")

    # Create trainer
    model = DiffusionModelDense(
        n_nodes=config["n_nodes"],
        n_layers=config["n_layers"],
        hidden_dim=config["hidden_dim"],
        node_dim=config["node_dim"],
        edge_dim=config["edge_dim"],
        time_dim=config["time_dim"],
    )

    noise_class = CategoricalNoise(config)
    energy_class = TSPEnergyClass(config)

    trainer = PPOTrainer(
        config=config,
        model=model,
        energy_class=energy_class,
        noise_class=noise_class,
        device=device
    )

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    trainer.save_checkpoint(checkpoint_path, epoch=5, additional_info={"test": "value"})

    # Load checkpoint
    checkpoint = trainer.load_checkpoint(checkpoint_path)

    assert checkpoint["epoch"] == 5, f"Expected epoch 5, got {checkpoint['epoch']}"
    assert checkpoint.get("test") == "value", "Additional info not preserved"

    # Cleanup
    os.unlink(checkpoint_path)

    print(f"  Checkpoint save/load successful [PASS]")
    return True


def test_noise_distribution():
    """Test noise distribution with categorical positions."""
    print("\nTesting noise distribution (categorical)...")

    n_nodes = 10

    config = {
        "n_nodes": n_nodes,
        "n_diffusion_steps": 5,
        "diff_schedule": "DiffUCO",
        "n_bernoulli_features": n_nodes,
    }

    noise_class = CategoricalNoise(config)

    # Test prior sampling
    batch_size = 4
    prior = noise_class.sample_prior((batch_size, n_nodes), torch.device("cpu"))

    assert prior.shape == (batch_size, n_nodes), f"Expected ({batch_size}, {n_nodes}), got {prior.shape}"
    assert torch.all(prior >= 0) and torch.all(prior < n_nodes), "Prior should be valid positions"

    print(f"  Prior shape: {prior.shape} [PASS]")
    print(f"  Prior range: [{prior.min().item()}, {prior.max().item()}] [PASS]")

    # Test noise step
    logits = torch.randn(batch_size, n_nodes, n_nodes)  # Model output
    X_t = torch.randint(0, n_nodes, (batch_size, n_nodes))  # Current positions
    t_idx = torch.tensor([3, 3, 3, 3])  # Timestep indices

    X_prev, log_prob = noise_class.calc_noise_step(logits, X_t, t_idx)

    assert X_prev.shape == (batch_size, n_nodes), f"Expected ({batch_size}, {n_nodes}), got {X_prev.shape}"
    assert log_prob.shape == (batch_size, n_nodes), f"Wrong log_prob shape: {log_prob.shape}"

    print(f"  Noise step output shape: {X_prev.shape} [PASS]")
    print("  Noise distribution test passed!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 50)
    print("Running Integration Tests (Permutation Formulation)")
    print("=" * 50)

    tests = [
        test_full_pipeline_initialization,
        test_forward_pass,
        test_dataloader,
        test_noise_distribution,
        test_single_train_step,
        test_sampling,
        test_checkpoint_save_load,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
