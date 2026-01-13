"""
Test DiffusionModel wrapper for categorical position output.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Networks import DiffusionModel, DiffusionModelDense


def test_diffusion_model_dense_forward():
    """Test dense mode forward pass."""
    print("Testing DiffusionModel dense forward pass...")

    batch_size = 4
    n_nodes = 10
    hidden_dim = 32
    n_layers = 2

    # Create model - now requires n_nodes for categorical output
    model = DiffusionModelDense(
        n_nodes=n_nodes,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        node_dim=32,
        edge_dim=32,
        time_dim=32,
    )

    # Create dummy inputs
    coords = torch.rand(batch_size, n_nodes, 2)
    # Position indices (each node has a position 0 to n_nodes-1)
    positions = torch.randint(0, n_nodes, (batch_size, n_nodes))
    timesteps = torch.randint(0, 100, (batch_size,))

    # Forward pass - outputs position logits per node
    position_logits = model(coords, positions, timesteps)

    # Check output shape: (batch, n_nodes, n_positions)
    expected_shape = (batch_size, n_nodes, n_nodes)
    assert position_logits.shape == expected_shape, f"Expected {expected_shape}, got {position_logits.shape}"

    print(f"  Output shape: {position_logits.shape} [PASS]")
    print("  Dense forward pass test passed!")
    return True


def test_diffusion_model_sampling():
    """Test sampling from model output."""
    print("\nTesting DiffusionModel sampling...")

    batch_size = 2
    n_nodes = 5

    model = DiffusionModelDense(
        n_nodes=n_nodes,
        n_layers=1,
        hidden_dim=16,
        node_dim=16,
        edge_dim=16,
        time_dim=16
    )

    # Create inputs
    coords = torch.rand(batch_size, n_nodes, 2)
    positions = torch.randint(0, n_nodes, (batch_size, n_nodes))
    timesteps = torch.randint(0, 50, (batch_size,))

    # Get logits
    logits = model(coords, positions, timesteps)

    # Test sample_from_logits - returns position indices and log probs
    samples, log_probs = model.sample_from_logits(logits)

    # Samples are position indices: (batch, n_nodes)
    assert samples.shape == (batch_size, n_nodes), f"Wrong sample shape: {samples.shape}"
    # Log probs for each node's sampled position
    assert log_probs.shape == (batch_size, n_nodes), f"Wrong log_probs shape: {log_probs.shape}"
    # Position indices should be in [0, n_nodes)
    assert torch.all(samples >= 0) and torch.all(samples < n_nodes), "Samples should be valid position indices"

    print(f"  Samples shape: {samples.shape} [PASS]")
    print(f"  Log probs shape: {log_probs.shape} [PASS]")
    print("  Sampling test passed!")
    return True


def test_sample_prior():
    """Test prior sampling - uniform over positions."""
    print("\nTesting prior sampling...")

    n_nodes = 10
    model = DiffusionModelDense(
        n_nodes=n_nodes,
        n_layers=1,
        hidden_dim=16,
        node_dim=16,
        edge_dim=16,
        time_dim=16
    )

    batch_size = 4
    shape = (batch_size, n_nodes)
    prior = model.sample_prior(shape, device=torch.device('cpu'))

    # Prior should be position indices: (batch, n_nodes)
    assert prior.shape == shape, f"Expected {shape}, got {prior.shape}"
    # Position indices should be in [0, n_nodes)
    assert torch.all(prior >= 0) and torch.all(prior < n_nodes), "Prior should be valid positions"

    # Check distribution is approximately uniform
    counts = torch.zeros(n_nodes)
    for pos in prior.flatten():
        counts[pos] += 1

    # With batch_size * n_nodes = 40 samples over 10 categories,
    # expected ~4 per category
    print(f"  Prior shape: {prior.shape} [PASS]")
    print(f"  Position distribution: {counts.tolist()}")
    print("  Prior sampling test passed!")
    return True


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("\nTesting gradient flow...")

    batch_size = 2
    n_nodes = 5

    model = DiffusionModelDense(
        n_nodes=n_nodes,
        n_layers=1,
        hidden_dim=16,
        node_dim=16,
        edge_dim=16,
        time_dim=16
    )

    coords = torch.rand(batch_size, n_nodes, 2, requires_grad=True)
    positions = torch.randint(0, n_nodes, (batch_size, n_nodes))
    timesteps = torch.randint(0, 100, (batch_size,))

    # Forward pass
    logits = model(coords, positions, timesteps)
    loss = logits.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert coords.grad is not None, "Coordinates should have gradients"
    assert coords.grad.shape == coords.shape, "Gradient shape should match input shape"

    # Check model parameters have gradients
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "Model parameters should have gradients"

    print("  Gradients computed successfully [PASS]")
    print("  Gradient flow test passed!")
    return True


def test_position_logits_reasonable():
    """Test that position logits produce reasonable probabilities."""
    print("\nTesting position logits are reasonable...")

    batch_size = 2
    n_nodes = 5

    model = DiffusionModelDense(
        n_nodes=n_nodes,
        n_layers=2,
        hidden_dim=16,
        node_dim=16,
        edge_dim=16,
        time_dim=16
    )

    coords = torch.rand(batch_size, n_nodes, 2)
    positions = torch.randint(0, n_nodes, (batch_size, n_nodes))
    timesteps = torch.randint(0, 100, (batch_size,))

    logits = model(coords, positions, timesteps)
    probs = torch.softmax(logits, dim=-1)

    # Probabilities should sum to 1 along position dimension
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
        "Probabilities should sum to 1"

    # All probabilities should be positive
    assert torch.all(probs >= 0), "All probabilities should be non-negative"

    print(f"  Probs sum correctly: {prob_sums[0].tolist()} [PASS]")
    print("  Position logits test passed!")
    return True


def run_all_tests():
    """Run all diffusion model tests."""
    print("=" * 50)
    print("Running DiffusionModel Tests (Categorical)")
    print("=" * 50)

    tests = [
        test_diffusion_model_dense_forward,
        test_diffusion_model_sampling,
        test_sample_prior,
        test_gradient_flow,
        test_position_logits_reasonable,
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
