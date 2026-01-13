"""
Test DiffusionModel wrapper.
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

    # Create model
    model = DiffusionModelDense(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        node_dim=32,
        edge_dim=32,
        time_dim=32,
        coord_dim=2,
        out_channels=2
    )

    # Create dummy inputs
    coords = torch.rand(batch_size, n_nodes, 2)
    adj_matrix = torch.rand(batch_size, n_nodes, n_nodes)  # Noisy edge values
    timesteps = torch.randint(0, 100, (batch_size,))

    # Forward pass
    edge_logits = model(coords, adj_matrix, timesteps)

    # Check output shape
    expected_shape = (batch_size, n_nodes, n_nodes, 2)
    assert edge_logits.shape == expected_shape, f"Expected {expected_shape}, got {edge_logits.shape}"

    print(f"  Output shape: {edge_logits.shape} [PASS]")
    print("  Dense forward pass test passed!")
    return True


def test_diffusion_model_sampling():
    """Test sampling from model output."""
    print("\nTesting DiffusionModel sampling...")

    batch_size = 2
    n_nodes = 5

    model = DiffusionModelDense(
        n_layers=1,
        hidden_dim=16,
        node_dim=16,
        edge_dim=16,
        time_dim=16
    )

    # Create inputs
    coords = torch.rand(batch_size, n_nodes, 2)
    adj_matrix = torch.rand(batch_size, n_nodes, n_nodes)
    timesteps = torch.randint(0, 50, (batch_size,))

    # Get logits
    logits = model(coords, adj_matrix, timesteps)

    # Test sample_from_logits
    samples, log_probs = model.sample_from_logits(logits)

    assert samples.shape == (batch_size, n_nodes, n_nodes), f"Wrong sample shape: {samples.shape}"
    assert log_probs.shape == (batch_size, n_nodes, n_nodes), f"Wrong log_probs shape: {log_probs.shape}"
    assert torch.all((samples == 0) | (samples == 1)), "Samples should be binary"

    print(f"  Samples shape: {samples.shape} [PASS]")
    print(f"  Log probs shape: {log_probs.shape} [PASS]")
    print("  Sampling test passed!")
    return True


def test_diffusion_model_edge_prediction():
    """Test edge probability and prediction."""
    print("\nTesting DiffusionModel edge prediction...")

    batch_size = 2
    n_nodes = 5

    model = DiffusionModelDense(
        n_layers=1,
        hidden_dim=16,
        node_dim=16,
        edge_dim=16,
        time_dim=16
    )

    coords = torch.rand(batch_size, n_nodes, 2)
    adj_matrix = torch.rand(batch_size, n_nodes, n_nodes)
    timesteps = torch.randint(0, 100, (batch_size,))

    # Test get_edge_probs
    probs = model.get_edge_probs(coords, adj_matrix, timesteps)
    assert probs.shape == (batch_size, n_nodes, n_nodes, 2)
    assert torch.all(probs >= 0) and torch.all(probs <= 1), "Probs should be in [0,1]"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, n_nodes, n_nodes), atol=1e-5), "Probs should sum to 1"

    # Test get_edge_prediction
    preds = model.get_edge_prediction(coords, adj_matrix, timesteps, threshold=0.5)
    assert preds.shape == (batch_size, n_nodes, n_nodes)
    assert torch.all((preds == 0) | (preds == 1)), "Predictions should be binary"

    print(f"  Edge probs shape: {probs.shape} [PASS]")
    print(f"  Edge predictions shape: {preds.shape} [PASS]")
    print("  Edge prediction test passed!")
    return True


def test_sample_prior():
    """Test prior sampling."""
    print("\nTesting prior sampling...")

    model = DiffusionModelDense(n_layers=1, hidden_dim=16, node_dim=16, edge_dim=16, time_dim=16)

    shape = (4, 10, 10)
    prior = model.sample_prior(shape, device=torch.device('cpu'))

    assert prior.shape == shape
    assert torch.all((prior == 0) | (prior == 1)), "Prior should be binary"
    # Check approximately 50% are 1s (with some tolerance)
    mean_val = prior.mean().item()
    assert 0.3 < mean_val < 0.7, f"Prior should be ~50% 1s, got {mean_val:.2%}"

    print(f"  Prior shape: {prior.shape} [PASS]")
    print(f"  Prior mean: {mean_val:.2%} (expected ~50%) [PASS]")
    print("  Prior sampling test passed!")
    return True


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("\nTesting gradient flow...")

    batch_size = 2
    n_nodes = 5

    model = DiffusionModelDense(
        n_layers=1,
        hidden_dim=16,
        node_dim=16,
        edge_dim=16,
        time_dim=16
    )

    coords = torch.rand(batch_size, n_nodes, 2, requires_grad=True)
    adj_matrix = torch.rand(batch_size, n_nodes, n_nodes)
    timesteps = torch.randint(0, 100, (batch_size,))

    # Forward pass
    logits = model(coords, adj_matrix, timesteps)
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


def run_all_tests():
    """Run all diffusion model tests."""
    print("=" * 50)
    print("Running DiffusionModel Tests")
    print("=" * 50)

    tests = [
        test_diffusion_model_dense_forward,
        test_diffusion_model_sampling,
        test_diffusion_model_edge_prediction,
        test_sample_prior,
        test_gradient_flow,
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
