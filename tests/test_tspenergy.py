"""
Test TSP Energy function.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EnergyFunctions import TSPEnergyClass


def test_tour_length_calculation():
    """Test tour length calculation for known tours."""
    print("Testing tour length calculation...")
    
    config = {"penalty_weight": 1.0}
    energy_fn = TSPEnergyClass(config)
    
    # Create a simple 4-node square tour
    # Nodes at (0,0), (1,0), (1,1), (0,1)
    coords = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ])
    
    # Optimal tour: 0-1-2-3-0 (perimeter = 4)
    adj_matrix = torch.tensor([
        [[0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 0, 1, 0]]
    ]).float()
    
    tour_length = energy_fn.calculate_tour_length(coords, adj_matrix)
    expected_length = 4.0  # Square perimeter
    
    assert torch.allclose(tour_length, torch.tensor([expected_length]), atol=1e-5),         f"Expected {expected_length}, got {tour_length.item()}"
    
    print(f"  Tour length: {tour_length.item():.4f} (expected: {expected_length}) [PASS]")
    print("  Tour length test passed!")
    return True


def test_degree_violations():
    """Test degree constraint violation calculation."""
    print("\nTesting degree violations...")
    
    config = {"penalty_weight": 1.0}
    energy_fn = TSPEnergyClass(config)
    
    # Valid tour (all degrees = 2)
    valid_adj = torch.tensor([
        [[0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 0, 1, 0]]
    ]).float()
    
    violations, total = energy_fn.calculate_degree_violations(valid_adj, target_degree=2)
    
    assert torch.allclose(violations, torch.zeros(1, 4), atol=1e-5),         f"Expected no violations, got {violations}"
    assert total.item() == 0, f"Expected 0 total violations, got {total.item()}"
    
    print(f"  Valid tour violations: {total.item()} [PASS]")
    
    # Invalid tour (missing edge)
    invalid_adj = torch.tensor([
        [[0, 1, 0, 0],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0]]
    ]).float()
    
    violations, total = energy_fn.calculate_degree_violations(invalid_adj, target_degree=2)
    
    assert total.item() > 0, "Expected violations for invalid tour"
    print(f"  Invalid tour violations: {total.item()} [PASS]")
    
    print("  Degree violations test passed!")
    return True


def test_energy_calculation():
    """Test full energy calculation."""
    print("\nTesting energy calculation...")
    
    config = {"penalty_weight": 10.0}
    energy_fn = TSPEnergyClass(config)
    
    # Create 4-node square
    coords = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ])
    
    # Valid tour
    valid_adj = torch.tensor([
        [[0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 0, 1, 0]]
    ]).float()
    
    energy, degree_viol, total_viol = energy_fn.calculate_Energy(coords, valid_adj)
    
    print(f"  Energy: {energy.item():.4f}")
    print(f"  Total violations: {total_viol.item():.4f}")
    
    # For valid tour, violations should be minimal
    assert total_viol.item() < 1.0, f"Expected low violations for valid tour"
    
    print("  Energy calculation test passed!")
    return True


def test_relaxed_energy():
    """Test relaxed (soft) energy calculation."""
    print("\nTesting relaxed energy...")
    
    config = {"penalty_weight": 1.0}
    energy_fn = TSPEnergyClass(config)
    
    batch_size = 2
    n_nodes = 5
    
    coords = torch.rand(batch_size, n_nodes, 2)
    p = torch.rand(batch_size, n_nodes, n_nodes)
    
    # Relaxed energy should be differentiable
    p.requires_grad = True
    
    energy = energy_fn.calculate_relaxed_Energy(coords, p)
    
    assert energy.shape == (batch_size,), f"Expected shape ({batch_size},), got {energy.shape}"
    
    # Check gradients flow
    loss = energy.sum()
    loss.backward()
    
    assert p.grad is not None, "Gradients should flow through relaxed energy"
    
    print(f"  Relaxed energy shape: {energy.shape} [PASS]")
    print(f"  Gradients computed: {p.grad is not None} [PASS]")
    print("  Relaxed energy test passed!")
    return True


def test_reward_calculation():
    """Test reward calculation for RL."""
    print("\nTesting reward calculation...")
    
    config = {"penalty_weight": 1.0}
    energy_fn = TSPEnergyClass(config)
    
    coords = torch.rand(4, 10, 2)
    adj_matrix = torch.rand(4, 10, 10)
    
    reward = energy_fn.get_reward(coords, adj_matrix, normalize=True)
    
    assert reward.shape == (4,), f"Expected shape (4,), got {reward.shape}"
    
    # Reward should be negative (since energy is positive)
    # Lower energy = higher reward = less negative
    
    print(f"  Reward shape: {reward.shape} [PASS]")
    print(f"  Reward values: {reward.tolist()}")
    print("  Reward calculation test passed!")
    return True


def run_all_tests():
    """Run all TSP energy tests."""
    print("=" * 50)
    print("Running TSP Energy Tests")
    print("=" * 50)
    
    tests = [
        test_tour_length_calculation,
        test_degree_violations,
        test_energy_calculation,
        test_relaxed_energy,
        test_reward_calculation,
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
