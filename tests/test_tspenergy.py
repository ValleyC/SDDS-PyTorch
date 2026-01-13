"""
Test TSP Energy function with permutation matrix formulation.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EnergyFunctions import TSPEnergyClass


def test_tour_length_calculation():
    """Test tour length calculation for known tours using permutation matrix."""
    print("Testing tour length calculation (permutation formulation)...")

    config = {"penalty_weight": 1.0, "n_nodes": 4}
    energy_fn = TSPEnergyClass(config)

    # Create a simple 4-node square tour
    # Nodes at (0,0), (1,0), (1,1), (0,1)
    coords = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ])

    # Optimal tour: node 0 at pos 0, node 1 at pos 1, node 2 at pos 2, node 3 at pos 3
    # Tour: 0 -> 1 -> 2 -> 3 -> 0 (perimeter = 4)
    # Position indices for each node
    positions = torch.tensor([[0, 1, 2, 3]])  # node i is at position i

    tour_length = energy_fn.calculate_tour_length(coords, positions)
    expected_length = 4.0  # Square perimeter

    assert torch.allclose(tour_length, torch.tensor([expected_length]), atol=1e-5), \
        f"Expected {expected_length}, got {tour_length.item()}"

    print(f"  Tour length: {tour_length.item():.4f} (expected: {expected_length}) [PASS]")
    print("  Tour length test passed!")
    return True


def test_permutation_constraints():
    """Test permutation matrix constraint violations."""
    print("\nTesting permutation constraints...")

    config = {"penalty_weight": 1.0, "n_nodes": 4}
    energy_fn = TSPEnergyClass(config)

    # Valid permutation: each node has unique position
    valid_positions = torch.tensor([[0, 1, 2, 3]])  # bijective mapping

    row_viol, col_viol = energy_fn.calculate_constraint_violations(valid_positions)

    assert torch.allclose(row_viol, torch.zeros(1), atol=1e-5), \
        f"Expected no row violations, got {row_viol}"
    assert torch.allclose(col_viol, torch.zeros(1), atol=1e-5), \
        f"Expected no col violations, got {col_viol}"

    print(f"  Valid permutation violations: row={row_viol.item():.4f}, col={col_viol.item():.4f} [PASS]")

    # Invalid permutation: two nodes at same position
    invalid_positions = torch.tensor([[0, 0, 2, 3]])  # nodes 0 and 1 both at position 0

    row_viol, col_viol = energy_fn.calculate_constraint_violations(invalid_positions)

    # Column sum for position 0 is 2 (violation), position 1 is 0 (violation)
    assert col_viol.item() > 0, "Expected column violations for duplicate positions"

    print(f"  Invalid permutation violations: row={row_viol.item():.4f}, col={col_viol.item():.4f} [PASS]")

    print("  Permutation constraints test passed!")
    return True


def test_energy_calculation():
    """Test full energy calculation with permutation matrix."""
    print("\nTesting energy calculation (permutation)...")

    config = {"penalty_weight": 10.0, "n_nodes": 4}
    energy_fn = TSPEnergyClass(config)

    # Create 4-node square
    coords = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ])

    # Valid tour (permutation)
    valid_positions = torch.tensor([[0, 1, 2, 3]])

    # calculate_Energy returns (energy, violations_per_node, constraint_penalty)
    energy, violations_per_node, constraint_penalty = energy_fn.calculate_Energy(coords, valid_positions)

    # Also get row/col violations separately
    row_viol, col_viol = energy_fn.calculate_constraint_violations(valid_positions)

    print(f"  Energy: {energy.item():.4f}")
    print(f"  Constraint penalty (HB): {constraint_penalty.item():.4f}")
    print(f"  Row violations: {row_viol.item():.4f}")
    print(f"  Col violations: {col_viol.item():.4f}")

    # For valid permutation, violations should be 0
    assert row_viol.item() < 0.1, f"Expected low row violations for valid tour"
    assert col_viol.item() < 0.1, f"Expected low col violations for valid tour"

    # Energy should be approximately tour length (4.0) for valid tour
    assert 3.5 < energy.item() < 4.5, f"Expected energy ~4.0, got {energy.item()}"

    print("  Energy calculation test passed!")
    return True


def test_different_tour_orders():
    """Test that different valid tours give same length for symmetric case."""
    print("\nTesting different tour orders...")

    config = {"penalty_weight": 1.0, "n_nodes": 4}
    energy_fn = TSPEnergyClass(config)

    # Square nodes
    coords = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ])

    # Different valid tours (all have length 4 for a square)
    # Tour 1: 0 -> 1 -> 2 -> 3 -> 0
    positions1 = torch.tensor([[0, 1, 2, 3]])
    # Tour 2: 0 -> 3 -> 2 -> 1 -> 0 (reverse)
    positions2 = torch.tensor([[0, 3, 2, 1]])

    length1 = energy_fn.calculate_tour_length(coords, positions1)
    length2 = energy_fn.calculate_tour_length(coords, positions2)

    print(f"  Tour 1 length: {length1.item():.4f}")
    print(f"  Tour 2 length: {length2.item():.4f}")

    assert torch.allclose(length1, length2, atol=1e-5), \
        f"Both tours should have same length, got {length1.item()} vs {length2.item()}"

    print("  Different tour orders test passed!")
    return True


def test_reward_calculation():
    """Test reward calculation for RL."""
    print("\nTesting reward calculation...")

    config = {"penalty_weight": 1.0, "n_nodes": 10}
    energy_fn = TSPEnergyClass(config)

    coords = torch.rand(4, 10, 2)
    positions = torch.randint(0, 10, (4, 10))

    reward = energy_fn.get_reward(coords, positions, normalize=True)

    assert reward.shape == (4,), f"Expected shape (4,), got {reward.shape}"

    # Reward should be negative (since energy is positive)
    # Lower energy = higher reward = less negative

    print(f"  Reward shape: {reward.shape} [PASS]")
    print(f"  Reward values: {reward.tolist()}")
    print("  Reward calculation test passed!")
    return True


def test_batch_processing():
    """Test that energy works correctly with batched inputs."""
    print("\nTesting batch processing...")

    config = {"penalty_weight": 1.0, "n_nodes": 5}
    energy_fn = TSPEnergyClass(config)

    batch_size = 8
    n_nodes = 5
    coords = torch.rand(batch_size, n_nodes, 2)
    positions = torch.randint(0, n_nodes, (batch_size, n_nodes))

    # calculate_Energy returns (energy, violations_per_node, constraint_penalty)
    energy, violations_per_node, constraint_penalty = energy_fn.calculate_Energy(coords, positions)

    assert energy.shape == (batch_size,), f"Expected shape ({batch_size},), got {energy.shape}"
    assert violations_per_node.shape == (batch_size, n_nodes), f"Expected shape ({batch_size}, {n_nodes}), got {violations_per_node.shape}"
    assert constraint_penalty.shape == (batch_size,), f"Expected shape ({batch_size},), got {constraint_penalty.shape}"

    # Also test constraint violations helper
    row_viol, col_viol = energy_fn.calculate_constraint_violations(positions)
    assert row_viol.shape == (batch_size,), f"Expected shape ({batch_size},), got {row_viol.shape}"
    assert col_viol.shape == (batch_size,), f"Expected shape ({batch_size},), got {col_viol.shape}"

    print(f"  Energy shape: {energy.shape} [PASS]")
    print(f"  Energy values: {energy[:3].tolist()} ...")
    print("  Batch processing test passed!")
    return True


def run_all_tests():
    """Run all TSP energy tests."""
    print("=" * 50)
    print("Running TSP Energy Tests (Permutation Formulation)")
    print("=" * 50)

    tests = [
        test_tour_length_calculation,
        test_permutation_constraints,
        test_energy_calculation,
        test_different_tour_orders,
        test_reward_calculation,
        test_batch_processing,
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
