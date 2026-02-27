"""
Debug script to verify double-exclusion bug in CVRP training.

This script checks:
1. Shapes of data at each stage
2. Whether energy_fn receives correct customer-only data
3. Whether there's any double-exclusion of depot/first customer
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.step_model import CVRPDiffusionStepModel
from diffusion.noise_schedule import CategoricalNoiseSchedule
from diffusion.trajectory import collect_cvrp_trajectory
from diffusion.cvrp_energy import compute_K, compute_cvrp_cost, create_energy_fn
from heatmap.cvrp.inst import gen_inst, gen_pyg_data, CAPACITIES


def debug_double_exclusion():
    """Debug test to verify data flow and check for double-exclusion."""
    print("=" * 70)
    print("DEBUG: Checking for Double-Exclusion Bug in CVRP Training")
    print("=" * 70)

    device = torch.device('cpu')

    # Test parameters
    n_nodes = 10  # 10 customers (+ depot = 11 total nodes)
    k_sparse = 5
    n_diffusion_steps = 3
    n_basis_states = 2
    T_temperature = 1.0
    hidden_dim = 32

    print(f"\n[Config]")
    print(f"  n_customers: {n_nodes}")
    print(f"  n_total_nodes (with depot): {n_nodes + 1}")
    print(f"  n_diffusion_steps: {n_diffusion_steps}")
    print(f"  n_basis_states: {n_basis_states}")

    # Generate CVRP instance (same as train_cvrp.py)
    print(f"\n[Step 1] Generate CVRP instance...")
    coors, demand, capacity = gen_inst(n_nodes, device)
    pyg_data = gen_pyg_data(coors, demand, capacity, k_sparse)

    print(f"  coors.shape: {coors.shape}  (should be ({n_nodes + 1}, 2) - includes depot)")
    print(f"  demand.shape: {demand.shape}  (should be ({n_nodes + 1},) - includes depot)")
    print(f"  demand[0] (depot): {demand[0].item()}  (should be 0)")
    print(f"  capacity: {capacity}")

    # Extract customer demands (same as train_cvrp.py line 464)
    customer_demands = demand[1:]
    print(f"\n[Step 2] Extract customer demands...")
    print(f"  customer_demands.shape: {customer_demands.shape}  (should be ({n_nodes},))")
    print(f"  customer_demands: {customer_demands.tolist()}")

    # Compute K (same as train_cvrp.py line 467)
    K_instance = compute_K(demand, capacity)
    print(f"\n[Step 3] Compute K...")
    print(f"  K_instance: {K_instance}")

    # Get node features (same as train_cvrp.py line 472)
    node_features = pyg_data.x.to(device)
    edge_index = pyg_data.edge_index.to(device)
    edge_attr = pyg_data.edge_attr.to(device)

    print(f"\n[Step 4] Graph data...")
    print(f"  node_features.shape: {node_features.shape}  (should be ({n_nodes + 1}, 3) - all nodes)")
    print(f"  edge_index.shape: {edge_index.shape}")

    # Create model
    K_max = K_instance + 2  # Some buffer
    model = CVRPDiffusionStepModel(
        n_classes=K_max,
        node_feat_dim=3,
        edge_dim=2,
        hidden_dim=hidden_dim,
        n_diffusion_steps=n_diffusion_steps,
        n_message_passes=2,
        n_random_features=5,
    ).to(device)

    noise_schedule = CategoricalNoiseSchedule(
        n_steps=n_diffusion_steps,
        n_classes=K_max,
        schedule='diffuco'
    )

    # Create node_graph_idx (same as train_cvrp.py)
    n_nodes_total = node_features.size(0)
    n_customers = n_nodes_total - 1
    n_graphs = 1
    depot_idx = 0

    node_graph_idx = torch.zeros(n_nodes_total, dtype=torch.long, device=device)
    customer_graph_idx = node_graph_idx[1:]  # Exclude depot

    print(f"\n[Step 5] Graph indices...")
    print(f"  n_nodes_total: {n_nodes_total}  (includes depot)")
    print(f"  n_customers: {n_customers}  (excludes depot)")
    print(f"  node_graph_idx.shape: {node_graph_idx.shape}  (all nodes)")
    print(f"  customer_graph_idx.shape: {customer_graph_idx.shape}  (customers only)")

    # Track what energy_fn receives
    energy_fn_calls = []

    def debug_energy_fn(X, node_graph_idx_arg, n_graphs_arg):
        """Wrapper to track energy_fn calls."""
        call_info = {
            'X_shape': X.shape,
            'X_values': X.tolist(),
            'node_graph_idx_shape': node_graph_idx_arg.shape,
            'n_graphs': n_graphs_arg,
        }
        energy_fn_calls.append(call_info)

        # Compute actual cost
        total_cost, _, _ = compute_cvrp_cost(
            X, coors, customer_demands, capacity, K_instance,
            use_projection=True,
            use_batched_sa=False,
        )
        return total_cost.unsqueeze(0)

    print(f"\n[Step 6] Collect trajectory...")
    model.eval()
    with torch.no_grad():
        buffer = collect_cvrp_trajectory(
            model=model,
            noise_schedule=noise_schedule,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_graph_idx=node_graph_idx,
            node_features=node_features,
            n_graphs=n_graphs,
            n_basis_states=n_basis_states,
            T_temperature=T_temperature,
            energy_fn=debug_energy_fn,
            device=device,
            eval_step_factor=1,
            K_valid=K_instance,
            depot_idx=depot_idx,
        )

    print(f"\n[Step 7] Trajectory buffer shapes...")
    print(f"  buffer.states.shape: {buffer.states.shape}")
    print(f"    Expected: ({n_diffusion_steps}, {n_customers}, {n_basis_states})")
    print(f"    Actual nodes dim: {buffer.states.shape[1]}")

    if buffer.states.shape[1] == n_customers:
        print(f"    [OK] States are CUSTOMER-ONLY (depot excluded)")
    elif buffer.states.shape[1] == n_nodes_total:
        print(f"    [ISSUE] States include ALL nodes (depot included)")
    else:
        print(f"    [ERROR] Unexpected shape!")

    print(f"\n  buffer.final_states.shape: {buffer.final_states.shape}")
    print(f"    Expected: ({n_customers}, {n_basis_states})")

    if buffer.final_states.shape[0] == n_customers:
        print(f"    [OK] Final states are CUSTOMER-ONLY")
    else:
        print(f"    [ISSUE] Final states have wrong shape!")

    print(f"\n[Step 8] Energy function calls analysis...")
    print(f"  Number of energy_fn calls: {len(energy_fn_calls)}")

    for i, call in enumerate(energy_fn_calls):
        print(f"\n  Call {i+1}:")
        print(f"    X.shape: {call['X_shape']}")
        print(f"    X values: {call['X_values']}")
        print(f"    node_graph_idx.shape: {call['node_graph_idx_shape']}")

        expected_n = n_customers
        actual_n = call['X_shape'][0]

        if actual_n == expected_n:
            print(f"    [OK] X has {actual_n} elements (= n_customers = {n_customers})")
        elif actual_n == expected_n - 1:
            print(f"    [BUG!] X has {actual_n} elements, MISSING 1 CUSTOMER!")
            print(f"           Expected {expected_n} customers, got {actual_n}")
            print(f"           This is the DOUBLE-EXCLUSION bug!")
        elif actual_n == n_nodes_total:
            print(f"    [ISSUE] X has {actual_n} elements (includes depot)")
        else:
            print(f"    [ERROR] Unexpected: X has {actual_n} elements")

    # Now simulate what train_cvrp.py CVRPPPOTrainer.train_step does
    print(f"\n[Step 9] Simulate train_step energy computation...")
    print(f"  (Checking if there's additional [1:] slicing in train_step)")

    # This is what happens in train_cvrp.py lines 336-341
    for b in range(n_basis_states):
        X_customers_0 = buffer.final_states[:, b]
        print(f"\n  Basis state {b}:")
        print(f"    X_customers_0.shape: {X_customers_0.shape}")
        print(f"    X_customers_0 values: {X_customers_0.tolist()}")

        # Check if this matches expected
        if X_customers_0.shape[0] == n_customers:
            print(f"    [OK] Shape matches n_customers={n_customers}")
        else:
            print(f"    [BUG] Shape mismatch! Expected {n_customers}, got {X_customers_0.shape[0]}")

    # Final verdict
    print(f"\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    issues_found = []

    # Check 1: States shape
    if buffer.states.shape[1] != n_customers:
        issues_found.append(f"States shape wrong: {buffer.states.shape[1]} vs expected {n_customers}")

    # Check 2: Final states shape
    if buffer.final_states.shape[0] != n_customers:
        issues_found.append(f"Final states shape wrong: {buffer.final_states.shape[0]} vs expected {n_customers}")

    # Check 3: Energy function input shape
    for i, call in enumerate(energy_fn_calls):
        if call['X_shape'][0] != n_customers:
            issues_found.append(f"Energy fn call {i+1}: X.shape[0]={call['X_shape'][0]} vs expected {n_customers}")

    if issues_found:
        print("\n[ISSUES FOUND]")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("\n[NO DOUBLE-EXCLUSION BUG FOUND]")
        print("All shapes are correct:")
        print(f"  - States: ({n_diffusion_steps}, {n_customers}, {n_basis_states}) ✓")
        print(f"  - Final states: ({n_customers}, {n_basis_states}) ✓")
        print(f"  - Energy fn receives: ({n_customers},) per call ✓")

    print("\n" + "=" * 70)

    return len(issues_found) == 0


if __name__ == "__main__":
    success = debug_double_exclusion()
    sys.exit(0 if success else 1)
