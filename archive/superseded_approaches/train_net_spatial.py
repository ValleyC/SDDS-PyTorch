"""
NetSpatialGNN Training & Evaluation — Stage 1: Supervised Displacement Regression

Modes:
  1. collect:  Run pure LNS and save solver traces
  2. train:    Supervised training — predict CP-SAT displacement from placement state
  3. eval:     Per-subproblem hint evaluation within LNS loop
  4. ablation: Compare topology_only / spatial_only / dual / no_hint

Usage:
    # Collect traces
    python train_net_spatial.py --mode collect --circuit ibm01 --n_iterations 500

    # Train on collected traces
    python train_net_spatial.py --mode train --circuits ibm01 --trace_dir traces/

    # Evaluate as warm-start hints in LNS
    python train_net_spatial.py --mode eval --circuit ibm01 --checkpoint checkpoints_netspatial/best.pt

    # Ablation study
    python train_net_spatial.py --mode ablation --circuit ibm01 --trace_dir traces/
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from benchmark_loader import load_bookshelf_circuit
from cpsat_solver import (
    legalize, compute_net_hpwl, check_overlap,
    solve_subset, solve_subset_guided,
)
from lns_solver import LNSSolver, compute_per_macro_rudy
from experience_buffer import ExperienceBuffer
from net_spatial_gnn import NetSpatialGNN


# ============================================================================
# Helpers
# ============================================================================

def load_circuit_data(args, circuit_name=None):
    """Load and legalize a circuit."""
    circuit = circuit_name or args.circuit
    circuit_dir = os.path.join(
        args.benchmark_base, "iccad04", "extracted", circuit)
    data = load_bookshelf_circuit(
        circuit_dir, circuit,
        macros_only=True,
        max_nodes=getattr(args, 'max_nodes', None),
        seed=args.seed,
    )

    positions = data['positions']
    sizes = data['node_features']
    nets = data['nets']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']
    N = data['n_components']

    print(f"Loaded {circuit}: {N} macros, {len(nets)} nets, "
          f"{edge_index.shape[1]} edges")

    ref_hpwl = compute_net_hpwl(positions, sizes, nets)
    _, ref_ov_pairs = check_overlap(positions, sizes)
    print(f"  Reference HPWL: {ref_hpwl:.4f}, overlapping pairs: {ref_ov_pairs}")

    if not getattr(args, 'skip_legalize', False) and ref_ov_pairs > 0:
        print(f"  Legalizing...")
        legal_pos = legalize(
            positions, sizes,
            time_limit=getattr(args, 'legalize_time_limit', 60.0),
            window_fraction=getattr(args, 'legalize_window', 0.3),
        )
        if legal_pos is not None:
            positions = legal_pos
            legal_hpwl = compute_net_hpwl(positions, sizes, nets)
            _, legal_ov = check_overlap(positions, sizes)
            print(f"  Post-legalize HPWL: {legal_hpwl:.4f}, overlaps: {legal_ov}")
        else:
            print(f"  Legalization failed, using reference")

    return {
        'positions': positions,
        'sizes': sizes,
        'nets': nets,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'ref_hpwl': ref_hpwl,
        'N': N,
    }


def compute_node_features_6d(positions, sizes, nets, macro_nets=None):
    """
    Build 6D node features from pre_positions only (no leakage).

    Features: [pos_x, pos_y, size_w, size_h, hpwl_norm, congestion_norm]
    """
    N = positions.shape[0]

    # Per-macro HPWL
    macro_hpwl = np.zeros(N, dtype=np.float32)
    for net in nets:
        if len(net) < 2:
            continue
        xs = [positions[n, 0] + dx for (n, dx, dy) in net]
        ys = [positions[n, 1] + dy for (n, dx, dy) in net]
        net_hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        for (node_idx, _, _) in net:
            macro_hpwl[node_idx] += net_hpwl
    hmax = macro_hpwl.max()
    if hmax > 1e-8:
        macro_hpwl /= hmax

    # Per-macro RUDY congestion
    if macro_nets is None:
        macro_nets = [[] for _ in range(N)]
        for ni, net in enumerate(nets):
            for (node_idx, _, _) in net:
                macro_nets[node_idx].append(ni)
    raw_rudy = compute_per_macro_rudy(positions, sizes, nets, macro_nets)
    p95 = np.percentile(raw_rudy, 95) if N > 1 else (raw_rudy.max() + 1e-8)
    congestion = np.clip(raw_rudy / (p95 + 1e-8), 0.0, 5.0).astype(np.float32)
    cmax = congestion.max()
    if cmax > 1e-8:
        congestion /= cmax

    features = np.column_stack([
        positions.astype(np.float32),          # 2: pos_x, pos_y
        sizes.astype(np.float32),              # 2: size_w, size_h
        macro_hpwl.reshape(-1, 1),             # 1: hpwl_norm
        congestion.reshape(-1, 1),             # 1: congestion_norm
    ])
    return features


def create_model(args, device):
    """Create NetSpatialGNN with args."""
    return NetSpatialGNN(
        node_input_dim=6,
        topo_edge_dim=4,
        spatial_edge_dim=6,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        grid_size=args.grid_size,
        k_spatial=args.k_spatial,
        mode=args.model_mode,
    ).to(device)


# ============================================================================
# Mode: collect
# ============================================================================

def cmd_collect(args):
    """Run pure heuristic LNS and save solver traces.

    Uses MLGuidedLNSSolver with model=None (pure heuristic, no ML) because
    it has built-in ExperienceBuffer integration that LNSSolver lacks.
    Functionally identical to LNSSolver for trace collection.
    """
    from ml_lns_solver import MLGuidedLNSSolver

    data = load_circuit_data(args)
    buf = ExperienceBuffer(max_size=args.buffer_size)

    solver = MLGuidedLNSSolver(
        positions=data['positions'],
        sizes=data['sizes'],
        nets=data['nets'],
        edge_index=data['edge_index'],
        edge_attr=data['edge_attr'],
        model=None,
        use_learned_subset=False,
        experience_buffer=buf,
        circuit_id=args.circuit,
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        seed=args.seed,
    )

    result = solver.solve(
        n_iterations=args.n_iterations,
        log_every=args.log_every,
    )

    os.makedirs(args.trace_dir, exist_ok=True)
    trace_path = os.path.join(args.trace_dir, f'{args.circuit}.pkl')
    buf.save(trace_path)
    stats = buf.stats()
    print(f"\nSaved {stats['size']} traces to {trace_path}")
    print(f"  Feasible rate: {stats['feasible_rate']:.2f}")
    print(f"  Mean improvement: {stats['mean_improvement']:.6f}")
    print(f"  Best HPWL: {result['best_hpwl']:.4f}")


# ============================================================================
# Mode: train
# ============================================================================

def cmd_train(args):
    """Supervised displacement regression on LNS traces."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load traces from all specified circuits
    circuits = [c.strip() for c in args.circuits.split(',')]
    all_targets = []

    for circuit in circuits:
        trace_path = os.path.join(args.trace_dir, f'{circuit}.pkl')
        if not os.path.exists(trace_path):
            print(f"  WARNING: {trace_path} not found, skipping")
            continue

        # Load circuit data to get nets (for feature recomputation)
        circuit_dir = os.path.join(
            args.benchmark_base, "iccad04", "extracted", circuit)
        cdata = load_bookshelf_circuit(
            circuit_dir, circuit, macros_only=True, seed=args.seed)
        nets = cdata['nets']

        # Build macro_nets index
        N = cdata['n_components']
        macro_nets = [[] for _ in range(N)]
        for ni, net in enumerate(nets):
            for (node_idx, _, _) in net:
                macro_nets[node_idx].append(ni)

        buf = ExperienceBuffer()
        buf.load(trace_path)
        targets = buf.get_displacement_targets(min_delta=args.min_delta)
        print(f"  {circuit}: {len(targets)} improving traces from {len(buf)} total")

        # Recompute features from pre_positions (feature leakage guard)
        for t in targets:
            t['node_features_6d'] = compute_node_features_6d(
                t['pre_positions'], t['sizes'], nets, macro_nets)
            t['nets'] = nets
        all_targets.extend(targets)

    if not all_targets:
        print("ERROR: No training data found")
        return

    print(f"\nTotal training examples: {len(all_targets)}")

    # Train/val split (80/20)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(all_targets))
    n_val = max(1, int(0.2 * len(all_targets)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Create model
    model = create_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model_mode}, {n_params:,} params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training
        model.train()
        rng.shuffle(train_idx)
        train_losses = []

        for batch_start in range(0, len(train_idx), args.batch_size):
            batch_indices = train_idx[batch_start:batch_start + args.batch_size]
            batch_loss = torch.tensor(0.0, device=device)
            batch_count = 0

            for idx in batch_indices:
                t = all_targets[idx]
                nf = torch.tensor(t['node_features_6d'], dtype=torch.float32, device=device)
                pos = torch.tensor(t['pre_positions'], dtype=torch.float32, device=device)
                sz = torch.tensor(t['sizes'], dtype=torch.float32, device=device)
                ei = torch.tensor(t['edge_index'], dtype=torch.long, device=device)
                ea = torch.tensor(t['edge_attr'], dtype=torch.float32, device=device)
                target_disp = torch.tensor(t['displacement'], dtype=torch.float32, device=device)
                subset = t['subset_indices']
                weight = t['weight']

                out = model(nf, pos, sz, ei, ea)
                pred_disp = out['displacement_pred']  # (V, 2)

                # Loss on subset macros only
                subset_mask = np.array([int(i) for i in subset])
                loss_subset = F.mse_loss(pred_disp[subset_mask], target_disp[subset_mask])
                batch_loss = batch_loss + weight * loss_subset
                batch_count += 1

            if batch_count > 0:
                batch_loss = batch_loss / batch_count
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(batch_loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for idx in val_idx:
                t = all_targets[idx]
                nf = torch.tensor(t['node_features_6d'], dtype=torch.float32, device=device)
                pos = torch.tensor(t['pre_positions'], dtype=torch.float32, device=device)
                sz = torch.tensor(t['sizes'], dtype=torch.float32, device=device)
                ei = torch.tensor(t['edge_index'], dtype=torch.long, device=device)
                ea = torch.tensor(t['edge_attr'], dtype=torch.float32, device=device)
                target_disp = torch.tensor(t['displacement'], dtype=torch.float32, device=device)
                subset = t['subset_indices']

                out = model(nf, pos, sz, ei, ea)
                pred_disp = out['displacement_pred']
                subset_mask = np.array([int(i) for i in subset])
                loss = F.mse_loss(pred_disp[subset_mask], target_disp[subset_mask])
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses) if train_losses else 0.0
        val_loss = np.mean(val_losses) if val_losses else 0.0

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, 'best.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'model_mode': args.model_mode,
                'hidden_dim': args.hidden_dim,
                'n_layers': args.n_layers,
                'k_spatial': args.k_spatial,
                'grid_size': args.grid_size,
            }, ckpt_path)

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved to {os.path.join(args.checkpoint_dir, 'best.pt')}")


# ============================================================================
# Mode: eval
# ============================================================================

def cmd_eval(args, model=None):
    """Two independent trajectory evaluation: no-hint vs hint.

    Design: TWO separate trajectories advance independently using the same
    subset selections (same strategy rotation, same k). Each iteration:
      1. Select subset from the no-hint trajectory's state (shared_solver)
      2. Solve WITHOUT hint from no-hint state → advance no-hint trajectory
      3. Solve WITH hint from hint state → advance hint trajectory
    Both trajectories use the same subsets for comparability, but each solves
    from its own accumulated state, giving a real trajectory comparison.

    Returns dict of summary metrics (for ablation mode).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_circuit_data(args)

    # Load model (can be passed from ablation mode)
    if model is None:
        model = create_model(args, device)
        if args.checkpoint and os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded model from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
        else:
            print("WARNING: No checkpoint loaded, using random weights")
    model.eval()

    positions = data['positions'].copy()
    sizes = data['sizes']
    nets = data['nets']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']
    N = data['N']

    # Build macro_nets index
    macro_nets = [[] for _ in range(N)]
    for ni, net in enumerate(nets):
        for (node_idx, _, _) in net:
            macro_nets[node_idx].append(ni)

    # No-hint trajectory solver (also drives subset selection)
    nohint_solver = LNSSolver(
        positions=positions.copy(),
        sizes=sizes,
        nets=nets,
        edge_index=edge_index,
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        seed=args.seed,
    )

    # Hint trajectory state (independent, advances with hint results)
    hint_traj_pos = positions.copy()

    strategies = ['random', 'worst_hpwl', 'congestion', 'connected']
    n_iters = args.n_iterations

    # Per-subproblem metrics
    records = []
    nohint_improved = 0
    hint_improved = 0

    print(f"\n--- Eval: {n_iters} iterations (two independent trajectories) ---")

    for it in range(n_iters):
        strategy = strategies[it % len(strategies)]
        k = min(args.subset_size, N)

        # Select subset from no-hint state (drives both trajectories)
        subset = nohint_solver.get_neighborhood(strategy, k)

        # --- No-hint trajectory: solve from nohint state ---
        nohint_pos = nohint_solver.current_pos
        nohint_hpwl_before = compute_net_hpwl(nohint_pos, sizes, nets)

        t0 = time.perf_counter()
        nohint_result = solve_subset(
            nohint_pos, sizes, nets, subset,
            time_limit=args.cpsat_time_limit,
            window_fraction=nohint_solver.window_fraction,
        )
        nohint_time = time.perf_counter() - t0

        nohint_delta = 0.0
        nohint_feasible = nohint_result is not None
        if nohint_feasible:
            nohint_hpwl_after = compute_net_hpwl(nohint_result, sizes, nets)
            nohint_delta = nohint_hpwl_before - nohint_hpwl_after
            if nohint_delta > 0:
                nohint_solver.current_pos = nohint_result
                nohint_solver._update_macro_hpwl()  # Fix #2: refresh for worst_hpwl
                nohint_improved += 1

        # --- Hint trajectory: solve from hint state ---
        hint_hpwl_before = compute_net_hpwl(hint_traj_pos, sizes, nets)

        with torch.no_grad():
            # Compute features from hint trajectory's own state (Fix #1)
            nf = compute_node_features_6d(hint_traj_pos, sizes, nets, macro_nets)
            nf_t = torch.tensor(nf, dtype=torch.float32, device=device)
            pos_t = torch.tensor(hint_traj_pos, dtype=torch.float32, device=device)
            sz_t = torch.tensor(sizes, dtype=torch.float32, device=device)
            ei_t = torch.tensor(edge_index, dtype=torch.long, device=device)
            ea_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)

            out = model(nf_t, pos_t, sz_t, ei_t, ea_t)
            delta_pred = out['displacement_pred'].cpu().numpy()

        hint_positions = hint_traj_pos + delta_pred

        t0 = time.perf_counter()
        hint_result = solve_subset_guided(
            hint_traj_pos, sizes, nets, subset,  # Fix #1: solve from hint state
            time_limit=args.cpsat_time_limit,
            window_fraction=nohint_solver.window_fraction,
            hint_positions=hint_positions,
        )
        hint_time = time.perf_counter() - t0

        hint_delta = 0.0
        hint_feasible = hint_result['new_positions'] is not None
        if hint_feasible:
            hint_hpwl_after = compute_net_hpwl(hint_result['new_positions'], sizes, nets)
            hint_delta = hint_hpwl_before - hint_hpwl_after
            if hint_delta > 0:
                hint_traj_pos = hint_result['new_positions'].copy()
                hint_improved += 1

        records.append({
            'nohint_delta': nohint_delta,
            'hint_delta': hint_delta,
            'nohint_time': nohint_time,
            'hint_time': hint_time,
            'nohint_feasible': nohint_feasible,
            'hint_feasible': hint_feasible,
        })

        if (it + 1) % args.log_every == 0:
            recent = records[-args.log_every:]
            avg_ht = np.mean([r['hint_time'] for r in recent])
            avg_nt = np.mean([r['nohint_time'] for r in recent])
            nohint_hpwl = compute_net_hpwl(nohint_solver.current_pos, sizes, nets)
            hint_hpwl = compute_net_hpwl(hint_traj_pos, sizes, nets)
            print(f"  [{it+1:3d}/{n_iters}] "
                  f"nohint={nohint_hpwl:.4f} hint={hint_hpwl:.4f} | "
                  f"avg time: nohint={avg_nt:.3f}s hint={avg_ht:.3f}s")

    # Summary statistics
    final_nohint_hpwl = compute_net_hpwl(nohint_solver.current_pos, sizes, nets)
    final_hint_hpwl = compute_net_hpwl(hint_traj_pos, sizes, nets)
    mean_nohint_delta = np.mean([r['nohint_delta'] for r in records])
    mean_hint_delta = np.mean([r['hint_delta'] for r in records])
    total_nohint_time = sum(r['nohint_time'] for r in records)
    total_hint_time = sum(r['hint_time'] for r in records)

    summary = {
        'mode': args.model_mode,
        'final_nohint_hpwl': final_nohint_hpwl,
        'final_hint_hpwl': final_hint_hpwl,
        'total_iters': n_iters,
        'nohint_improved': nohint_improved,
        'hint_improved': hint_improved,
        'mean_nohint_delta': mean_nohint_delta,
        'mean_hint_delta': mean_hint_delta,
        'total_nohint_time': total_nohint_time,
        'total_hint_time': total_hint_time,
    }

    print(f"\n{'='*60}")
    print(f"Eval Summary ({args.circuit}, {n_iters} iterations)")
    print(f"{'='*60}")
    print(f"  Model mode:        {args.model_mode}")
    print(f"  No-hint final HPWL: {final_nohint_hpwl:.4f}")
    print(f"  Hint final HPWL:    {final_hint_hpwl:.4f}")
    print(f"  Improving iters:    no-hint={nohint_improved}  hint={hint_improved}")
    print(f"  Mean delta/iter:    no-hint={mean_nohint_delta:.6f}  "
          f"hint={mean_hint_delta:.6f}")
    print(f"  Total time:         no-hint={total_nohint_time:.1f}s  "
          f"hint={total_hint_time:.1f}s")
    print(f"  Avg subproblem:     no-hint={total_nohint_time/max(n_iters,1):.3f}s  "
          f"hint={total_hint_time/max(n_iters,1):.3f}s")
    print(f"{'='*60}")

    return summary


# ============================================================================
# Mode: sweep — time-limit diagnostic
# ============================================================================

def cmd_sweep(args):
    """Sweep CP-SAT time limits to find where warm-start helps.

    From ONE fixed state, pick N subsets. For each subset, solve at each time
    budget with and without hint. Pure paired comparison — no trajectory drift.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_circuit_data(args)

    model = create_model(args, device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model from {args.checkpoint}")
    else:
        print("WARNING: No checkpoint loaded, using random weights")
    model.eval()

    positions = data['positions'].copy()
    sizes = data['sizes']
    nets = data['nets']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']
    N = data['N']

    macro_nets = [[] for _ in range(N)]
    for ni, net in enumerate(nets):
        for (node_idx, _, _) in net:
            macro_nets[node_idx].append(ni)

    solver = LNSSolver(
        positions=positions.copy(), sizes=sizes, nets=nets,
        edge_index=edge_index, congestion_weight=args.congestion_weight,
        subset_size=args.subset_size, window_fraction=args.window_fraction,
        cpsat_time_limit=5.0, seed=args.seed,
    )

    # Predict hints once from the fixed state
    with torch.no_grad():
        nf = compute_node_features_6d(positions, sizes, nets, macro_nets)
        nf_t = torch.tensor(nf, dtype=torch.float32, device=device)
        pos_t = torch.tensor(positions, dtype=torch.float32, device=device)
        sz_t = torch.tensor(sizes, dtype=torch.float32, device=device)
        ei_t = torch.tensor(edge_index, dtype=torch.long, device=device)
        ea_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)
        out = model(nf_t, pos_t, sz_t, ei_t, ea_t)
        delta_pred = out['displacement_pred'].cpu().numpy()
    hint_positions = positions + delta_pred

    time_limits = [float(t) for t in args.sweep_times.split(',')]
    n_subproblems = args.n_iterations
    strategies = ['random', 'worst_hpwl', 'congestion', 'connected']
    base_hpwl = compute_net_hpwl(positions, sizes, nets)

    # Pre-generate subsets (same for all time limits)
    subsets = []
    for i in range(n_subproblems):
        strategy = strategies[i % len(strategies)]
        k = min(args.subset_size, N)
        subsets.append(solver.get_neighborhood(strategy, k))

    print(f"\n--- Sweep: {n_subproblems} subproblems x {len(time_limits)} time limits ---")
    print(f"  Base HPWL: {base_hpwl:.4f}")

    results = {}
    for tl in time_limits:
        nohint_deltas = []
        hint_deltas = []
        hint_wins = 0

        for i, subset in enumerate(subsets):
            # No-hint
            nh_result = solve_subset(
                positions, sizes, nets, subset,
                time_limit=tl, window_fraction=solver.window_fraction,
            )
            nh_delta = 0.0
            if nh_result is not None:
                nh_delta = base_hpwl - compute_net_hpwl(nh_result, sizes, nets)
            nohint_deltas.append(nh_delta)

            # Hint
            h_result = solve_subset_guided(
                positions, sizes, nets, subset,
                time_limit=tl, window_fraction=solver.window_fraction,
                hint_positions=hint_positions,
            )
            h_delta = 0.0
            if h_result['new_positions'] is not None:
                h_delta = base_hpwl - compute_net_hpwl(h_result['new_positions'], sizes, nets)
            hint_deltas.append(h_delta)

            if h_delta > nh_delta:
                hint_wins += 1

        results[tl] = {
            'mean_nohint': np.mean(nohint_deltas),
            'mean_hint': np.mean(hint_deltas),
            'hint_wins': hint_wins,
            'n': n_subproblems,
        }
        print(f"  t={tl:5.1f}s: nohint_delta={np.mean(nohint_deltas):+.4f}  "
              f"hint_delta={np.mean(hint_deltas):+.4f}  "
              f"hint_wins={hint_wins}/{n_subproblems}")

    # Summary table
    print(f"\n{'='*65}")
    print(f"TIME-LIMIT SWEEP ({args.circuit}, {n_subproblems} subproblems)")
    print(f"{'='*65}")
    print(f"{'Time(s)':>8s} {'NoHint Δ':>12s} {'Hint Δ':>12s} "
          f"{'Hint Wins':>10s} {'Winner':>10s}")
    print(f"{'-'*65}")
    for tl in time_limits:
        r = results[tl]
        winner = 'HINT' if r['mean_hint'] > r['mean_nohint'] else 'NO-HINT'
        print(f"{tl:>8.1f} {r['mean_nohint']:>+12.4f} {r['mean_hint']:>+12.4f} "
              f"{r['hint_wins']:>4d}/{r['n']:<4d}  {winner:>10s}")
    print(f"{'='*65}")


# ============================================================================
# Mode: ablation
# ============================================================================

def cmd_ablation(args):
    """Compare topology_only / spatial_only / dual / no_hint.

    Runs all 4 conditions and prints a final comparison table.
    no_hint baseline uses solve_subset (no model, no hints) for paired comparison.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    # --- No-hint baseline (no model at all) ---
    print(f"\n{'='*60}")
    print(f"Ablation: no_hint (baseline)")
    print(f"{'='*60}")

    # Run eval with a dummy model that predicts zero displacement
    args.model_mode = 'topology_only'  # architecture doesn't matter for no-hint
    dummy_model = create_model(args, device)
    # Zero out displacement head so predictions are exactly zero
    with torch.no_grad():
        for p in dummy_model.displacement_head.parameters():
            p.zero_()
    summary = cmd_eval(args, model=dummy_model)
    # For no-hint baseline, the relevant number is final_shared_hpwl
    # (hint with zero displacement = same as no hint, since AddHint uses current pos)
    results['no_hint'] = summary

    # --- Train and evaluate each learned mode ---
    for mode in ['topology_only', 'spatial_only', 'dual']:
        print(f"\n{'='*60}")
        print(f"Ablation: {mode}")
        print(f"{'='*60}")

        args.model_mode = mode
        args.checkpoint_dir = os.path.join(args.ablation_dir, f'ckpt_{mode}')

        # Train
        cmd_train(args)

        # Eval
        args.checkpoint = os.path.join(args.checkpoint_dir, 'best.pt')
        summary = cmd_eval(args)
        results[mode] = summary

    # --- Final comparison table ---
    print(f"\n{'='*78}")
    print(f"ABLATION RESULTS ({args.circuit}, {args.n_iterations} iterations)")
    print(f"{'='*78}")
    print(f"{'Mode':<18s} {'NoHint HPWL':>12s} {'Hint HPWL':>12s} "
          f"{'Improved':>10s} {'Mean Δ':>10s} {'Time(s)':>8s}")
    print(f"{'-'*78}")
    for mode_name in ['no_hint', 'topology_only', 'spatial_only', 'dual']:
        r = results.get(mode_name)
        if r is None:
            continue
        # For no_hint baseline, hint HPWL is same as nohint (zero displacement)
        hint_hpwl_str = f"{r['final_hint_hpwl']:>12.4f}" if mode_name != 'no_hint' else f"{'N/A':>12s}"
        hint_impr_str = f"{r['hint_improved']:>4d}/{r['total_iters']:<4d}" if mode_name != 'no_hint' else f"{'N/A':>10s}"
        hint_delta_str = f"{r['mean_hint_delta']:>10.6f}" if mode_name != 'no_hint' else f"{'N/A':>10s}"
        hint_time_str = f"{r['total_hint_time']:>8.1f}" if mode_name != 'no_hint' else f"{'N/A':>8s}"
        print(f"{mode_name:<18s} "
              f"{r['final_nohint_hpwl']:>12.4f} "
              f"{hint_hpwl_str} "
              f"{hint_impr_str} "
              f"{hint_delta_str} "
              f"{hint_time_str}")
    print(f"{'='*78}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NetSpatialGNN — Stage 1: Supervised Displacement Warm-Start')

    parser.add_argument('--mode', type=str, required=True,
                        choices=['collect', 'train', 'eval', 'ablation', 'sweep'])

    # Data
    parser.add_argument('--circuit', type=str, default='ibm01')
    parser.add_argument('--circuits', type=str, default='ibm01',
                        help='Comma-separated circuits for training')
    parser.add_argument('--benchmark_base', type=str, default='benchmarks')
    parser.add_argument('--max_nodes', type=int, default=None)

    # LNS parameters (for collect/eval)
    parser.add_argument('--n_iterations', type=int, default=500)
    parser.add_argument('--subset_size', type=int, default=30)
    parser.add_argument('--window_fraction', type=float, default=0.15)
    parser.add_argument('--cpsat_time_limit', type=float, default=5.0)
    parser.add_argument('--congestion_weight', type=float, default=0.0)

    # Legalization
    parser.add_argument('--legalize_time_limit', type=float, default=60.0)
    parser.add_argument('--legalize_window', type=float, default=0.3)
    parser.add_argument('--skip_legalize', action='store_true')

    # Model architecture
    parser.add_argument('--model_mode', type=str, default='dual',
                        choices=['topology_only', 'spatial_only', 'dual'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--k_spatial', type=int, default=8)
    parser.add_argument('--grid_size', type=int, default=32)

    # Training
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Min delta_cost to include trace in training (0=all improving)')

    # Paths
    parser.add_argument('--trace_dir', type=str, default='traces')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_netspatial')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints_lns')
    parser.add_argument('--ablation_dir', type=str, default='ablation_netspatial')

    # Sweep
    parser.add_argument('--sweep_times', type=str, default='0.2,0.5,1.0,2.0,5.0',
                        help='Comma-separated CP-SAT time limits for sweep mode')

    # Misc
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == 'collect':
        cmd_collect(args)
    elif args.mode == 'train':
        cmd_train(args)
    elif args.mode == 'eval':
        cmd_eval(args)
    elif args.mode == 'ablation':
        cmd_ablation(args)
    elif args.mode == 'sweep':
        cmd_sweep(args)


if __name__ == '__main__':
    main()
