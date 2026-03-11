"""
ML-Guided LNS Runner — Neural Neighborhood Search

Modes:
  1. collect:  Run LNS and collect solver traces (mixed heuristic + random subsets)
  2. train:    Train NeighborhoodGNN on traces (Stage B: weighted BCE warm-start)
  3. reinforce: Online REINFORCE fine-tuning with value baseline (Stage C)
  4. deploy:   Run ML-guided LNS with trained model (inference, confidence fallback)
  5. ablation: Compare conditions wall-clock matched

Usage:
    # Phase 1: Collect solver traces from pure LNS
    python run_ml_lns.py collect --circuit ibm01 --n_iterations 500

    # Phase 2: Warm-start GNN on collected traces (weighted BCE)
    python run_ml_lns.py train --experience_paths traces/ibm01.pkl --epochs 50

    # Phase 3 (optional): REINFORCE fine-tuning online
    python run_ml_lns.py reinforce --circuit ibm01 --model_path checkpoints_ml/neighborhood_best.pt

    # Phase 4: Deploy ML-guided LNS
    python run_ml_lns.py deploy --circuit ibm01 --model_path checkpoints_ml/neighborhood_best.pt

    # Ablation: Compare all conditions
    python run_ml_lns.py ablation --circuit ibm01 --model_path checkpoints_ml/neighborhood_best.pt
"""

import argparse
import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from benchmark_loader import load_bookshelf_circuit
from cpsat_solver import legalize, compute_net_hpwl, check_overlap, check_boundary, solve_subset
from lns_solver import LNSSolver
from ml_lns_solver import MLGuidedLNSSolver
from trust_region_model import NeighborhoodGNN, build_node_features
from experience_buffer import ExperienceBuffer


def load_circuit_data(args):
    """Load circuit and optionally legalize."""
    circuit_dir = os.path.join(
        args.benchmark_base, "iccad04", "extracted", args.circuit)
    data = load_bookshelf_circuit(
        circuit_dir, args.circuit,
        macros_only=True,
        max_nodes=args.max_nodes,
        seed=args.seed,
    )

    positions = data['positions']
    sizes = data['node_features']
    nets = data['nets']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']

    N = data['n_components']
    print(f"Loaded {args.circuit}: {N} macros, {len(nets)} nets, "
          f"{edge_index.shape[1]} edges")

    ref_hpwl = compute_net_hpwl(positions, sizes, nets)
    _, ref_ov_pairs = check_overlap(positions, sizes)
    print(f"  Reference HPWL: {ref_hpwl:.4f}, overlapping pairs: {ref_ov_pairs}")

    # Legalize if needed
    if not args.skip_legalize and ref_ov_pairs > 0:
        print(f"  Legalizing (time_limit={args.legalize_time_limit}s)...")
        legal_pos = legalize(
            positions, sizes,
            time_limit=args.legalize_time_limit,
            window_fraction=args.legalize_window,
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
    }


def create_model(args, device):
    """Create NeighborhoodGNN with args."""
    return NeighborhoodGNN(
        input_dim=14,
        edge_dim=4,
        hidden_dim=args.hidden_dim,
        n_message_passes=args.n_message_passes,
    ).to(device)


def load_model_checkpoint(model, path, device):
    """Load model checkpoint. Returns epoch number."""
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        print(f"Loaded model from {path} (epoch {epoch})")
        return epoch
    else:
        print(f"WARNING: No model loaded — using random initialization")
        return 0


# ===========================================================================
# Mode: collect — Run pure LNS and save solver traces
# ===========================================================================

def cmd_collect(args):
    """Collect solver traces from LNS for offline training."""
    data = load_circuit_data(args)
    buf = ExperienceBuffer(max_size=args.buffer_size)

    solver = MLGuidedLNSSolver(
        positions=data['positions'],
        sizes=data['sizes'],
        nets=data['nets'],
        edge_index=data['edge_index'],
        edge_attr=data['edge_attr'],
        model=None,  # pure heuristic LNS
        use_learned_subset=False,
        experience_buffer=buf,
        circuit_id=args.circuit,
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        n_iterations=args.n_iterations,
        seed=args.seed,
    )

    result = solver.solve(
        n_iterations=args.n_iterations,
        log_every=args.log_every,
    )

    # Save traces
    os.makedirs(args.trace_dir, exist_ok=True)
    trace_path = os.path.join(args.trace_dir, f'{args.circuit}.pkl')
    buf.save(trace_path)
    stats = buf.stats()
    print(f"\nSaved {stats['size']} traces to {trace_path}")
    print(f"  Feasible rate: {stats['feasible_rate']:.2f}")
    print(f"  Mean improvement: {stats['mean_improvement']:.6f}")
    if 'strategies' in stats:
        print(f"  Strategy breakdown:")
        for s, info in stats['strategies'].items():
            print(f"    {s}: {info['count']} attempts, {info['improving']} improving")

    # Save best placement
    os.makedirs(args.save_dir, exist_ok=True)
    np.savez(
        os.path.join(args.save_dir, f'{args.circuit}_collect_best.npz'),
        positions=result['best_positions'],
        sizes=data['sizes'],
        hpwl=result['best_hpwl'],
        ref_hpwl=data['ref_hpwl'],
    )


# ===========================================================================
# Mode: train — Stage B: Weighted BCE warm-start on improving traces
# ===========================================================================

def cmd_train(args):
    """
    Train NeighborhoodGNN on collected solver traces.

    Loss: Weighted BCE on subset membership.
    - For each experience: subset_mask = binary(selected macros), weight ∝ max(0, delta_cost/time)
    - The GNN learns which macros, when included in a subset, led to improvements.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Training on {device}")

    # Load experience
    buf = ExperienceBuffer(max_size=args.buffer_size)
    for path in args.experience_paths:
        print(f"Loading traces from {path}")
        buf.load(path)
    stats = buf.stats()
    print(f"  Total: {stats['size']} traces, {stats['n_circuits']} circuits")
    print(f"  Feasible rate: {stats['feasible_rate']:.2f}")
    print(f"  Mean improvement: {stats['mean_improvement']:.6f}")

    # Create model
    model = create_model(args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(args.seed)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params} parameters")
    print(f"Training for {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"Loss: Weighted BCE on subset membership\n")

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        n_batches = max(1, stats['size'] // args.batch_size)
        for _ in range(n_batches):
            batch = buf.sample_batch(
                batch_size=args.batch_size,
                feasible_only=True,
                cost_aware=True,
                rng=rng,
            )
            if len(batch) == 0:
                continue

            targets = buf.get_subset_targets(batch)
            if len(targets['subset_masks']) == 0:
                continue

            # Train on each experience individually (variable graph sizes)
            batch_loss = 0.0
            n_valid = 0
            optimizer.zero_grad()

            for i, exp in enumerate(batch):
                if exp['node_features'] is None:
                    continue
                if i >= len(targets['subset_masks']):
                    break

                nf = torch.tensor(exp['node_features'], dtype=torch.float32, device=device)
                ei = torch.tensor(exp['edge_index'], dtype=torch.long, device=device)
                ea = torch.tensor(exp['edge_attr'], dtype=torch.float32, device=device)

                outputs = model(nf, ei, ea)
                logits = outputs['subset_logits']  # (V,)

                mask_target = torch.tensor(
                    targets['subset_masks'][i], dtype=torch.float32, device=device)
                weight = float(targets['weights'][i])

                # Weighted BCE: model learns subset membership from improving traces
                loss = weight * F.binary_cross_entropy_with_logits(
                    logits, mask_target, reduction='mean')
                loss.backward()
                batch_loss += loss.item()
                n_valid += 1

            if n_valid > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(batch_loss / n_valid)

        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        if mean_loss < best_loss:
            best_loss = mean_loss
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = os.path.join(args.save_dir, 'neighborhood_best.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'args': vars(args),
            }, model_path)

        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch+1:4d}/{args.epochs}: "
                  f"loss={mean_loss:.6f} (best={best_loss:.6f})")

    # Save final model
    model_path = os.path.join(args.save_dir, 'neighborhood_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'loss': mean_loss,
        'args': vars(args),
    }, model_path)
    print(f"\nSaved model to {model_path}")


# ===========================================================================
# Mode: reinforce — Stage C: Online REINFORCE fine-tuning
# ===========================================================================

def cmd_reinforce(args):
    """
    Online REINFORCE fine-tuning of NeighborhoodGNN.

    Runs ML-guided LNS with exploration enabled. After each iteration,
    computes REINFORCE gradient:
        ∇J = (R - V(s)) * ∇ log P(S|s)
    where R = max(0, delta_cost) / solve_time, V(s) = value baseline.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data = load_circuit_data(args)

    # Load pre-trained model (warm-start from Stage B)
    model = create_model(args, device)
    load_model_checkpoint(model, args.model_path, device)
    model.train()

    # Separate param groups: policy params (backbone + subset_head) vs value params
    value_params = set(id(p) for p in model.value_mlp.parameters())
    policy_params = [p for p in model.parameters() if id(p) not in value_params]
    optimizer = optim.Adam([
        {'params': policy_params, 'lr': args.lr},
        {'params': model.value_mlp.parameters(), 'lr': args.lr * 3},
    ])

    buf = ExperienceBuffer(max_size=args.buffer_size)

    solver = MLGuidedLNSSolver(
        positions=data['positions'],
        sizes=data['sizes'],
        nets=data['nets'],
        edge_index=data['edge_index'],
        edge_attr=data['edge_attr'],
        model=model,
        device=device,
        use_learned_subset=True,
        confidence_threshold=0.0,  # always use learned (need on-policy)
        explore_temperature=args.explore_temperature,
        explore=True,  # Gumbel-top-k exploration
        experience_buffer=buf,
        circuit_id=args.circuit,
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        n_iterations=args.n_iterations,
        seed=args.seed,
    )

    print(f"\nREINFORCE fine-tuning: {args.n_iterations} iterations")
    print(f"  explore_temperature={args.explore_temperature}")
    print(f"  update_every={args.update_every}")
    print()

    edge_index_t = torch.tensor(data['edge_index'], dtype=torch.long, device=device)
    edge_attr_t = torch.tensor(data['edge_attr'], dtype=torch.float32, device=device)

    # Collect rollout buffer for mini-batch updates
    rollout_nf = []
    rollout_log_probs = []
    rollout_rewards = []
    rollout_values = []

    for i in range(args.n_iterations):
        # Compute node features
        node_features = solver._compute_node_features()
        nf_t = torch.tensor(node_features, dtype=torch.float32, device=device)

        # Forward pass WITH gradients
        model.train()
        outputs = model(nf_t, edge_index_t, edge_attr_t)
        scores = outputs['subset_scores']
        value = outputs['value']  # (1,)

        k = min(solver.subset_size, solver.N)
        indices, log_prob = model.select_subset(
            scores, k,
            temperature=args.explore_temperature,
            explore=True,
        )
        subset = indices.detach().cpu().numpy()

        # CP-SAT solve (detached)
        t0 = time.time()
        new_positions = solve_subset(
            solver.current_pos, solver.sizes, solver.nets, subset,
            time_limit=solver.cpsat_time_limit,
            window_fraction=solver.window_fraction,
        )
        dt = time.time() - t0

        # Compute reward (before state update changes current_cost)
        if new_positions is not None:
            new_hpwl = compute_net_hpwl(new_positions, solver.sizes, solver.nets)
            new_cost = solver._compute_cost(new_positions, new_hpwl)
            delta_cost = solver.current_cost - new_cost  # positive = improvement
            reward = max(0.0, delta_cost) / max(dt, 0.1)
        else:
            reward = 0.0

        # Store for batch update
        rollout_log_probs.append(log_prob)
        rollout_rewards.append(reward)
        rollout_values.append(value.squeeze())

        # Update solver state: acceptance, SA, window/subset adaptation
        solver.update_state(new_positions, subset, 'learned')

        # Batch update
        if (i + 1) % args.update_every == 0 and len(rollout_log_probs) > 0:
            log_probs_t = torch.stack(rollout_log_probs)
            rewards_t = torch.tensor(rollout_rewards, dtype=torch.float32, device=device)
            values_t = torch.stack(rollout_values)

            # Normalize rewards
            if rewards_t.std() > 1e-8:
                rewards_norm = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
            else:
                rewards_norm = rewards_t

            # REINFORCE: policy gradient with value baseline
            advantage = rewards_norm - values_t.detach()
            policy_loss = -(log_probs_t * advantage).mean()

            # Value loss
            value_loss = F.mse_loss(values_t, rewards_norm.detach())

            # Combined backward (shared graph between policy_loss and value_loss)
            total_loss = policy_loss + value_loss
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (i + 1) % (args.update_every * 5) == 0 or i == args.n_iterations - 1:
                mean_r = rewards_t.mean().item()
                print(f"  [{i+1:4d}/{args.n_iterations}] "
                      f"best={solver.best_hpwl:.4f} "
                      f"reward={mean_r:.4f} "
                      f"policy_loss={policy_loss.item():.4f} "
                      f"value_loss={value_loss.item():.4f}")

            rollout_log_probs = []
            rollout_rewards = []
            rollout_values = []

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, 'neighborhood_reinforce.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.n_iterations,
        'best_hpwl': solver.best_hpwl,
        'args': vars(args),
    }, model_path)

    ratio = solver.best_hpwl / max(data['ref_hpwl'], 1e-8)
    print(f"\nREINFORCE complete. Best HPWL={solver.best_hpwl:.4f}, "
          f"ref={data['ref_hpwl']:.4f}, ratio={ratio:.3f}")
    print(f"Saved to {model_path}")


# ===========================================================================
# Mode: deploy — Run ML-guided LNS with trained model
# ===========================================================================

def cmd_deploy(args):
    """Run ML-guided LNS with a trained NeighborhoodGNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data = load_circuit_data(args)

    model = create_model(args, device)
    load_model_checkpoint(model, args.model_path, device)
    model.eval()

    buf = ExperienceBuffer(max_size=args.buffer_size) if args.collect_traces else None

    solver = MLGuidedLNSSolver(
        positions=data['positions'],
        sizes=data['sizes'],
        nets=data['nets'],
        edge_index=data['edge_index'],
        edge_attr=data['edge_attr'],
        model=model,
        device=device,
        use_learned_subset=args.use_learned_subset,
        confidence_threshold=args.confidence_threshold,
        explore=False,  # inference mode
        experience_buffer=buf,
        circuit_id=args.circuit,
        congestion_weight=args.congestion_weight,
        subset_size=args.subset_size,
        window_fraction=args.window_fraction,
        cpsat_time_limit=args.cpsat_time_limit,
        n_iterations=args.n_iterations,
        seed=args.seed,
    )

    result = solver.solve(
        n_iterations=args.n_iterations,
        log_every=args.log_every,
    )

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    tag = 'ml' if args.use_learned_subset else 'baseline'
    np.savez(
        os.path.join(args.save_dir, f'{args.circuit}_{tag}_best.npz'),
        positions=result['best_positions'],
        sizes=data['sizes'],
        hpwl=result['best_hpwl'],
        ref_hpwl=data['ref_hpwl'],
    )

    if buf is not None:
        trace_path = os.path.join(args.trace_dir, f'{args.circuit}_deploy.pkl')
        os.makedirs(args.trace_dir, exist_ok=True)
        buf.save(trace_path)

    ratio = result['best_hpwl'] / max(data['ref_hpwl'], 1e-8)
    print(f"\nResult: HPWL={result['best_hpwl']:.4f}, "
          f"ref={data['ref_hpwl']:.4f}, ratio={ratio:.3f}")


# ===========================================================================
# Mode: ablation — Compare conditions wall-clock matched
# ===========================================================================

def cmd_ablation(args):
    """
    Run ablation study with wall-clock matched conditions.

    Conditions:
    1. pure_lns: Standard heuristic LNS (no ML)
    2. random_subset: Random subset selection (ML-free baseline)
    3. learned_no_fallback: NeighborhoodGNN subset, no confidence fallback
    4. learned_with_fallback: NeighborhoodGNN subset + confidence-gated fallback
    """
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data = load_circuit_data(args)

    model = create_model(args, device)
    load_model_checkpoint(model, args.model_path, device)
    model.eval()

    conditions = [
        # (name, use_learned_subset, model, confidence_threshold)
        ('pure_lns',              False, None,  0.0),
        ('random_subset',         False, None,  0.0),
        ('learned_no_fallback',   True,  model, 0.0),
        ('learned_with_fallback', True,  model, args.confidence_threshold),
    ]

    results = {}

    for name, use_learned, m, conf_thresh in conditions:
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"{'='*60}")

        # For random_subset, we override strategy selection
        if name == 'random_subset':
            solver = MLGuidedLNSSolver(
                positions=data['positions'].copy(),
                sizes=data['sizes'],
                nets=data['nets'],
                edge_index=data['edge_index'],
                edge_attr=data['edge_attr'],
                model=None,
                device=device,
                use_learned_subset=False,
                circuit_id=args.circuit,
                congestion_weight=args.congestion_weight,
                subset_size=args.subset_size,
                window_fraction=args.window_fraction,
                cpsat_time_limit=args.cpsat_time_limit,
                n_iterations=args.n_iterations,
                seed=args.seed,
            )
            # Force random-only strategy
            solver.strategies = ['random']
        else:
            solver = MLGuidedLNSSolver(
                positions=data['positions'].copy(),
                sizes=data['sizes'],
                nets=data['nets'],
                edge_index=data['edge_index'],
                edge_attr=data['edge_attr'],
                model=m,
                device=device,
                use_learned_subset=use_learned,
                confidence_threshold=conf_thresh,
                explore=False,  # inference mode
                circuit_id=args.circuit,
                congestion_weight=args.congestion_weight,
                subset_size=args.subset_size,
                window_fraction=args.window_fraction,
                cpsat_time_limit=args.cpsat_time_limit,
                n_iterations=args.n_iterations,
                seed=args.seed,
            )

        t0 = time.time()
        result = solver.solve(
            n_iterations=args.n_iterations,
            log_every=args.log_every,
        )
        wall_time = time.time() - t0

        overlap, n_ov = check_overlap(result['best_positions'], data['sizes'])
        boundary = check_boundary(result['best_positions'], data['sizes'])
        ratio = result['best_hpwl'] / max(data['ref_hpwl'], 1e-8)

        results[name] = {
            'best_hpwl': float(result['best_hpwl']),
            'ref_hpwl': float(data['ref_hpwl']),
            'ratio': float(ratio),
            'overlap': float(overlap),
            'overlap_pairs': int(n_ov),
            'boundary': float(boundary),
            'wall_time': float(wall_time),
            'n_improved': int(solver.n_improved),
            'n_infeasible': int(solver.n_infeasible),
            'n_learned': int(solver.n_learned_subsets),
            'n_fallback': int(solver.n_fallback_subsets),
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"Ablation Summary: {args.circuit}")
    print(f"{'='*80}")
    print(f"{'Condition':<25} {'HPWL':>10} {'Ratio':>8} {'Overlap':>10} "
          f"{'Wall(s)':>8} {'Improv':>7} {'Infeas':>7} {'Learned':>8}")
    print('-' * 85)
    for name, r in results.items():
        learned_str = f"{r['n_learned']}/{r['n_learned']+r['n_fallback']}" if r['n_learned'] > 0 else '-'
        print(f"{name:<25} {r['best_hpwl']:10.4f} {r['ratio']:8.3f} "
              f"{r['overlap']:10.6f} {r['wall_time']:8.1f} "
              f"{r['n_improved']:7d} {r['n_infeasible']:7d} {learned_str:>8}")

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f'{args.circuit}_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation results to {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Neural Neighborhood Search for Chip Placement')
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')

    # Common args
    def add_common_args(p):
        p.add_argument('--circuit', type=str, default='ibm01')
        p.add_argument('--benchmark_base', type=str, default='benchmarks')
        p.add_argument('--max_nodes', type=int, default=None)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--cpu', action='store_true')
        # Legalization
        p.add_argument('--skip_legalize', action='store_true')
        p.add_argument('--legalize_time_limit', type=float, default=60.0)
        p.add_argument('--legalize_window', type=float, default=0.3)
        # LNS
        p.add_argument('--n_iterations', type=int, default=200)
        p.add_argument('--subset_size', type=int, default=30)
        p.add_argument('--window_fraction', type=float, default=0.15)
        p.add_argument('--cpsat_time_limit', type=float, default=5.0)
        p.add_argument('--congestion_weight', type=float, default=0.0)
        p.add_argument('--log_every', type=int, default=10)
        # Model
        p.add_argument('--hidden_dim', type=int, default=64)
        p.add_argument('--n_message_passes', type=int, default=5)
        # Output
        p.add_argument('--save_dir', type=str, default='checkpoints_ml')
        p.add_argument('--buffer_size', type=int, default=50000)

    # Collect
    p_collect = subparsers.add_parser('collect', help='Collect solver traces')
    add_common_args(p_collect)
    p_collect.add_argument('--trace_dir', type=str, default='traces')

    # Train (Stage B: BCE warm-start)
    p_train = subparsers.add_parser('train', help='Train GNN on traces (BCE warm-start)')
    p_train.add_argument('--experience_paths', nargs='+', required=True)
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch_size', type=int, default=32)
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--cpu', action='store_true')
    p_train.add_argument('--hidden_dim', type=int, default=64)
    p_train.add_argument('--n_message_passes', type=int, default=5)
    p_train.add_argument('--save_dir', type=str, default='checkpoints_ml')
    p_train.add_argument('--buffer_size', type=int, default=50000)

    # Reinforce (Stage C: online REINFORCE)
    p_rl = subparsers.add_parser('reinforce', help='REINFORCE fine-tuning (online)')
    add_common_args(p_rl)
    p_rl.add_argument('--model_path', type=str, default='checkpoints_ml/neighborhood_best.pt')
    p_rl.add_argument('--lr', type=float, default=1e-4)
    p_rl.add_argument('--explore_temperature', type=float, default=0.5)
    p_rl.add_argument('--update_every', type=int, default=10)
    p_rl.add_argument('--trace_dir', type=str, default='traces')

    # Deploy
    p_deploy = subparsers.add_parser('deploy', help='Deploy ML-guided LNS')
    add_common_args(p_deploy)
    p_deploy.add_argument('--model_path', type=str, default='checkpoints_ml/neighborhood_best.pt')
    p_deploy.add_argument('--use_learned_subset', action='store_true', default=True)
    p_deploy.add_argument('--no_learned_subset', dest='use_learned_subset', action='store_false')
    p_deploy.add_argument('--confidence_threshold', type=float, default=0.05)
    p_deploy.add_argument('--collect_traces', action='store_true')
    p_deploy.add_argument('--trace_dir', type=str, default='traces')

    # Ablation
    p_ablation = subparsers.add_parser('ablation', help='Run ablation study')
    add_common_args(p_ablation)
    p_ablation.add_argument('--model_path', type=str, default='checkpoints_ml/neighborhood_best.pt')
    p_ablation.add_argument('--confidence_threshold', type=float, default=0.05)

    args = parser.parse_args()

    if args.mode == 'collect':
        cmd_collect(args)
    elif args.mode == 'train':
        cmd_train(args)
    elif args.mode == 'reinforce':
        cmd_reinforce(args)
    elif args.mode == 'deploy':
        cmd_deploy(args)
    elif args.mode == 'ablation':
        cmd_ablation(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
