"""
ALNS + CP-SAT Parameter Sweep

Tunes the pure solver baseline under equal wall-clock budget.
Records anytime HPWL curves, final metrics, and per-config statistics.

Usage:
    # Phase 1: coarse screening (main geometric knobs)
    python sweep_alns.py --phase coarse --wall_clock 300

    # Phase 2: narrowed follow-up on best configs
    python sweep_alns.py --phase fine --wall_clock 600

    # Phase 3: adaptation logic
    python sweep_alns.py --phase adapt --wall_clock 300

    # Single config test
    python sweep_alns.py --phase single --subset_size 30 --window_fraction 0.15 --cpsat_time_limit 5.0 --wall_clock 60
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_loader import load_bookshelf_circuit
from cpsat_solver import legalize, compute_net_hpwl, check_overlap, check_boundary
from lns_solver import LNSSolver, compute_rudy_np


# ── Circuit sets ──

TRAIN_CIRCUITS = ['ibm01', 'ibm03', 'ibm07']
VAL_CIRCUITS = ['ibm08', 'ibm09']
ALL_CIRCUITS = TRAIN_CIRCUITS + VAL_CIRCUITS

SEEDS = [42, 123, 7]


def load_circuit(circuit_name, benchmark_base='benchmarks', seed=42):
    """Load and legalize a benchmark circuit. Returns (positions, sizes, nets, edge_index, metadata)."""
    circuit_dir = os.path.join(benchmark_base, 'iccad04', 'extracted', circuit_name)
    data = load_bookshelf_circuit(circuit_dir, circuit_name, macros_only=True, seed=seed)

    positions = data['positions']
    sizes = data['node_features']
    nets = data['nets']
    edge_index = data['edge_index']
    N = data['n_components']

    ref_hpwl = compute_net_hpwl(positions, sizes, nets)
    _, ref_ov_pairs = check_overlap(positions, sizes)

    return {
        'positions': positions,
        'sizes': sizes,
        'nets': nets,
        'edge_index': edge_index,
        'N': N,
        'ref_hpwl': ref_hpwl,
        'ref_ov_pairs': ref_ov_pairs,
        'circuit_name': circuit_name,
    }


def legalize_circuit(circuit_data, time_limit=60.0, window_fraction=0.3, num_workers=4):
    """Legalize a circuit's reference placement. Returns legalized positions."""
    pos = circuit_data['positions']
    sizes = circuit_data['sizes']

    if circuit_data['ref_ov_pairs'] == 0:
        return pos.copy()

    legal_pos = legalize(pos, sizes, time_limit=time_limit, window_fraction=window_fraction,
                         num_workers=num_workers)
    if legal_pos is None:
        print(f"  WARNING: legalization failed for {circuit_data['circuit_name']}, using reference")
        return pos.copy()
    return legal_pos


def run_lns_wallclock(
    positions, sizes, nets, edge_index,
    subset_size, window_fraction, cpsat_time_limit,
    congestion_weight, plateau_threshold, adapt_threshold,
    adaptive, wall_clock, seed,
):
    """
    Run LNS for a fixed wall-clock budget.

    If adaptive=False, disable grow/shrink by setting thresholds very high.

    Returns dict with metrics and anytime curve.
    """
    if not adaptive:
        # Effectively disable adaptation
        plateau_threshold = 999999
        adapt_threshold = 999999

    solver = LNSSolver(
        positions=positions.copy(),
        sizes=sizes,
        nets=nets,
        edge_index=edge_index,
        congestion_weight=congestion_weight,
        subset_size=subset_size,
        window_fraction=window_fraction,
        cpsat_time_limit=cpsat_time_limit,
        plateau_threshold=plateau_threshold,
        adapt_threshold=adapt_threshold,
        seed=seed,
    )

    anytime_curve = []  # (elapsed_s, best_hpwl)
    n_iterations = 0
    n_accepted = 0
    n_improved = 0
    n_infeasible = 0
    total_solve_time = 0.0

    start_time = time.time()
    anytime_curve.append((0.0, solver.best_hpwl))

    while True:
        elapsed = time.time() - start_time
        if elapsed >= wall_clock:
            break

        metrics = solver.step()
        n_iterations += 1
        total_solve_time += metrics['time']

        if metrics['accepted']:
            n_accepted += 1
        if metrics['improved']:
            n_improved += 1
            anytime_curve.append((time.time() - start_time, solver.best_hpwl))
        if not metrics['accepted'] and metrics['delta_cost'] == 0.0 and not metrics['improved']:
            # Infeasible
            pass

    elapsed = time.time() - start_time
    anytime_curve.append((elapsed, solver.best_hpwl))

    # Final overlap/boundary check
    overlap_area, overlap_pairs = check_overlap(solver.best_pos, sizes)
    boundary = check_boundary(solver.best_pos, sizes)

    # Strategy stats
    strategy_stats = {}
    for s in solver.strategies:
        attempts = solver.strategy_attempts[s]
        successes = solver.strategy_successes[s]
        strategy_stats[s] = {
            'attempts': attempts,
            'successes': successes,
            'rate': successes / max(attempts, 1),
        }

    return {
        'best_hpwl': solver.best_hpwl,
        'initial_hpwl': anytime_curve[0][1],
        'n_iterations': n_iterations,
        'n_accepted': n_accepted,
        'n_improved': n_improved,
        'n_infeasible': solver.n_infeasible,
        'overlap_area': overlap_area,
        'overlap_pairs': overlap_pairs,
        'boundary': boundary,
        'elapsed_s': elapsed,
        'improvement_per_second': (anytime_curve[0][1] - solver.best_hpwl) / max(elapsed, 0.01),
        'anytime_curve': anytime_curve,
        'strategy_stats': strategy_stats,
    }


def generate_configs_coarse():
    """Phase 1: coarse screening of main geometric knobs."""
    configs = []
    for ss, wf, tl in itertools.product(
        [10, 20, 30, 40, 60],        # subset_size
        [0.05, 0.10, 0.15, 0.25, 0.40],  # window_fraction
        [0.1, 0.3, 1.0, 3.0, 5.0],   # cpsat_time_limit
    ):
        configs.append({
            'subset_size': ss,
            'window_fraction': wf,
            'cpsat_time_limit': tl,
            'congestion_weight': 0.0,
            'plateau_threshold': 20,
            'adapt_threshold': 30,
            'adaptive': True,
        })
    return configs


def generate_configs_trimmed():
    """Trimmed coarse sweep: ~38 configs filtered by feasibility constraint ss*wf <= volume_limit(tl)."""
    volume_limits = {0.1: 2.0, 0.3: 4.0, 1.0: 8.0, 3.0: 15.0, 5.0: 24.0}
    configs = []
    seen = set()
    for ss, wf, tl in itertools.product(
        [10, 20, 30, 40, 60],
        [0.05, 0.10, 0.15, 0.25],
        [0.1, 0.3, 1.0, 3.0, 5.0],
    ):
        if ss * wf > volume_limits[tl]:
            continue
        if tl == 0.1 and ss >= 30:  # overhead-dominated for large subsets
            continue
        key = (ss, wf, tl)
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            'subset_size': ss,
            'window_fraction': wf,
            'cpsat_time_limit': tl,
            'congestion_weight': 0.0,
            'plateau_threshold': 20,
            'adapt_threshold': 30,
            'adaptive': True,
        })
    return configs


def generate_configs_fine(top_configs):
    """Phase 2: fine-grained sweep around top configs from Phase 1."""
    configs = []
    for base in top_configs:
        ss = base['subset_size']
        wf = base['window_fraction']
        tl = base['cpsat_time_limit']

        # Vary each param around the base value
        for ss_delta in [-5, 0, 5]:
            for wf_mult in [0.75, 1.0, 1.25]:
                for tl_mult in [0.5, 1.0, 2.0]:
                    configs.append({
                        'subset_size': max(5, ss + ss_delta),
                        'window_fraction': round(max(0.02, wf * wf_mult), 3),
                        'cpsat_time_limit': round(max(0.05, tl * tl_mult), 2),
                        'congestion_weight': 0.0,
                        'plateau_threshold': 20,
                        'adapt_threshold': 30,
                        'adaptive': True,
                    })

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = (c['subset_size'], c['window_fraction'], c['cpsat_time_limit'])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def generate_configs_adapt():
    """Phase 3: sweep adaptation logic (uses best geometric config)."""
    configs = []
    # Will be paired with best geometric config from phase 1/2
    for adaptive in [True, False]:
        for pt in [10, 20, 40]:
            for at in [10, 20, 40]:
                if not adaptive and (pt != 10 or at != 10):
                    continue  # skip irrelevant combos when adaptive=off
                configs.append({
                    'adaptive': adaptive,
                    'plateau_threshold': pt,
                    'adapt_threshold': at,
                    'congestion_weight': 0.0,
                })
    return configs


def generate_configs_congestion():
    """Phase 4: congestion weight sweep."""
    configs = []
    for cw in [0.0, 0.05, 0.1, 0.2]:
        configs.append({
            'congestion_weight': cw,
        })
    return configs


def config_key(config):
    """Short string key for a config."""
    parts = []
    parts.append(f"ss{config['subset_size']}")
    parts.append(f"wf{config['window_fraction']}")
    parts.append(f"tl{config['cpsat_time_limit']}")
    if config.get('congestion_weight', 0) > 0:
        parts.append(f"cw{config['congestion_weight']}")
    if not config.get('adaptive', True):
        parts.append('noAdapt')
    else:
        parts.append(f"pt{config['plateau_threshold']}_at{config['adapt_threshold']}")
    return '_'.join(parts)


def run_sweep(
    configs, circuits, seeds, wall_clock, benchmark_base,
    legalize_time_limit, legalize_window, save_dir,
):
    """Run full parameter sweep."""
    os.makedirs(save_dir, exist_ok=True)

    # Load and legalize circuits once
    print(f"\n{'='*70}")
    print(f"Loading and legalizing {len(circuits)} circuits...")
    print(f"{'='*70}")

    circuit_cache = {}
    for circuit_name in circuits:
        print(f"\n  Loading {circuit_name}...")
        cdata = load_circuit(circuit_name, benchmark_base)
        print(f"    N={cdata['N']}, nets={len(cdata['nets'])}, ref_hpwl={cdata['ref_hpwl']:.4f}, overlaps={cdata['ref_ov_pairs']}")

        # Cache legalized positions to disk — same legal_pos used for all seeds (consistency)
        legal_path = os.path.join(save_dir, f'{circuit_name}_legal_pos.npy')
        if os.path.exists(legal_path):
            legal_pos = np.load(legal_path)
            legal_ov = check_overlap(legal_pos, cdata['sizes'])[1]
            if legal_ov > 0:
                print(f"    Cached legalization has {legal_ov} overlaps — re-legalizing...")
                os.remove(legal_path)
            else:
                print(f"    Loaded cached legalization")
        if not os.path.exists(legal_path):
            print(f"    Legalizing ({legalize_time_limit}s budget)...", end=' ', flush=True)
            t0 = time.time()
            legal_pos = legalize_circuit(cdata, legalize_time_limit, legalize_window,
                                         num_workers=4)
            print(f"done in {time.time() - t0:.1f}s")
            np.save(legal_path, legal_pos)

        legal_hpwl = compute_net_hpwl(legal_pos, cdata['sizes'], cdata['nets'])
        print(f"    post-legal HPWL: {legal_hpwl:.4f}")

        circuit_cache[circuit_name] = {
            **cdata,
            'legal_positions': legal_pos,
            'legal_hpwl': legal_hpwl,
        }

    # Auto-resume: always read from results_path if it exists
    results_path = os.path.join(save_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"\nResuming: {len(all_results)} existing results")
    else:
        all_results = []

    completed_keys = set()
    for r in all_results:
        k = (r['config_key'], r['circuit'], r['seed'])
        completed_keys.add(k)

    total_runs = len(configs) * len(circuits) * len(seeds)
    skipped = 0
    run_idx = 0

    print(f"\n{'='*70}")
    print(f"Sweep: {len(configs)} configs x {len(circuits)} circuits x {len(seeds)} seeds = {total_runs} runs")
    print(f"Wall clock per run: {wall_clock}s")
    print(f"Estimated total time: {total_runs * wall_clock / 3600:.1f}h (upper bound)")
    print(f"{'='*70}\n")

    for ci, config in enumerate(configs):
        ck = config_key(config)

        for circuit_name in circuits:
            cdata = circuit_cache[circuit_name]

            for seed in seeds:
                run_idx += 1
                k = (ck, circuit_name, seed)

                if k in completed_keys:
                    skipped += 1
                    continue

                print(f"[{run_idx}/{total_runs}] {ck} | {circuit_name} | seed={seed} ...", end=' ', flush=True)

                result = run_lns_wallclock(
                    positions=cdata['legal_positions'],
                    sizes=cdata['sizes'],
                    nets=cdata['nets'],
                    edge_index=cdata['edge_index'],
                    subset_size=config['subset_size'],
                    window_fraction=config['window_fraction'],
                    cpsat_time_limit=config['cpsat_time_limit'],
                    congestion_weight=config.get('congestion_weight', 0.0),
                    plateau_threshold=config.get('plateau_threshold', 20),
                    adapt_threshold=config.get('adapt_threshold', 30),
                    adaptive=config.get('adaptive', True),
                    wall_clock=wall_clock,
                    seed=seed,
                )

                # Record
                entry = {
                    'config_key': ck,
                    'config': config,
                    'circuit': circuit_name,
                    'seed': seed,
                    'N': cdata['N'],
                    'ref_hpwl': cdata['ref_hpwl'],
                    'legal_hpwl': cdata['legal_hpwl'],
                    'best_hpwl': result['best_hpwl'],
                    'ratio_vs_ref': result['best_hpwl'] / max(cdata['ref_hpwl'], 1e-8),
                    'ratio_vs_legal': result['best_hpwl'] / max(cdata['legal_hpwl'], 1e-8),
                    'n_iterations': result['n_iterations'],
                    'n_accepted': result['n_accepted'],
                    'n_improved': result['n_improved'],
                    'n_infeasible': result['n_infeasible'],
                    'overlap_pairs': result['overlap_pairs'],
                    'overlap_area': result['overlap_area'],
                    'boundary': result['boundary'],
                    'elapsed_s': result['elapsed_s'],
                    'improvement_per_second': result['improvement_per_second'],
                    'anytime_hpwl_10s': _anytime_at(result['anytime_curve'], 10),
                    'anytime_hpwl_30s': _anytime_at(result['anytime_curve'], 30),
                    'anytime_hpwl_60s': _anytime_at(result['anytime_curve'], 60),
                    'anytime_hpwl_120s': _anytime_at(result['anytime_curve'], 120),
                    'anytime_hpwl_300s': _anytime_at(result['anytime_curve'], 300),
                    'strategy_stats': result['strategy_stats'],
                }

                all_results.append(entry)

                ratio = entry['ratio_vs_ref']
                print(f"HPWL={result['best_hpwl']:.2f} ({ratio:.3f}x ref) "
                      f"iters={result['n_iterations']} "
                      f"improved={result['n_improved']} "
                      f"infeas={result['n_infeasible']}")

                # Save after each run (crash-safe)
                with open(results_path, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)

    if skipped > 0:
        print(f"\nSkipped {skipped} already-completed runs")

    # Print summary
    print_summary(all_results, circuits)
    return all_results


def _anytime_at(curve, t):
    """Get best HPWL at time t from anytime curve."""
    best = curve[0][1]
    for elapsed, hpwl in curve:
        if elapsed <= t:
            best = hpwl
        else:
            break
    return best


def print_summary(results, circuits):
    """Print ranked summary of configs."""
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*70}")

    # Group by config
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        by_config[r['config_key']].append(r)

    # Compute mean ratio_vs_ref per config (across circuits and seeds)
    config_scores = []
    for ck, runs in by_config.items():
        ratios = [r['ratio_vs_ref'] for r in runs]
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        mean_iters = np.mean([r['n_iterations'] for r in runs])
        mean_infeas = np.mean([r['n_infeasible'] for r in runs])
        mean_improved = np.mean([r['n_improved'] for r in runs])

        # Per-circuit mean ratio
        per_circuit = {}
        for c in circuits:
            c_runs = [r for r in runs if r['circuit'] == c]
            if c_runs:
                per_circuit[c] = np.mean([r['ratio_vs_ref'] for r in c_runs])

        config_scores.append({
            'config_key': ck,
            'config': runs[0]['config'],
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'mean_iters': mean_iters,
            'mean_infeas_rate': mean_infeas / max(mean_iters, 1),
            'mean_improved': mean_improved,
            'n_runs': len(runs),
            'per_circuit': per_circuit,
        })

    # Sort by mean ratio (lower = better)
    config_scores.sort(key=lambda x: x['mean_ratio'])

    # Print top 20
    print(f"\nTop 20 configs (by mean ratio vs reference, lower = better):\n")
    print(f"{'Rank':>4} {'Config':<45} {'Mean':>6} {'Std':>6} {'Iters':>6} {'Infeas%':>7} {'Impr':>5} ", end='')
    for c in circuits:
        print(f" {c:>7}", end='')
    print()
    print('-' * (80 + 8 * len(circuits)))

    for i, cs in enumerate(config_scores[:20]):
        print(f"{i+1:4d} {cs['config_key']:<45} {cs['mean_ratio']:6.3f} {cs['std_ratio']:6.3f} "
              f"{cs['mean_iters']:6.0f} {cs['mean_infeas_rate']:6.1%} {cs['mean_improved']:5.0f} ", end='')
        for c in circuits:
            if c in cs['per_circuit']:
                print(f" {cs['per_circuit'][c]:7.3f}", end='')
            else:
                print(f"     N/A", end='')
        print()

    # Print worst 5 for contrast
    if len(config_scores) > 25:
        print(f"\nWorst 5 configs:\n")
        for cs in config_scores[-5:]:
            print(f"     {cs['config_key']:<45} {cs['mean_ratio']:6.3f}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='ALNS + CP-SAT Parameter Sweep')

    parser.add_argument('--phase', type=str, default='trimmed',
                        choices=['trimmed', 'coarse', 'fine', 'adapt', 'congestion', 'single'],
                        help='Sweep phase')
    parser.add_argument('--wall_clock', type=float, default=300,
                        help='Wall-clock budget per run (seconds)')
    parser.add_argument('--benchmark_base', type=str, default='benchmarks',
                        help='Benchmark base directory')
    parser.add_argument('--save_dir', type=str, default='sweep_results',
                        help='Output directory')
    parser.add_argument('--circuits', type=str, nargs='+', default=None,
                        help='Override circuit list')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Override seed list')
    parser.add_argument('--legalize_time_limit', type=float, default=60.0,
                        help='Legalization time limit')
    parser.add_argument('--legalize_window', type=float, default=0.3,
                        help='Legalization window')
    parser.add_argument('--train_only', action='store_true',
                        help='Only run on train circuits')

    # Single-config params
    parser.add_argument('--subset_size', type=int, default=30)
    parser.add_argument('--window_fraction', type=float, default=0.15)
    parser.add_argument('--cpsat_time_limit', type=float, default=5.0)
    parser.add_argument('--congestion_weight', type=float, default=0.0)
    parser.add_argument('--plateau_threshold', type=int, default=20)
    parser.add_argument('--adapt_threshold', type=int, default=30)
    parser.add_argument('--adaptive', action='store_true', default=True)
    parser.add_argument('--no_adaptive', dest='adaptive', action='store_false')

    # Fine phase: path to coarse results
    parser.add_argument('--coarse_results', type=str, default=None,
                        help='Path to coarse results.json for fine phase')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top configs from coarse to expand in fine phase')

    # Adapt/congestion phase: base config
    parser.add_argument('--base_subset_size', type=int, default=None)
    parser.add_argument('--base_window_fraction', type=float, default=None)
    parser.add_argument('--base_cpsat_time_limit', type=float, default=None)

    args = parser.parse_args()

    circuits = args.circuits or (TRAIN_CIRCUITS if args.train_only else ALL_CIRCUITS)
    seeds = args.seeds or SEEDS

    phase_dir = os.path.join(args.save_dir, args.phase)

    if args.phase == 'trimmed':
        configs = generate_configs_trimmed()
        print(f"Trimmed sweep: {len(configs)} configs")

    elif args.phase == 'coarse':
        configs = generate_configs_coarse()
        print(f"Phase 1 (coarse): {len(configs)} configs")

    elif args.phase == 'fine':
        if args.coarse_results is None:
            coarse_path = os.path.join(args.save_dir, 'coarse', 'results.json')
        else:
            coarse_path = args.coarse_results

        if not os.path.exists(coarse_path):
            print(f"ERROR: coarse results not found at {coarse_path}")
            print("Run --phase coarse first, or specify --coarse_results")
            sys.exit(1)

        with open(coarse_path) as f:
            coarse_results = json.load(f)

        # Find top K configs
        from collections import defaultdict
        by_config = defaultdict(list)
        for r in coarse_results:
            by_config[r['config_key']].append(r)

        config_means = []
        for ck, runs in by_config.items():
            mean_ratio = np.mean([r['ratio_vs_ref'] for r in runs])
            config_means.append((mean_ratio, runs[0]['config']))
        config_means.sort()

        top_configs = [c for _, c in config_means[:args.top_k]]
        print(f"Top {args.top_k} configs from coarse phase:")
        for i, c in enumerate(top_configs):
            print(f"  {i+1}. ss={c['subset_size']} wf={c['window_fraction']} tl={c['cpsat_time_limit']}")

        configs = generate_configs_fine(top_configs)
        print(f"\nPhase 2 (fine): {len(configs)} configs")

    elif args.phase == 'adapt':
        base_configs = generate_configs_adapt()
        # Apply base geometric params
        ss = args.base_subset_size or 30
        wf = args.base_window_fraction or 0.15
        tl = args.base_cpsat_time_limit or 5.0
        configs = []
        for bc in base_configs:
            configs.append({
                'subset_size': ss,
                'window_fraction': wf,
                'cpsat_time_limit': tl,
                **bc,
            })
        print(f"Phase 3 (adapt): {len(configs)} configs (base: ss={ss}, wf={wf}, tl={tl})")

    elif args.phase == 'congestion':
        base_configs = generate_configs_congestion()
        ss = args.base_subset_size or 30
        wf = args.base_window_fraction or 0.15
        tl = args.base_cpsat_time_limit or 5.0
        configs = []
        for bc in base_configs:
            configs.append({
                'subset_size': ss,
                'window_fraction': wf,
                'cpsat_time_limit': tl,
                'plateau_threshold': 20,
                'adapt_threshold': 30,
                'adaptive': True,
                **bc,
            })
        print(f"Phase 4 (congestion): {len(configs)} configs")

    elif args.phase == 'single':
        configs = [{
            'subset_size': args.subset_size,
            'window_fraction': args.window_fraction,
            'cpsat_time_limit': args.cpsat_time_limit,
            'congestion_weight': args.congestion_weight,
            'plateau_threshold': args.plateau_threshold,
            'adapt_threshold': args.adapt_threshold,
            'adaptive': args.adaptive,
        }]
        print(f"Single config: {config_key(configs[0])}")

    run_sweep(
        configs=configs,
        circuits=circuits,
        seeds=seeds,
        wall_clock=args.wall_clock,
        benchmark_base=args.benchmark_base,
        legalize_time_limit=args.legalize_time_limit,
        legalize_window=args.legalize_window,
        save_dir=phase_dir,
    )


if __name__ == '__main__':
    main()
