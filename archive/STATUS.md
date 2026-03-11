# SDDS-PyTorch: Chip Placement Port Status

## Overview

Port of energy-guided continuous diffusion chip placement from DIffUCO (JAX) to SDDS-PyTorch.
Uses SDDS (Score-based Diffusion with Denoising Score-matching) + PPO for unsupervised chip placement optimization.

## Implementation Status

### Completed (Phase A: Faithful Port)

| File | Description | Source |
|------|-------------|--------|
| `chip_placement_energy.py` | HPWL + overlap + boundary energy function | DIffUCO ChipPlacementEnergy |
| `chip_placement_data.py` | Synthetic chip instance generator (Chip_10/20/50) | DIffUCO ChipDatasetGenerator |
| `continuous_step_model.py` | Gaussian diffusion model (mean + log-var heads, GNN backbone) | DIffUCO ContinuousHead |
| `noise_schedule.py` | Added `GaussianNoiseSchedule` (alongside existing categorical) | DIffUCO GaussianNoise |
| `trajectory.py` | Added `collect_continuous_trajectory` (per-step energy) | DIffUCO PPO_Trainer scan_body |
| `ppo_trainer.py` | Added `ContinuousPPOTrainer` class | DIffUCO PPO_Trainer |
| `train_chip_placement.py` | Training script with synthetic + benchmark data support | New |
| `benchmark_loader.py` | BookShelf parser for ICCAD04 benchmarks | New |
| `visualize_benchmark.py` | Benchmark data visualization and compatibility checks | New |

### Benchmark Data

- **Location**: `benchmarks/iccad04/extracted/ibm01-ibm18/`
- **Format**: BookShelf (.nodes, .nets, .pl, .scl)
- **Source**: [ICCAD04 IBM Benchmarks](http://vlsicad.eecs.umich.edu/BK/ICCAD04bench/)
- **Macro-only filtering**: Height > standard cell row height (verified against chipdiffusion for all 17 circuits)
- **ibm05**: 0 macros (all standard cells), skipped

### Verified Working

- Macro identification matches chipdiffusion DEF/LEF `CLASS BLOCK` for all circuits
- BFS subsampling preserves graph connectivity
- Bounding-box normalization fills [-1,1] canvas properly
- Energy computation (HPWL, overlap, boundary) produces valid values
- Gradient flow through energy function (no NaN/Inf)
- Data format compatible with training pipeline (positions, sizes, edge_index, edge_attr)

### Visualization Outputs

In `viz_output/`:
- `macro_only_circuits.png` — ibm01-04, ibm06 macro placements side by side
- `ibm01_macro_detail.png` — ibm01 macro placement with/without edges + size distribution

## Usage

### Training with synthetic data
```bash
python train_chip_placement.py --dataset Chip_20_components --n_diffusion_steps 50 --N_anneal 500
```

### Training with benchmark data
```bash
python train_chip_placement.py --benchmark --circuit ibm01 --n_diffusion_steps 50 --N_anneal 500
```

### Visualizing benchmark data
```bash
python visualize_benchmark.py
```

## TODO / Known Issues

1. **`--macros_only` flag**: `train_chip_placement.py` load_data() doesn't pass `macros_only=True` yet
2. **Edge count mismatch**: BookShelf star-decomposition gives fewer edges than DEF/LEF multi-pin nets (1720 vs 2040 for ibm01)
3. **Phase B improvements** (deferred):
   - Fix HPWL normalization in overlap penalty (degenerate collapse when `overlap * HPWL -> 0`)
   - Explore discretized grid formulation
   - Density-based overlap penalties
   - Basis state parallelization
