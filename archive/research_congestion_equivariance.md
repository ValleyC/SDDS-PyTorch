# Research: Congestion Metrics & E(2) Equivariance for LNS+CP-SAT Placement

Date: 2026-03-05

---

## Part 1: Congestion-Aware Placement — Metrics, Literature, Formulation

### 1.1 Standard Congestion Metrics

#### RUDY (Rectangular Uniform wire DensitY)
- **Origin**: Spindler & Johannes, DATE 2007
- **The standard fast congestion proxy** used by DREAMPlace, CircuitNet, and modern placers

**Formula**:
```
For net k with bounding box width w_k, height h_k:
    RUDY_k at tile (i,j) = [(w_k + h_k) / (w_k × h_k)] × (overlap_area / tile_area)

Total RUDY(i,j) = Σ_nets RUDY_k(i,j)
```

The term `(w_k + h_k)/(w_k × h_k)` = `HPWL_k / bbox_area_k` uniformly distributes expected wire demand across the net's bounding box.

**Variants** (from CircuitNet documentation):
- **RUDY_long**: Only multi-tile nets
- **RUDY_short**: Only single-tile nets
- **PinRUDY**: Pin-based — for each pin, accumulate `(w_k + h_k)/(w_k × h_k)` at pin's tile
- **PinRUDY_long**: PinRUDY restricted to multi-tile nets

#### RSMT-based Congestion
- **FLUTE** (Fast Lookup Table-based Estimation) — O(1) for ≤8 pins
- More accurate than RUDY but much slower (must construct Steiner tree per net)
- Used in detailed routing, not typically in early placement

#### Pin Density
```
pin_density(i,j) = count of pins in tile (i,j)
reduced_capacity = capacity - N_pins × pin_blockage_factor
```
CRISP (ICCAD 2009) showed pin density is a critical overlooked congestion source.

#### Global Routing Congestion (GRC) — Gold Standard
```
Overflow(e) = max(0, Usage(e) - Capacity(e))
TOF = Σ_edges Overflow(e)          # Total Overflow (primary ISPD metric)
MOF = max_edges Overflow(e)         # Maximum Overflow (tiebreaker)
```
Requires running a global router (FastRoute, NCTU-GR).

### 1.2 Key Papers: "HPWL Is Not Enough"

| Paper | Venue | Key Finding |
|-------|-------|-------------|
| **ChiPBench** | NeurIPS 2025 | MacroHPWL has **weak correlation** with final wirelength, WNS, TNS, power |
| **RoutePlacer** | KDD 2024 | GNN-predicted routability as placement objective → 16% TOF reduction |
| **RUPlace** | DAC 2025 | Unified placement+routing via ADMM + Wasserstein → theoretically-grounded congestion |
| **GOALPlace** | 2024 | Post-route optimized density as learning target → better PPA than HPWL |
| **LaMPlace** | ICLR 2025 Oral | Cross-stage WNS/TNS as direct objective → 43% WNS improvement, best congestion |
| **CRISP** | ICCAD 2009 | First to show pin density is critical overlooked congestion source |
| **DREAMPlace-Cong** | DATE 2021 | FCN on RUDY+PinRUDY+MacroRegion to predict routing congestion → 9% reduction |

### 1.3 ChiPBench Evaluation Metrics (Full List)

| Metric | Stage | Description |
|--------|-------|-------------|
| MacroHPWL | After macro placement | HPWL of macro-only nets |
| HPWL | After global placement | HPWL of all nets |
| Wirelength | After routing | Actual routed wirelength |
| Congestion (%) | After CTS | Wire density |
| WNS (ns) | After routing | Worst negative slack (timing) |
| TNS (ns) | After routing | Total negative slack |
| NVP | After routing | Number of violating paths |
| Power (nW) | After routing | Total power |
| Area (um²) | After routing | Final chip area |

Tool: OpenROAD (Yosys + RePlAce + TritonCTS + TritonRoute)

### 1.4 Contest Evaluation Standards

**ISPD 2011 Routability-Driven Placement Contest**:
- Primary: `TOT_OF × (1 + PLACER_RUNTIME_FACTOR)`
- Tiebreaker: Maximum Overflow (MOF)

**ISPD 2008 Global Routing Contest**: Ranked by (1) TOF, (2) MOF, (3) wirelength

### 1.5 CP-SAT Congestion Formulation Options

#### Option A: RUDY as LNS acceptance criterion (simplest, recommended first)
```python
cost = HPWL + λ_cong × peak_RUDY
# Compute RUDY after CP-SAT solve, use in accept/reject
```
No CP-SAT reformulation needed. Works with existing pipeline.

#### Option B: Grid-based capacity constraints in CP-SAT
```python
# 1D cumulative decomposition (avoids 2D nonlinearity):
for col in range(G):
    intervals = [macro_y_interval[m] for m in subset_macros overlapping col]
    demands = [overlap_width_with_column(m, col) for m in ...]
    model.AddCumulative(intervals, demands, capacity_per_column)
```
Uses CP-SAT's built-in `AddCumulative` constraint for routing channel capacity.

#### Option C: Binary tile presence (approximate)
```python
for gx, gy in grid:
    present_vars = [model.NewBoolVar() for m in subset]
    # present=1 if macro overlaps tile
    model.Add(sum(present[m] * area[m]) <= tile_capacity + excess)
    # Minimize sum of excess in objective
```

**Recommendation**: Start with Option A (post-solve RUDY), graduate to Option B if needed.

### 1.6 What To Report in Paper

- **Minimum**: HPWL + RUDY + pin density
- **Better**: Run FastRoute for true TOF/MOF
- **Best**: Full OpenROAD pipeline (ChiPBench-style) for WNS/TNS/power

### 1.7 Evaluation Tools

| Tool | Type | Notes |
|------|------|-------|
| FastRoute (Iowa State) | Academic global router | Used as eval router in DREAMPlace |
| NCTU-GR (NCTU) | Academic global router | 1.05x faster than ISPD08 winners |
| OpenROAD | Full open-source EDA | TritonRoute for detailed routing; used by ChiPBench |
| Cadence Innovus | Commercial | Gold standard for industry PPA evaluation |
| FLUTE | RSMT estimator | O(1) for ≤8 pins; industry standard wirelength estimation |

### 1.8 ICCAD04 Benchmark Note

ICCAD04 benchmarks have **no bundled congestion evaluation**. To evaluate:
1. Convert BookShelf placement → DEF format
2. Run FastRoute or NCTU-GR
3. Measure TOF/MOF

Alternative: Compute RUDY directly from placement (no router needed).

---

## Part 2: E(2) Equivariance for Chip Placement

### 2.1 The Gap in Literature

**Only ONE paper uses equivariance for placement**: MacroRank (ASP-DAC 2023, PKU) — translation equivariance for placement **ranking** (quality prediction), not generation.

**Nobody has used E(2)-equivariant GNNs for placement optimization/generation.**

### 2.2 Chip Placement Symmetries

| Symmetry | Type | Preserves HPWL? |
|----------|------|-----------------|
| Translation (shift all macros) | Continuous | Yes |
| Horizontal reflection | Discrete | Yes |
| Vertical reflection | Discrete | Yes |
| 90° rotation | Discrete | Yes (if canvas is square, or with dim swap) |

These form the **dihedral group D4** (square canvas) or **Klein four-group** (rectangular canvas).

### 2.3 EDISCO — Our Prior Work

**EDISCO**: Equivariant Continuous-Time Diffusion Solver for Graph Combinatorial Optimization
- Submitted to ICLR (desk rejected due to hallucinated reference)
- Uses **EGNN** (E(n) Equivariant GNN) + continuous-time Markov chains (CTMCs)
- Results: TSP-1000 error 2.73% → 0.54%, TSP-10000 error 2.68% → 1.20%
- **33-50% less training data** than non-equivariant methods (DIFUSCO, DISCO)
- Extended to CVRP and Euclidean Steiner Tree

**Existing codebase locations**:
- `d:\Codes\EDISCO Journal\EDISCO/` — Journal version
- `d:\Codes\ICML 2026 for EDISCO\EDISCO/` — ICML submission version
- `d:\Codes\ICLR-2026-Rebuttal\EDISCO/` — Rebuttal version

**Key reusable files**:
- `edisco/models/egnn_encoder.py` — E(2)-equivariant GNN encoder
- `edisco/utils/equivariance_utils.py` — E(2) transform testing (rotation, translation, reflection)
- `edisco/models/continuous_score_network.py` — Score network architecture

### 2.4 Related Work on Equivariance in CO

| Paper | Venue | What |
|-------|-------|------|
| **MacroRank** | ASP-DAC 2023 | Translation equivariance for placement quality ranking |
| **GIE-GNNs** | LION 2025 | Geometrically invariant/equivariant GNNs for TSP algorithm selection |
| **Dubins-Aware NCO** | Drones 2026 | Rotary Phase Encoding (RoPhE) for SO(2) equivariant routing |
| **Ouyang et al.** | SN CS 2024 | Equivariance + local search for TSP generalization |
| **Symmetry Breaking & Equivariance** | arXiv:2312.09016 | Theory of relaxed equivariance for CO |
| **ACE** | arXiv:2505.13631 | Adaptive Constrained Equivariance (learned symmetry level) |
| **PEnGUiN** | RLC 2025 | Partially equivariant GNNs unifying EGNN and standard GNN |

### 2.5 E(2)-Equivariant Architectures for Placement

**EGNN (recommended)** — Satorras et al., arXiv:2102.09844
- Updates both node features AND coordinates per layer
- Coordinate updates: weighted sum of relative position differences
- Lightweight (no higher-order tensors)
- Natural fit: operates on 2D coordinates in graph structure
- PyTorch: `lucidrains/egnn-pytorch`, `vgsatorras/egnn`

**Alternatives**: Vector Neurons (SO(3), overkill for 2D), SE(2) Steerable CNNs (grid-based, not graph), RoPhE (SO(2) only).

### 2.6 How Equivariance + Symmetry Breaking Work Together

They are **complementary at different levels**:

| Level | Mechanism | Effect |
|-------|-----------|--------|
| **ML (GNN)** | E(2) equivariance | Fewer training traces needed; predictions transform correctly under symmetry; better cross-circuit generalization |
| **OR (CP-SAT)** | Symmetry breaking constraints | Prunes equivalent branches in search tree; speeds up sub-problem solving |

**In our pipeline**:

**GNN side** — Replace TrustRegionGNN backbone with EGNN from EDISCO:
- Coordinates `(x, y)` as geometric input (equivariant updates)
- Displacement/subset predictions: **equivariant** (rotate input → rotate output)
- Window size / value predictions: **invariant** (don't change under symmetry)
- 4-8x effective training data from D4 symmetry group for free

**CP-SAT side** — Add symmetry-breaking constraints in `solve_subset()`:
```python
# If macros i, j have identical size and similar connectivity:
if size[i] == size[j] and connectivity_similar(i, j):
    model.Add(x[i] <= x[j])  # break permutation symmetry
```

### 2.7 Why This Is a Strong Paper Angle

1. **No prior work** on E(2)-equivariant GNNs for placement generation
2. Builds on our **own EDISCO expertise** (proven on TSP/CVRP)
3. **Dual symmetry exploitation** (ML equivariance + OR symmetry breaking) is novel
4. **Quantifiable benefit**: equivariance reduces required training traces by 33-50% (proven in EDISCO)
5. Addresses a real need: ML-guided LNS requires training data across circuits — equivariance makes this feasible

---

## Part 3: Proposed Paper Contributions

### Contribution Summary

1. **First CP-SAT formulation for macro placement** with NoOverlap2D (no prior work)
2. **E(2)-equivariant GNN for LNS search control** — adapting EDISCO's EGNN to predict subsets + windows
3. **Dual symmetry exploitation**: equivariance (ML) + symmetry breaking (OR) — novel combination
4. **Congestion-aware objective** (HPWL + RUDY) — addresses known HPWL-as-proxy weakness
5. **Conflict-driven neighborhood selection** — using CP-SAT's conflict analysis to guide LNS

### Ablation Matrix

| Condition | Equivariant | Sym Breaking | Congestion | ML Subset |
|-----------|-------------|--------------|------------|-----------|
| Pure LNS | - | - | HPWL only | Heuristic |
| + Sym Breaking | - | Yes | HPWL only | Heuristic |
| + Congestion | - | - | HPWL+RUDY | Heuristic |
| + EGNN (non-equiv) | No | - | HPWL only | Learned |
| + EGNN (equiv) | Yes | - | HPWL only | Learned |
| Full system | Yes | Yes | HPWL+RUDY | Learned |

---

## References

### Congestion
- Spindler & Johannes, "Fast and Accurate Routing Demand Estimation" (DATE 2007) — RUDY
- ChiPBench, "Benchmarking End-To-End Performance of AI-Based Chip Placement" (NeurIPS 2025)
- RoutePlacer, "An End-to-End Routability-Aware Placer with GNN" (KDD 2024)
- RUPlace, "Optimizing Routability via Unified Placement and Routing" (DAC 2025)
- GOALPlace, "Begin with the End in Mind" (2024)
- LaMPlace, "Learning to Optimize Cross-Stage Metrics in Macro Placement" (ICLR 2025)
- CRISP, "Congestion Reduction by Iterated Spreading during Placement" (ICCAD 2009)
- DREAMPlace Congestion, "Global Placement with DL-Enabled Routability Optimization" (DATE 2021)
- CircuitNet documentation (RUDY formulas)
- ISPD 2011 Routability-Driven Placement Contest
- FastRoute (Iowa State)
- NCTU-GR (IEEE TVLSI)
- OpenROAD project

### Equivariance
- EGNN, Satorras et al. (arXiv:2102.09844) — E(n) Equivariant GNNs
- MacroRank (ASP-DAC 2023) — Translation equivariance for placement ranking
- EDISCO — Our work: E(2) equivariant diffusion for CO
- DIFUSCO (NeurIPS 2023) — Graph-based diffusion for CO
- GIE-GNNs (LION 2025) — Geometrically invariant/equivariant GNNs for TSP
- Dubins-Aware NCO (Drones 2026) — RoPhE for SE(2) equivariance
- Symmetry Breaking & Equivariant NNs (arXiv:2312.09016)
- ACE: Adaptive Constrained Equivariance (arXiv:2505.13631)
- PEnGUiN: Partially Equivariant GNNs (RLC 2025)

### Ising/Potts (From Previous Research)
- Peterson & Söderberg, "Potts MFT for optimization" (IJNS 1989)
- Kanamaru et al., "Ising slot placement" (IEEE ICCE 2019)
- Grusdt et al., "FPGA Placement via Quantum Annealing" (ACM FPGA 2024)
- Cook et al., "GPU Ising for VLSI max-cut" (2018)
- Schuetz et al., "CO with Physics-Inspired GNNs" (Nature MI 2022)
- "Restoring Sparsity in Potts Machines via MF Constraints" (arXiv 2026)

### Placement Baselines
- DREAMPlace, Gu et al. (ICCAD 2020)
- ePlace, Lu et al. (TODAES 2014)
- IncreMacro (ISPD 2024) — Iterative macro refinement via LP
- AlphaChip (Nature 2021)
- Chip Placement with Diffusion (arXiv:2407.12282)
