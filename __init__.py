"""
SDDS-PyTorch: Chip Placement with LNS + CP-SAT.

Core modules:
  - cpsat_solver: OR-Tools CP-SAT wrapper (legalize, solve_subset, HPWL)
  - lns_solver: Large Neighborhood Search orchestrator
  - benchmark_loader: ICCAD04 BookShelf parser
  - net_spatial_gnn: Dual-stream GNN architecture
  - greedy_placer: Sequential heatmap-based placement decoder
"""
