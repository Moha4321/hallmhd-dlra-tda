# hallmhd-dlra-torch

> 3D Hall-MHD pseudo-spectral simulation with Dynamical Low-Rank Approximation,
> topological disruption detection, and holographic boundary observers.
> Runs entirely on a single Apple Mac Mini M4 (16 GB unified RAM).

---

## What this is

A 12-month research codebase implementing:

1. A verified 3D pseudo-spectral Hall-MHD solver in sheared flux-tube geometry
2. A Dynamical Low-Rank Approximation (DLRA) engine enabling effective 512³ resolution within 16 GB RAM (conditional on Month 5 SVD test)
3. A 3D persistent homology pipeline (Betti-0/1/2) for topological confinement diagnostics
4. A Physics-Informed CNN trained on boundary data as a holographic disruption alarm

Target: ITER-relevant plasma parameters. Platform: Apple Silicon M4, JAX + Metal.

---

## Repo layout

```
hallmhd-dlra-torch/
├── src/
│   ├── engine/          # 3D pseudo-spectral Hall-MHD solver (Phase 1)
│   ├── dlra/            # Projector-splitting DLRA engine (Phase 2A)
│   ├── tda/             # Cubical persistent homology pipeline (Phase 3)
│   ├── ml/              # PI-CNN holographic observer (Phase 5)
│   └── diagnostics/     # Chirikov, null-point tracking, virtual coils
├── tests/               # Benchmark tests B1–B5 (pre-registered, Month 1)
├── docs/
│   └── theory/          # LaTeX source for analytical derivations
├── scripts/             # Profiling, Hero Run launcher, data export
├── notebooks/           # Exploratory analysis and figure generation
├── data/                # Simulation snapshots (gitignored, see Zenodo)
└── outputs/             # Figures and results (gitignored)
```

---

## Phase status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Architecture & physics design | 🔄 In progress |
| 1 | 3D spectral engine + SVD test | ⏳ Pending |
| 2A | DLRA engine (conditional) | ⏳ Pending |
| 2B | Full-rank extended runs (alternative) | ⏳ Pending |
| 3 | 3D topological diagnostics | ⏳ Pending |
| 4 | Hero Run | ⏳ Pending |
| 5 | PI-CNN holographic observer | ⏳ Pending |
| 6 | Papers + open release | ⏳ Pending |

---

## Physics parameters (sealed Month 1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| B_T | 5.3 T | ITER on-axis guide field |
| epsilon | 0.1 | Poloidal-to-toroidal field ratio |
| d_i | 0.1 | Ion inertial length (normalised) |
| S_eff | 1e6 | Lundquist number at grid scale |
| gamma | 5/3 | Adiabatic index |

---

## Benchmark tests (pre-registered, must pass before Month 5 SVD test)

- **B1** — Furth–Killeen–Rosenbluth tearing mode growth rate
- **B2** — Alfvén wave dispersion in 3D flux-tube geometry
- **B3** — Kraichnan k^{-3} perpendicular energy spectrum
- **B4** — Goldreich–Sridhar k^{-2}_parallel parallel energy spectrum
- **B5** — Quadrupolar Bz structure during Hall reconnection

Acceptance thresholds are fixed in `tests/benchmarks.py`. No post-hoc adjustment.

---

## Hardware target

- Apple Mac Mini M4, 16 GB unified RAM
- JAX with Metal backend (GPU) + Apple Accelerate (CPU BLAS/LAPACK)
- NVMe SSD >= 2 TB, > 3 GB/s for checkpoint streaming

---

## Installation

```bash
git clone https://github.com/<org>/hallmhd-dlra-torch
cd hallmhd-dlra-torch
pip install -e ".[dev]"
```

Requires Python >= 3.11, JAX >= 0.4.25 with Metal plugin, gudhi >= 3.9.

---

## Pre-registration commitments

1. The analytical prediction for singular value decay sigma_k(Z) <= C * exp(-alpha * k)
   is sealed in `docs/theory/svd_prediction_sealed.tex` before Month 5 results are inspected.
2. Benchmark thresholds B1–B5 are fixed in `tests/benchmarks.py` before the solver is built.

---

## License

MIT. Simulation data deposited on Zenodo upon manuscript submission.

## Citation

Pre-print to be posted on arXiv (physics.plasm-ph) simultaneously with journal submission.