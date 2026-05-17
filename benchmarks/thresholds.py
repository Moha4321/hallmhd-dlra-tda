r"""
thresholds.py
=============

Pre-registered numerical acceptance thresholds for benchmark tests B1–B5.

These values are **sealed at the end of Phase~0 (Month~1)** and must not
be altered after the solver is implemented.  They are the single source of
truth consumed by :mod:`benchmarks.benchmark_definitions` check functions
and the LaTeX document ``docs/benchmark_suite.tex``.

Sealing protocol
----------------
After committing this file, generate a SHA-256 hash and record it in
``docs/phase0_architecture.tex``:

.. code-block:: bash

    shasum -a 256 benchmarks/thresholds.py

Any modification after that commit constitutes a pre-registration
violation and must be disclosed in the manuscript.

Physical basis for each threshold
----------------------------------

**B1 — FKR tearing mode growth rate**

The Furth–Killeen–Rosenbluth linear growth rate for a Harris sheet is
[FKR1963]_:

.. math::

    \gamma_\text{FKR} = 0.55\,\eta^{3/5} k^{2/5} \Delta'^{4/5}

We require the solver to match this to within 5 % in the resistive limit
:math:`d_i \to 0`.  The 5 % tolerance accounts for finite-grid corrections
at the target resolution 128×128×64.

**B2 — Alfvén wave dispersion**

Shear Alfvén waves obey :math:`\omega = k_\parallel v_A`.  The 1 %
tolerance is tighter than B1 because the Alfvén wave is a linear,
non-reconnecting mode; any larger error indicates a spectral operator bug.

**B3 — Kraichnan** :math:`k^{-3}` **perpendicular spectrum**

Kraichnan's 2D MHD prediction gives :math:`E(k_\perp)\sim k_\perp^{-3}`.
The ±0.15 window around −3 allows for finite-inertial-range effects at the
target resolution.

**B4 — Goldreich–Sridhar** :math:`k_\parallel^{-2}` **parallel spectrum**

In strong Alfvénic turbulence [GS1995]_ :math:`E(k_\parallel)\sim
k_\parallel^{-2}`.  The ±0.20 window is wider than B3 because the parallel
inertial range is shorter at :math:`N_z = 64`.

**B5 — Quadrupolar** :math:`B_z` **Hall signature**

The Hall term generates a four-fold out-of-plane :math:`B_z` pattern
[Birn2001]_.  Amplitude threshold 0.05 :math:`B_0` ensures the Hall
physics is physically resolved (not swamped by numerical noise).
Correlation threshold 0.90 ensures the correct symmetry class.

References
----------
.. [FKR1963]  Furth, Killeen & Rosenbluth, *Phys. Fluids* **6**, 459 (1963).
.. [GS1995]   Goldreich & Sridhar, *Astrophys. J.* **438**, 763 (1995).
.. [Birn2001] Birn et al., *J. Geophys. Res.* **106**, 3715 (2001).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------------
# B1 — FKR tearing mode growth rate
# ---------------------------------------------------------------------------

#: Maximum relative error between measured and analytic FKR growth rate.
#: :math:`|\gamma_\text{meas} - \gamma_\text{FKR}| / \gamma_\text{FKR} \leq 0.05`
B1_MAX_RELATIVE_ERROR: Final[float] = 0.05


# ---------------------------------------------------------------------------
# B2 — Alfvén wave dispersion
# ---------------------------------------------------------------------------

#: Maximum relative error in :math:`\omega(k_\parallel)` across all
#: wavenumbers :math:`k_\parallel \in \{1, \ldots, 8\}`.
#: :math:`\max_{k_\parallel}|\omega_\text{meas} - k_\parallel v_A|
#: / (k_\parallel v_A) \leq 0.01`
B2_MAX_RELATIVE_ERROR: Final[float] = 0.01

#: Wavenumbers (in units of :math:`2\pi/L_z`) at which dispersion is tested.
B2_K_PARALLEL_RANGE: Final[tuple[int, ...]] = tuple(range(1, 9))  # 1..8


# ---------------------------------------------------------------------------
# B3 — Kraichnan k^{-3} perpendicular energy spectrum
# ---------------------------------------------------------------------------

#: Acceptable range for the fitted perpendicular spectral index.
#: Ideal value: −3.  Window: ±0.15.
B3_INDEX_MIN: Final[float] = -3.15
B3_INDEX_MAX: Final[float] = -2.85

#: Wavenumber range (in units of :math:`2\pi/L_\perp`) for the power-law fit.
#: Excludes the energy-injection range (low k) and dissipation range (high k).
B3_FIT_K_MIN: Final[int] = 4
B3_FIT_K_MAX: Final[int] = 32   # = (2/3) * N_perp/2 dealiasing cutoff


# ---------------------------------------------------------------------------
# B4 — Goldreich–Sridhar k_parallel^{-2} parallel spectrum
# ---------------------------------------------------------------------------

#: Acceptable range for the fitted parallel spectral index.
#: Ideal value: −2.  Window: ±0.20.
B4_INDEX_MIN: Final[float] = -2.20
B4_INDEX_MAX: Final[float] = -1.80

#: Wavenumber range for the parallel power-law fit.
B4_FIT_K_MIN: Final[int] = 2
B4_FIT_K_MAX: Final[int] = 16   # = (2/3) * N_z/2 dealiasing cutoff


# ---------------------------------------------------------------------------
# B5 — Quadrupolar B_z Hall signature
# ---------------------------------------------------------------------------

#: Minimum peak amplitude of the out-of-plane field :math:`\max|B_z|/B_0`.
B5_MIN_AMPLITUDE: Final[float] = 0.05

#: Minimum Pearson correlation of the measured :math:`B_z` pattern with the
#: analytical quadrupolar template.
B5_MIN_CORRELATION: Final[float] = 0.90


# ---------------------------------------------------------------------------
# Convenience dataclass (for passing to test functions as a single object)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkThresholds:
    r"""
    All pre-registered thresholds in a single frozen dataclass.

    Instantiate with :data:`THRESHOLDS` (the canonical singleton).
    """
    # B1
    b1_max_rel_error: float = B1_MAX_RELATIVE_ERROR

    # B2
    b2_max_rel_error: float = B2_MAX_RELATIVE_ERROR
    b2_k_parallel_range: tuple[int, ...] = B2_K_PARALLEL_RANGE

    # B3
    b3_index_min: float = B3_INDEX_MIN
    b3_index_max: float = B3_INDEX_MAX
    b3_fit_k_min: int   = B3_FIT_K_MIN
    b3_fit_k_max: int   = B3_FIT_K_MAX

    # B4
    b4_index_min: float = B4_INDEX_MIN
    b4_index_max: float = B4_INDEX_MAX
    b4_fit_k_min: int   = B4_FIT_K_MIN
    b4_fit_k_max: int   = B4_FIT_K_MAX

    # B5
    b5_min_amplitude:   float = B5_MIN_AMPLITUDE
    b5_min_correlation: float = B5_MIN_CORRELATION


#: Canonical singleton — import this everywhere.
THRESHOLDS: Final[BenchmarkThresholds] = BenchmarkThresholds()


if __name__ == "__main__":
    t = THRESHOLDS
    print("Pre-registered benchmark thresholds (Phase 0 — sealed)\n")
    print(f"  B1  FKR growth rate        rel. error ≤ {t.b1_max_rel_error:.0%}")
    print(f"  B2  Alfvén dispersion      rel. error ≤ {t.b2_max_rel_error:.0%}  "
          f"  k∥ ∈ {list(t.b2_k_parallel_range)}")
    print(f"  B3  Kraichnan k⁻³          index ∈ [{t.b3_index_min}, {t.b3_index_max}]  "
          f"  k⊥ fit range [{t.b3_fit_k_min}, {t.b3_fit_k_max}]")
    print(f"  B4  Goldreich–Sridhar k⁻²  index ∈ [{t.b4_index_min}, {t.b4_index_max}]  "
          f"  k∥ fit range [{t.b4_fit_k_min}, {t.b4_fit_k_max}]")
    print(f"  B5  Quadrupolar Bz         amp > {t.b5_min_amplitude}  corr > {t.b5_min_correlation}")