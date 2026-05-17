r"""
benchmarks.py
=============

Pre-registered benchmark tests **B1–B5** for the 3D Hall-MHD solver.

All acceptance thresholds are fixed here at the end of Phase~0 (Month~1)
and are sealed before any solver code is written.  A solver that does not
pass all five tests is not permitted to proceed to the Month~4–5 SVD
spectrum measurement.

.. warning::
    These thresholds must not be altered after the solver is implemented.
    Any modification constitutes a pre-registration violation and must be
    disclosed in the manuscript.

Benchmark definitions
---------------------

**B1 — Furth–Killeen–Rosenbluth (FKR) tearing mode growth rate**

For a Harris current sheet with half-width :math:`a` and resistivity
:math:`\eta`, the linear tearing growth rate is [FKR1963]_:

.. math::

    \gamma_\text{FKR} = 0.55\,\eta^{3/5}\,k^{2/5}\,\Delta'^{4/5}

where :math:`\Delta'` is the tearing stability index and :math:`k` the
wavenumber along the sheet.  The solver must reproduce :math:`\gamma_\text{FKR}`
to within :math:`\pm 5\%` in the resistive MHD limit (:math:`d_i\to 0`).

**B2 — Alfvén wave dispersion in 3D flux-tube geometry**

A shear Alfvén wave propagating along :math:`\hat{z}` obeys
:math:`\omega = k_\parallel v_A`.  The solver must reproduce the
dispersion relation to within :math:`\pm 1\%` for
:math:`k_\parallel\in[1,8]` (in units of :math:`2\pi/L_z`).

**B3 — Kraichnan :math:`k^{-3}` perpendicular energy spectrum**

In 2D MHD turbulence (and in the :math:`k_\parallel=0` slice of
anisotropic 3D), the magnetic energy spectrum obeys
:math:`E(k_\perp)\sim k_\perp^{-3}` in the inertial range [Kraichnan1965]_.
The measured perpendicular spectral index must lie in
:math:`[-3.15,\,-2.85]`.

**B4 — Goldreich–Sridhar :math:`k_\parallel^{-2}` parallel spectrum**

In strong Alfvénic turbulence [GS1995]_, the parallel energy spectrum
obeys :math:`E(k_\parallel)\sim k_\parallel^{-2}`.  The measured parallel
spectral index must lie in :math:`[-2.20,\,-1.80]`.

**B5 — Quadrupolar :math:`B_z` structure during Hall reconnection**

The Hall term generates a characteristic quadrupolar out-of-plane magnetic
field :math:`B_z` during fast reconnection [Birn2001]_.  The solver must
produce a :math:`B_z` pattern with four-fold symmetry, with peak amplitude
:math:`\max|B_z| > 0.05\,B_0` and correlation :math:`r > 0.90` with the
analytical quadrupolar template.

References
----------
.. [FKR1963] Furth, Killeen & Rosenbluth, *Phys. Fluids* **6**, 459 (1963).
.. [Kraichnan1965] Kraichnan, *Phys. Fluids* **10**, 1417 (1967).
.. [GS1995] Goldreich & Sridhar, *Astrophys. J.* **438**, 763 (1995).
.. [Birn2001] Birn et al., *J. Geophys. Res.* **106**, 3715 (2001).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Threshold dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BenchmarkSpec:
    r"""
    Specification for a single pre-registered benchmark test.

    Parameters
    ----------
    name:
        Short identifier (e.g. ``"B1"``).
    description:
        One-line human description.
    check:
        Callable ``(solver_output: dict) -> bool`` that returns ``True``
        iff the solver passes this benchmark.
    threshold_description:
        LaTeX-renderable string describing the acceptance criterion,
        for inclusion in the manuscript.
    """
    name: str
    description: str
    check: Callable[[dict], bool]
    threshold_description: str


# ---------------------------------------------------------------------------
# B1 — FKR tearing mode growth rate
# ---------------------------------------------------------------------------

def _b1_check(result: dict) -> bool:
    r"""
    Pass criterion: measured :math:`\gamma` is within 5% of
    :math:`\gamma_\text{FKR}`.

    Expected key in *result*: ``"gamma_measured"`` (float).
    """
    gamma_meas: float = result["gamma_measured"]
    gamma_fkr:  float = result["gamma_fkr_analytic"]
    rel_error = abs(gamma_meas - gamma_fkr) / abs(gamma_fkr)
    return bool(rel_error <= 0.05)


B1 = BenchmarkSpec(
    name="B1",
    description="FKR tearing mode growth rate (resistive MHD limit, d_i -> 0)",
    check=_b1_check,
    threshold_description=(
        r"$|\gamma_\text{measured} - \gamma_\text{FKR}| / \gamma_\text{FKR} \leq 0.05$"
    ),
)


# ---------------------------------------------------------------------------
# B2 — Alfvén wave dispersion
# ---------------------------------------------------------------------------

def _b2_check(result: dict) -> bool:
    r"""
    Pass criterion: max relative error in :math:`\omega(k_\parallel)` over
    :math:`k_\parallel\in\{1,\ldots,8\}` is below 1%.

    Expected key in *result*: ``"omega_measured"`` and ``"omega_analytic"``
    (both 1-D arrays of length 8).
    """
    omega_m = np.asarray(result["omega_measured"])
    omega_a = np.asarray(result["omega_analytic"])
    rel_errors = np.abs(omega_m - omega_a) / np.abs(omega_a)
    return bool(np.max(rel_errors) <= 0.01)


B2 = BenchmarkSpec(
    name="B2",
    description="Alfvén wave dispersion omega = k_parallel * v_A in 3D flux-tube geometry",
    check=_b2_check,
    threshold_description=(
        r"$\max_{k_\parallel} |\omega_\text{meas} - k_\parallel v_A| "
        r"/ (k_\parallel v_A) \leq 0.01$"
    ),
)


# ---------------------------------------------------------------------------
# B3 — Kraichnan k^{-3} perpendicular energy spectrum
# ---------------------------------------------------------------------------

def _b3_check(result: dict) -> bool:
    r"""
    Pass criterion: fitted perpendicular spectral index lies in
    :math:`[-3.15,\,-2.85]`.

    Expected key in *result*: ``"perp_spectral_index"`` (float, should be ~ -3).
    """
    idx: float = result["perp_spectral_index"]
    return bool(-3.15 <= idx <= -2.85)


B3 = BenchmarkSpec(
    name="B3",
    description="Kraichnan k^{-3} perpendicular magnetic energy spectrum",
    check=_b3_check,
    threshold_description=(
        r"Fitted index $\alpha_\perp \in [-3.15,\,-2.85]$ "
        r"(ideal: $\alpha_\perp = -3$)"
    ),
)


# ---------------------------------------------------------------------------
# B4 — Goldreich–Sridhar k_parallel^{-2} parallel spectrum
# ---------------------------------------------------------------------------

def _b4_check(result: dict) -> bool:
    r"""
    Pass criterion: fitted parallel spectral index lies in
    :math:`[-2.20,\,-1.80]`.

    Expected key in *result*: ``"par_spectral_index"`` (float, should be ~ -2).
    """
    idx: float = result["par_spectral_index"]
    return bool(-2.20 <= idx <= -1.80)


B4 = BenchmarkSpec(
    name="B4",
    description="Goldreich–Sridhar k_parallel^{-2} parallel magnetic energy spectrum",
    check=_b4_check,
    threshold_description=(
        r"Fitted index $\alpha_\parallel \in [-2.20,\,-1.80]$ "
        r"(ideal: $\alpha_\parallel = -2$)"
    ),
)


# ---------------------------------------------------------------------------
# B5 — Quadrupolar B_z structure during Hall reconnection
# ---------------------------------------------------------------------------

def _b5_check(result: dict) -> bool:
    r"""
    Pass criterion: out-of-plane :math:`B_z` field shows four-fold symmetry
    with :math:`\max|B_z| > 0.05\,B_0` and Pearson correlation :math:`r > 0.90`
    with the analytical quadrupolar template.

    Expected keys in *result*:
        - ``"bz_max_amplitude"`` (float, normalised to B_0 = 1)
        - ``"bz_quadrupole_correlation"`` (float in [-1, 1])
    """
    amp:  float = result["bz_max_amplitude"]
    corr: float = result["bz_quadrupole_correlation"]
    return bool(amp > 0.05 and corr > 0.90)


B5 = BenchmarkSpec(
    name="B5",
    description=(
        "Quadrupolar B_z structure during Hall reconnection "
        "(Hall term signature)"
    ),
    check=_b5_check,
    threshold_description=(
        r"$\max|B_z| > 0.05\,B_0$ and Pearson $r(B_z, \text{template}) > 0.90$"
    ),
)


# ---------------------------------------------------------------------------
# Benchmark suite (ordered; all must pass before Month-5 SVD test)
# ---------------------------------------------------------------------------

BENCHMARK_SUITE: tuple[BenchmarkSpec, ...] = (B1, B2, B3, B4, B5)


def run_suite(results: dict[str, dict]) -> dict[str, bool]:
    r"""
    Evaluate the full benchmark suite.

    Parameters
    ----------
    results:
        Mapping from benchmark name (``"B1"`` … ``"B5"``) to a dict of
        solver outputs as expected by each ``check`` function.

    Returns
    -------
    dict[str, bool]
        Pass/fail status for each benchmark.

    Raises
    ------
    RuntimeError
        If any benchmark fails — the solver must not proceed to Phase 1.
    """
    status: dict[str, bool] = {}
    failures: list[str] = []

    for spec in BENCHMARK_SUITE:
        if spec.name not in results:
            raise KeyError(
                f"Benchmark {spec.name} result not provided. "
                "All five benchmarks must be evaluated."
            )
        passed = spec.check(results[spec.name])
        status[spec.name] = passed
        if not passed:
            failures.append(spec.name)

    if failures:
        raise RuntimeError(
            f"Benchmark(s) FAILED: {failures}. "
            "Solver must not proceed to Phase 1 (SVD test)."
        )

    return status


def print_suite_report(status: dict[str, bool]) -> None:
    r"""Print a human-readable pass/fail summary."""
    print("\nBenchmark Suite Report (Phase 0 — pre-registered thresholds)")
    print("=" * 60)
    for spec in BENCHMARK_SUITE:
        flag = "PASS ✓" if status.get(spec.name) else "FAIL ✗"
        print(f"  {spec.name}  [{flag}]  {spec.description}")
    all_pass = all(status.values())
    print("-" * 60)
    print(f"  Overall: {'ALL PASSED — proceed to Phase 1' if all_pass else 'FAILURES DETECTED — do not proceed'}")
    print()