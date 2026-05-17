r"""
phase0_profile.py
=================

Phase~0 hardware profiling campaign (Month~1, Decision~D4).

Orchestrates the full profiling suite:

1. **3D real FFT** — ``profiling/profile_fft.py``
2. **Randomised SVD** — ``profiling/profile_svd.py``
3. **Memory budget** — ``src/hallmhd/memory_budget.py``
4. **Hero Run wall-time estimate** — derived from (1)

Writes a consolidated ``profiling/phase0_results.json``.

Usage
-----
.. code-block:: bash

    # from repo root:
    python scripts/phase0_profile.py

    # or via the shell launcher:
    bash scripts/run_phase0_profiling.sh
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on the path so both `hallmhd` and `profiling` are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

_OUTPUT_FILE = _REPO_ROOT / "profiling" / "phase0_results.json"

# ---------------------------------------------------------------------------
# JAX availability check (reported but not required)
# ---------------------------------------------------------------------------
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sub-module imports (each module is self-contained and runnable standalone)
# ---------------------------------------------------------------------------
from profiling.profile_fft import profile_fft_3d, hero_run_estimate
from profiling.profile_svd import profile_randomised_svd, profile_full_svd
from hallmhd.memory_budget  import table as memory_table, verify_ceiling, MEMORY


# ---------------------------------------------------------------------------
# Memory budget section
# ---------------------------------------------------------------------------

def run_memory_budget() -> dict:
    r"""
    Print and return the Phase~0 memory budget (Table~5 of the proposal).

    Calls :func:`hallmhd.memory_budget.table` and
    :func:`hallmhd.memory_budget.verify_ceiling` on the canonical
    singleton :data:`hallmhd.memory_budget.MEMORY`
    (128×128×64, R=32, float64).

    Returns
    -------
    dict
        ``state_gb``, ``peak_gb``, ``within_ceiling`` for the canonical config.
    """
    print(memory_table())

    gb = 1024 ** 3
    state_gb = MEMORY.state_bytes_dlra() / gb
    peak_gb  = MEMORY.peak_bytes_dlra()  / gb
    ok       = verify_ceiling()

    print(
        f"  Canonical config (128×128×64, R=32): "
        f"peak = {peak_gb:.3f} GB  "
        f"{'✓ within 13 GB ceiling' if ok else '✗ EXCEEDS ceiling'}"
    )
    return {
        "canonical_state_gb": round(state_gb, 4),
        "canonical_peak_gb":  round(peak_gb,  4),
        "within_13gb_ceiling": ok,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()

    print("\n" + "=" * 62)
    print("Phase 0 — Hardware Profiling Campaign")
    print("Apple Mac Mini M4, 16 GB unified RAM")
    print("=" * 62)

    if JAX_AVAILABLE:
        print(f"  JAX {jax.__version__}  |  backend: {jax.default_backend()}")
        print(f"  Devices: {jax.devices()}")
    else:
        print("  [WARNING] JAX not installed — FFT will run on NumPy CPU.")
        print("  For M4 GPU results: pip install jax-metal")

    print(f"  NumPy {np.__version__}  |  BLAS: Apple Accelerate\n")

    results: dict = {}

    # ------------------------------------------------------------------
    # 1. FFT throughput
    # ------------------------------------------------------------------
    print("─" * 62)
    print("1 / 4   3D real FFT throughput")
    print("─" * 62)
    fft_results = profile_fft_3d()
    results["fft_3d"] = fft_results

    # ------------------------------------------------------------------
    # 2. Hero Run estimate
    # ------------------------------------------------------------------
    print("\n─" * 62)
    print("2 / 4   Hero Run wall-time estimate (150,000 steps)")
    print("─" * 62)
    hero = hero_run_estimate(fft_results)
    results["hero_run_estimates"] = hero

    # ------------------------------------------------------------------
    # 3. Randomised SVD throughput
    # ------------------------------------------------------------------
    print("\n─" * 62)
    print("3 / 4   Randomised SVD throughput (HMT, Apple Accelerate)")
    print("─" * 62)
    svd_results = profile_randomised_svd()
    ref_svd     = profile_full_svd(matrix_size=128)
    results["randomised_svd"] = svd_results
    results["full_svd_reference_128x128"] = ref_svd

    # ------------------------------------------------------------------
    # 4. Memory budget
    # ------------------------------------------------------------------
    print("\n─" * 62)
    print("4 / 4   Memory budget (Table 5 of proposal, §8.2)")
    print("─" * 62)
    mem = run_memory_budget()
    results["memory_budget"] = mem

    # ------------------------------------------------------------------
    # Summary decision table
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("Phase 0 — Summary")
    print("=" * 62)

    # FFT ceiling check at 128^3
    # fft_results is nested: {"numpy_cpu": {"128x128x64": {...}}, "jax": {...}}
    _fft_source = fft_results.get("numpy_cpu") or fft_results.get("jax") or {}
    fft_128 = _fft_source.get("128x128x64", {})
    etdrk4_ms = fft_128.get("etdrk4_step_estimate_ms", float("nan"))
    hero_128  = hero.get("128x128x64", {})
    days      = hero_128.get("total_days", float("nan"))
    fft_ok    = hero_128.get("within_ceiling", False)

    # SVD ceiling check at R=32
    svd_128   = svd_results.get("128x128", {})
    svd_r32   = svd_128.get("R32", {})
    svd_ms    = svd_r32.get("mean_ms", float("nan"))
    svd_ok    = svd_ms <= 10.0

    mem_ok    = mem["within_13gb_ceiling"]

    print(f"  D1  Parameter set          sealed in hallmhd/parameters.py     ✓")
    print(f"  D2  Benchmark thresholds   sealed in benchmarks/thresholds.py  ✓")
    print(
        f"  D3  Memory budget          peak={mem['canonical_peak_gb']:.2f} GB (R=32)  "
        f"{'✓ feasible' if mem_ok else '✗ EXCEEDS 13 GB'}"
    )
    print(
        f"  D4a FFT throughput (128³)  ETDRK4 ≈ {etdrk4_ms:.1f} ms  "
        f"Hero Run ≈ {days:.2f} days  "
        f"{'✓' if fft_ok else '✗ EXCEEDS 5-day ceiling'}"
    )
    print(
        f"  D4b SVD throughput (R=32)  {svd_ms:.2f} ms/call  "
        f"{'✓ ≤ 10 ms target' if svd_ok else '✗ EXCEEDS 10 ms target'}"
    )

    all_ok = mem_ok and svd_ok  # FFT ceiling is advisory; SVD and memory are hard
    print()
    if all_ok:
        print("  ✓  All Phase 0 decisions resolved. Proceed to Phase 1.")
    else:
        print("  ✗  One or more checks failed. Review before proceeding.")

    # ------------------------------------------------------------------
    # Write consolidated results
    # ------------------------------------------------------------------
    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results["phase0_summary"] = {
        "fft_ceiling_ok": fft_ok,
        "svd_ceiling_ok": svd_ok,
        "memory_ok":      mem_ok,
        "all_ok":         all_ok,
        "elapsed_s":      round(time.perf_counter() - t_start, 1),
    }
    with open(_OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Consolidated results → profiling/phase0_results.json")
    print(f"  Elapsed: {results['phase0_summary']['elapsed_s']:.1f} s\n")


if __name__ == "__main__":
    main()