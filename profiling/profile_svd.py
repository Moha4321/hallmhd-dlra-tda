r"""
profile_svd.py
==============

Phase~0, Decision~D4: Apple Accelerate SVD throughput measurement.

Benchmarks the Halko–Martinsson–Tropp randomised SVD on
:math:`128\times128` matrices at ranks :math:`R\in\{8,16,32,64\}`.
This is the performance-critical kernel for the DLRA K/S/L steps
(Phase~2A, §4.3.1).

The target is :math:`\lesssim 10\,\text{ms}` per call at
:math:`128\times128`, :math:`R=32` (§8.3 of the proposal).

Results are written to ``profiling/svd_results.json``.

Usage
-----
.. code-block:: bash

    python profiling/profile_svd.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_OUTPUT_FILE = _REPO_ROOT / "profiling" / "svd_results.json"

N_WARMUP = 5
N_TRIALS = 20

# Matrix sizes and ranks to benchmark
MATRIX_SIZES: list[int] = [128, 256]
RANKS: list[int] = [8, 16, 32, 64]
N_OVERSAMPLE: int = 10          # oversampling parameter p in HMT


# ---------------------------------------------------------------------------
# Randomised SVD (Halko–Martinsson–Tropp, 2011)
# ---------------------------------------------------------------------------

def randomised_svd(
    A: np.ndarray,
    rank: int,
    n_oversample: int = N_OVERSAMPLE,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Randomised SVD of :math:`A \in \mathbb{R}^{m \times n}` to rank
    :math:`R` via the Halko–Martinsson–Tropp algorithm [HMT2011]_.

    Algorithm
    ---------
    1. Draw :math:`\Omega \in \mathbb{R}^{n \times (R+p)}`,
       :math:`p` = *n_oversample*.
    2. Form :math:`Y = A\Omega` and orthogonalise:
       :math:`Q, \_ = \mathrm{QR}(Y)`.
    3. Project: :math:`B = Q^\top A`.
    4. Thin SVD of :math:`B`:
       :math:`\hat{U}, \Sigma, V^\top = \mathrm{SVD}(B)`.
    5. Recover: :math:`U = Q\hat{U}`.

    Complexity
    ----------
    :math:`O(mnR)` vs :math:`O(mn\min(m,n))` for full SVD.
    On Apple Silicon, steps 2 and 4 are dispatched to Accelerate
    BLAS/LAPACK via NumPy's linkage.

    Parameters
    ----------
    A:
        Input matrix, ``float64``.
    rank:
        Target rank :math:`R`.
    n_oversample:
        Oversampling :math:`p`.  Default 10 gives near-optimal accuracy.
    rng:
        Optional ``numpy.random.Generator`` for reproducibility.

    Returns
    -------
    U : ndarray, shape (m, rank)
    S : ndarray, shape (rank,)
    Vt : ndarray, shape (rank, n)

    References
    ----------
    .. [HMT2011] N. Halko, P. G. Martinsson & J. A. Tropp,
       "Finding structure with randomness," *SIAM Rev.* **53**, 217 (2011).
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    m, n = A.shape
    k = rank + n_oversample

    Omega = rng.standard_normal((n, k)).astype(A.dtype)   # Gaussian sketch
    Y     = A @ Omega                                       # (m, k)
    Q, _  = np.linalg.qr(Y)                               # (m, k) orthonormal
    B     = Q.T @ A                                         # (k, n)
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)  # thin SVD of small B
    U     = Q @ U_hat                                       # (m, k) → (m, rank)

    return U[:, :rank], S[:rank], Vt[:rank, :]


# ---------------------------------------------------------------------------
# Core timer
# ---------------------------------------------------------------------------

def _timer_ms(fn, n_warmup: int = N_WARMUP, n_trials: int = N_TRIALS) -> dict:
    r"""Time a zero-argument callable in milliseconds."""
    for _ in range(n_warmup):
        fn()

    times: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
    }


# ---------------------------------------------------------------------------
# SVD profiling
# ---------------------------------------------------------------------------

def profile_randomised_svd(
    matrix_sizes: list[int] = MATRIX_SIZES,
    ranks: list[int] = RANKS,
    n_oversample: int = N_OVERSAMPLE,
    n_warmup: int = N_WARMUP,
    n_trials: int = N_TRIALS,
) -> dict:
    r"""
    Benchmark :func:`randomised_svd` at each ``(matrix_size, rank)`` pair.

    The DLRA K-step, S-step, and L-step each call the randomised SVD once
    on a matrix of size :math:`N_\text{dir} \times R`, where
    :math:`N_\text{dir} \in \{N_x, N_y, N_z\}`.  At the baseline grid
    (128×128×64) the largest factor is :math:`128`.

    Parameters
    ----------
    matrix_sizes:
        Square matrix side-lengths :math:`N` to benchmark.
    ranks:
        Ranks :math:`R` to benchmark.
    n_oversample:
        HMT oversampling parameter :math:`p`.
    n_warmup, n_trials:
        Profiling parameters.

    Returns
    -------
    dict
        Keyed by ``"NxN"`` → ``"RR"`` → timing dict.
    """
    results: dict = {}
    rng = np.random.default_rng(seed=42)

    for N in matrix_sizes:
        key_n = f"{N}x{N}"
        results[key_n] = {}
        A = rng.standard_normal((N, N)).astype(np.float64)

        print(f"\n  Matrix size {N}×{N}  (float64, Accelerate BLAS)")
        print(f"  {'Rank':>6}  {'mean (ms)':>10}  {'std (ms)':>9}  {'min (ms)':>9}  {'Target ≤10 ms':>14}")
        print("  " + "-" * 56)

        for R in ranks:
            def fn(rank=R, mat=A):
                randomised_svd(mat, rank, n_oversample)

            timing = _timer_ms(fn, n_warmup=n_warmup, n_trials=n_trials)
            ok = "✓" if timing["mean_ms"] <= 10.0 else "✗"
            print(
                f"  R={R:>3}   "
                f"{timing['mean_ms']:>10.2f}  "
                f"{timing['std_ms']:>9.2f}  "
                f"{timing['min_ms']:>9.2f}  "
                f"{'  ' + ok:>14}"
            )
            results[key_n][f"R{R}"] = timing

    return results


# ---------------------------------------------------------------------------
# Full SVD comparison (reference)
# ---------------------------------------------------------------------------

def profile_full_svd(
    matrix_size: int = 128,
    n_warmup: int = N_WARMUP,
    n_trials: int = N_TRIALS,
) -> dict:
    r"""
    Benchmark ``numpy.linalg.svd`` (full) at *matrix_size* × *matrix_size*.

    Provides a reference to quantify the speedup from randomisation.

    Returns
    -------
    dict
        Single timing dict with key ``"full_svd"``.
    """
    rng = np.random.default_rng(seed=0)
    A   = rng.standard_normal((matrix_size, matrix_size)).astype(np.float64)

    def fn():
        np.linalg.svd(A, full_matrices=False)

    timing = _timer_ms(fn, n_warmup=n_warmup, n_trials=n_trials)
    print(
        f"\n  Full SVD {matrix_size}×{matrix_size}: "
        f"mean={timing['mean_ms']:.2f} ms  (reference)"
    )
    return {"full_svd": timing}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> dict:
    print("\n" + "=" * 62)
    print("Phase 0 — Randomised SVD (HMT) throughput profiling")
    print("=" * 62)
    print(f"  NumPy version  : {np.__version__}")
    print(f"  BLAS/LAPACK    : Apple Accelerate (via NumPy linkage)")
    print(f"  n_warmup       : {N_WARMUP}")
    print(f"  n_trials       : {N_TRIALS}")
    print(f"  n_oversample   : {N_OVERSAMPLE}")

    results: dict = {}

    # --- Randomised SVD at all (size, rank) pairs ---
    results["randomised_svd"] = profile_randomised_svd()

    # --- Full SVD reference at 128×128 ---
    print("\n--- Full SVD reference (128×128) ---")
    ref = profile_full_svd(matrix_size=128)
    results["full_svd_reference_128x128"] = ref

    # --- Speedup summary ---
    print("\n--- Randomised vs Full SVD speedup (128×128) ---")
    full_ms = ref["full_svd"]["mean_ms"]
    rand_results_128 = results["randomised_svd"].get("128x128", {})
    print(f"  {'Rank':>6}  {'Rand (ms)':>10}  {'Speedup':>10}")
    print("  " + "-" * 32)
    for R in RANKS:
        rand_ms = rand_results_128.get(f"R{R}", {}).get("mean_ms", float("nan"))
        speedup = full_ms / rand_ms if rand_ms > 0 else float("nan")
        print(f"  R={R:>3}   {rand_ms:>10.2f}  {speedup:>9.1f}×")

    # --- Save ---
    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written → {_OUTPUT_FILE.relative_to(_REPO_ROOT)}")

    return results


if __name__ == "__main__":
    main()