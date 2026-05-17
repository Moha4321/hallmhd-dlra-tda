r"""
profile_fft.py
==============

Phase~0, Decision~D4: Apple GPU throughput measurement for 3D real FFT.

Measures forward + inverse :math:`\mathcal{F}^{-1}\{\mathcal{F}\{f\}\}` at
resolutions :math:`64\times64\times32`, :math:`128\times128\times64`, and
:math:`256\times256\times128`.

Precision notes
---------------
* **Metal (M1/M2/M3/M4 GPU)**: JAX Metal only supports ``float32``.
  The profiler detects this and runs ``float32`` on Metal, then also
  runs ``float64`` via NumPy on CPU for reference.  The proposal's solver
  will use ``float64``; the Metal path gives a *lower bound* on step cost
  (float32 is faster).  The CPU NumPy timing gives the conservative
  estimate used for Hero Run planning.
* **CPU with JAX_ENABLE_X64**: set ``JAX_ENABLE_X64=1`` to get true
  float64 JAX timings on CPU.

The ETDRK4 integrator requires approximately 15 FFT roundtrips per timestep.

Usage
-----
.. code-block:: bash

    python profiling/profile_fft.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_OUTPUT_FILE = _REPO_ROOT / "profiling" / "fft_results.json"

N_WARMUP = 5
N_TRIALS = 20

RESOLUTIONS: list[tuple[int, int, int]] = [
    (64,  64,  32),
    (128, 128, 64),
    (256, 256, 128),
]

# ---------------------------------------------------------------------------
# JAX import + Metal float64 detection
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    _backend     = jax.default_backend()
    _METAL       = _backend.upper() == "METAL"
    _JAX_DTYPE   = jnp.float32 if _METAL else jnp.float64
    _DTYPE_LABEL = "float32 (Metal limit)" if _METAL else "float64"
    JAX_AVAILABLE = True

except ImportError:
    JAX_AVAILABLE = False
    _METAL        = False
    _backend      = "none"
    _JAX_DTYPE    = None
    _DTYPE_LABEL  = "n/a"


# ---------------------------------------------------------------------------
# Core timer
# ---------------------------------------------------------------------------

def _timer_ms(fn, n_warmup: int = N_WARMUP, n_trials: int = N_TRIALS) -> dict:
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
# JAX FFT benchmark
# ---------------------------------------------------------------------------

def _profile_jax_fft(
    resolutions: list[tuple[int, int, int]],
    n_warmup: int,
    n_trials: int,
) -> dict:
    results: dict = {}

    for nx, ny, nz in resolutions:
        key = f"{nx}x{ny}x{nz}"
        print(f"  [jax/{_backend}/{_DTYPE_LABEL}]  3D rFFT {key} ...", flush=True)

        x = jnp.ones((nx, ny, nz), dtype=_JAX_DTYPE)

        @jit
        def _roundtrip(arr):
            return jnp.fft.irfftn(jnp.fft.rfftn(arr))

        _ = _roundtrip(x).block_until_ready()

        def fn():
            _roundtrip(x).block_until_ready()

        timing = _timer_ms(fn, n_warmup=n_warmup, n_trials=n_trials)
        timing["backend"]  = _backend
        timing["dtype"]    = _DTYPE_LABEL
        timing["etdrk4_step_estimate_ms"] = timing["mean_ms"] * 15

        if _METAL:
            timing["note"] = (
                "float32 only on Metal. float64 cost ~2-4x higher. "
                "See numpy_cpu entry for conservative Hero Run estimate."
            )

        results[key] = timing
        print(
            f"         mean = {timing['mean_ms']:7.2f} ms  "
            f"std = {timing['std_ms']:6.2f} ms  "
            f"→  ETDRK4 step ≈ {timing['etdrk4_step_estimate_ms']:.1f} ms"
        )
        if _METAL:
            print(f"         ⚠  Metal float32 — multiply by ~2–4x for float64 estimate")

    return results


# ---------------------------------------------------------------------------
# NumPy CPU float64 benchmark (always runs — conservative Hero Run baseline)
# ---------------------------------------------------------------------------

def _profile_numpy_fft(
    resolutions: list[tuple[int, int, int]],
    n_warmup: int,
    n_trials: int,
) -> dict:
    results: dict = {}

    for nx, ny, nz in resolutions:
        key = f"{nx}x{ny}x{nz}"
        print(f"  [numpy/cpu/float64]  3D rFFT {key} ...", flush=True)

        x = np.ones((nx, ny, nz), dtype=np.float64)

        def fn():
            np.fft.irfftn(np.fft.rfftn(x))

        timing = _timer_ms(fn, n_warmup=n_warmup, n_trials=n_trials)
        timing["backend"] = "numpy_cpu"
        timing["dtype"]   = "float64"
        timing["etdrk4_step_estimate_ms"] = timing["mean_ms"] * 15
        results[key] = timing

        print(
            f"         mean = {timing['mean_ms']:7.2f} ms  "
            f"std = {timing['std_ms']:6.2f} ms  "
            f"→  ETDRK4 step ≈ {timing['etdrk4_step_estimate_ms']:.1f} ms"
        )

    return results


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def profile_fft_3d(
    resolutions: list[tuple[int, int, int]] = RESOLUTIONS,
    n_warmup: int = N_WARMUP,
    n_trials: int = N_TRIALS,
) -> dict:
    r"""
    Benchmark 3D real FFT roundtrip.

    Always runs NumPy CPU float64 (conservative Hero Run estimate).
    If JAX is available, also runs on the active JAX backend
    (Metal = float32, reported separately).

    Returns
    -------
    dict with keys ``"jax"`` and/or ``"numpy_cpu"``, each mapping
    resolution strings to timing dicts.
    """
    results: dict = {}

    if JAX_AVAILABLE:
        print(f"\n  JAX backend: {_backend}  |  dtype: {_DTYPE_LABEL}")
        results["jax"] = _profile_jax_fft(resolutions, n_warmup, n_trials)
        print()

    print("  NumPy CPU float64 (Hero Run planning baseline):")
    results["numpy_cpu"] = _profile_numpy_fft(resolutions, n_warmup, n_trials)

    return results


def hero_run_estimate(fft_results: dict, n_steps: int = 150_000) -> dict:
    r"""
    Derive Hero Run wall-time from FFT timings.

    Uses ``numpy_cpu`` float64 as the conservative baseline.
    If only JAX Metal (float32) timings exist, applies a 3x penalty.

    Returns
    -------
    dict keyed by resolution string.
    """
    estimates: dict = {}

    if "numpy_cpu" in fft_results:
        source  = fft_results["numpy_cpu"]
        label   = "numpy_cpu float64"
        penalty = 1.0
    else:
        source  = fft_results.get("jax", {})
        label   = "jax Metal float32 x3 penalty"
        penalty = 3.0

    print(f"  Source: {label}")

    for key, timing in source.items():
        etdrk4_ms  = timing["etdrk4_step_estimate_ms"] * penalty
        total_days = etdrk4_ms * 1e-3 * n_steps / 86_400
        within     = total_days <= 5.0
        estimates[key] = {
            "etdrk4_ms":      round(etdrk4_ms, 2),
            "total_days":     round(total_days, 3),
            "within_ceiling": within,
            "source":         label,
        }
        flag = "OK  within 5-day ceiling" if within else "EXCEEDS — reduce steps"
        print(
            f"  {key:>14}  ETDRK4={etdrk4_ms:6.1f} ms  "
            f"Hero Run ~{total_days:.2f} days  [{flag}]"
        )

    return estimates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> dict:
    print("\n" + "=" * 62)
    print("Phase 0 — 3D real FFT throughput profiling")
    print("=" * 62)

    if JAX_AVAILABLE:
        print(f"  JAX {jax.__version__}  |  backend: {_backend}")
        print(f"  Devices : {jax.devices()}")
        if _METAL:
            print("  NOTE: Metal backend — profiling in float32.")
            print("        float64 timings from NumPy/CPU path below.")
    else:
        print("  JAX not found — NumPy CPU only.")

    print(f"  NumPy {np.__version__}")
    print()

    fft_results = profile_fft_3d()

    print("\n--- Hero Run wall-time estimates (150,000 steps) ---")
    hero = hero_run_estimate(fft_results)

    output = {"fft_3d": fft_results, "hero_run_estimates": hero}

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written -> {_OUTPUT_FILE.relative_to(_REPO_ROOT)}")

    return output


if __name__ == "__main__":
    main()