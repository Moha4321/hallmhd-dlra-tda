#!/usr/bin/env bash
# =============================================================================
# run_phase0_profiling.sh
#
# Phase 0 hardware profiling launcher (Month 1, Decision D4).
#
# Runs the full profiling suite from the repo root:
#   1. FFT throughput   → profiling/fft_results.json
#   2. SVD throughput   → profiling/svd_results.json
#   3. Memory budget    → stdout (from hallmhd.memory_budget)
#   4. Consolidated     → profiling/phase0_results.json
#
# Usage
# -----
#   bash scripts/run_phase0_profiling.sh
#
# Requirements
# ------------
#   conda activate base          (or whichever env has the deps)
#   pip install -e ".[dev]"      (installs hallmhd package in editable mode)
#   pip install jax-metal        (for M4 GPU backend; optional but recommended)
# =============================================================================

set -euo pipefail

# Resolve repo root regardless of where the script is called from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "============================================================"
echo "  hallmhd-dlra-tda  —  Phase 0 Profiling"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Repo: $REPO_ROOT"
echo "============================================================"

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
echo ""
echo "--- Environment ---"
python --version
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import jax; print('jax', jax.__version__, '|', jax.default_backend())" \
    || echo "[WARNING] JAX not found — FFT will use NumPy CPU fallback."

echo ""
echo "--- Package import check ---"
python -c "from hallmhd.parameters import GRID, MEMORY; print('hallmhd.parameters  ✓')"
python -c "from hallmhd.memory_budget import MEMORY, verify_ceiling; print('hallmhd.memory_budget ✓')"
python -c "from benchmarks.thresholds import THRESHOLDS; print('benchmarks.thresholds ✓')"

# ---------------------------------------------------------------------------
# 1. FFT profiling (standalone, writes profiling/fft_results.json)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 1 / 4 — FFT throughput"
echo "============================================================"
JAX_PLATFORMS=cpu python profiling/profile_fft.py

# ---------------------------------------------------------------------------
# 2. SVD profiling (standalone, writes profiling/svd_results.json)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 2 / 4 — Randomised SVD throughput"
echo "============================================================"
JAX_PLATFORMS=cpu python profiling/profile_svd.py

# ---------------------------------------------------------------------------
# 3. Memory budget (quick print from memory_budget module)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 3 / 4 — Memory budget (Table 5)"
echo "============================================================"
JAX_PLATFORMS=cpu python -m hallmhd.memory_budget

# ---------------------------------------------------------------------------
# 4. Consolidated orchestrator (writes profiling/phase0_results.json)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 4 / 4 — Consolidated Phase 0 summary"
echo "============================================================"
JAX_PLATFORMS=cpu python scripts/phase0_profile.py

# ---------------------------------------------------------------------------
# 5. Seal: SHA-256 hash of thresholds.py
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Sealing benchmarks/thresholds.py"
echo "============================================================"
HASH=$(shasum -a 256 benchmarks/thresholds.py)
echo "  SHA-256: $HASH"
echo "  Record this in docs/phase0_architecture.tex before proceeding."
echo ""
echo "  To verify later:"
echo "    shasum -a 256 --check <(echo '$HASH')"

echo ""
echo "============================================================"
echo "  Phase 0 complete. Review profiling/phase0_results.json"
echo "  then proceed to Phase 1 (3D spectral engine)."
echo "============================================================"
echo ""