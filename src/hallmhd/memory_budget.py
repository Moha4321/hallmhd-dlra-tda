r"""
memory_budget.py
================

Convenience re-export of :class:`~hallmhd.parameters.MemoryBudget` and
the canonical memory report for Phase~0.

The full implementation lives in :mod:`hallmhd.parameters` alongside the
dataclasses it depends on (``GridConfig``, ``DLRAConfig``).  This module
exposes a single public function :func:`report` and the pre-constructed
singleton :data:`MEMORY` so that profiling scripts and the shell launcher
can do::

    from hallmhd.memory_budget import MEMORY, report, verify_ceiling

without importing from ``parameters`` directly.

All arithmetic follows Table~5 of the proposal (§8.2):

.. math::

    M_\text{state}^{\text{full}} &= N_x N_y N_z \cdot N_f \cdot 8\,\text{B} \\
    M_\text{peak}^{\text{full}}  &= 5 \times 1.2 \times M_\text{state}^{\text{full}} \\
    M_\text{state}^{\text{DLRA}} &= \bigl[(N_x+N_y+N_z)R + R^3\bigr]
                                     \cdot N_f \cdot 8\,\text{B} \\
    M_\text{peak}^{\text{DLRA}}  &= 6 \times M_\text{state}^{\text{DLRA}}

Hardware ceiling: 13 GB operating (16 GB physical minus 3 GB OS + JAX cache).
"""

from __future__ import annotations

from hallmhd.parameters import (
    DLRAConfig,
    GridConfig,
    MemoryBudget,
    GRID,
    DLRA,
    MEMORY,
)

__all__ = [
    "MemoryBudget",
    "GridConfig",
    "DLRAConfig",
    "MEMORY",
    "report",
    "verify_ceiling",
    "table",
]

# Hard ceiling from the proposal (§8.2): 16 GB physical − 3 GB headroom.
_CEILING_GB: float = 13.0
_GB: int = 1024 ** 3


def report() -> str:
    r"""Return the canonical memory budget report string (prints Table~5)."""
    return MEMORY.report()


def verify_ceiling(memory: MemoryBudget | None = None) -> bool:
    r"""
    Return ``True`` iff the DLRA peak usage fits within the 13 GB ceiling.

    Parameters
    ----------
    memory:
        Budget object to check.  Defaults to the canonical singleton
        :data:`MEMORY` (128×128×64, R=32).
    """
    mb = memory or MEMORY
    return mb.peak_bytes_dlra() / _GB <= _CEILING_GB


# ---------------------------------------------------------------------------
# Table 5 reproduction
# ---------------------------------------------------------------------------

_CONFIGS: list[tuple[str, GridConfig, int | None]] = [
    ("64×64×32   full-rank",   GridConfig(nx=64,  ny=64,  nz=32),   None),
    ("128×128×64 full-rank",   GridConfig(nx=128, ny=128, nz=64),   None),
    ("128×128×64 R=32",        GridConfig(nx=128, ny=128, nz=64),   32),
    ("256×256×128 R=32",       GridConfig(nx=256, ny=256, nz=128),  32),
    ("512×512×128 R=32",       GridConfig(nx=512, ny=512, nz=128),  32),
]


def table() -> str:
    r"""
    Reproduce Table~5 of the proposal as a plain-text string.

    Returns
    -------
    str
        Formatted memory budget table for all key configurations.
    """
    header = (
        f"\n{'Configuration':<26} {'State (GB)':>11} "
        f"{'Peak (GB)':>10} {'≤13 GB':>8}"
    )
    sep = "-" * 60
    rows = [header, sep]

    for label, grid, rank in _CONFIGS:
        r = rank or 32
        mb = MemoryBudget(grid=grid, dlra=DLRAConfig(r_initial=r))
        if rank is None:
            state = mb.state_bytes_fullrank()
            peak  = mb.peak_bytes_fullrank()
        else:
            state = mb.state_bytes_dlra(rank)
            peak  = mb.peak_bytes_dlra(rank)

        feasible = "YES" if peak / _GB <= _CEILING_GB else "NO (chunked)"
        rows.append(
            f"  {label:<24} {state/_GB:>11.3f} {peak/_GB:>10.3f} {feasible:>8}"
        )

    rows.append(sep)
    rows.append(f"  Hardware ceiling (operating): {_CEILING_GB:.1f} GB")
    rows.append(f"  Physical RAM:                 16.0 GB\n")
    return "\n".join(rows)


if __name__ == "__main__":
    print(report())
    print(table())
    ok = verify_ceiling()
    print(f"DLRA (R=32, 128×128×64) within 13 GB ceiling: {'YES ✓' if ok else 'NO ✗'}")