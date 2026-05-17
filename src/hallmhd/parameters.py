r"""
parameters.py
=============

Sealed physical and numerical parameters for the 3D Hall-MHD DLRA simulation.

All values fixed at the end of Phase~0 (Month~1) and must not be altered
without a documented decision record.  Any change after the Month~5 SVD
test constitutes a protocol violation.

Physical model
--------------
The governing system is the compressible 7-field Hall-MHD system in a
sheared flux-tube domain
:math:`(x,y,z)\in[0,L_x]\times[0,L_y]\times[0,L_z]`:

.. math::

    \frac{\partial\rho}{\partial t} &= -\nabla\cdot(\rho\mathbf{v})

    \frac{\partial(\rho\mathbf{v})}{\partial t} &=
        -\nabla\cdot(\rho\mathbf{v}\mathbf{v})
        - \nabla P
        + \mathbf{J}\times\mathbf{B}
        + \nu_4\nabla^4\mathbf{v}

    \frac{\partial\mathbf{B}}{\partial t} &=
        \nabla\times(\mathbf{v}\times\mathbf{B})
        - d_i\nabla\times\!\left(\frac{\mathbf{J}\times\mathbf{B}}{\rho}\right)
        + \eta_4\nabla^4\mathbf{B}

    \frac{\partial P}{\partial t} &=
        -\mathbf{v}\cdot\nabla P
        - \gamma P\nabla\cdot\mathbf{v}
        + (\gamma-1)\,\eta_4|\mathbf{J}|^2

where :math:`\mathbf{J}=\nabla\times\mathbf{B}/\mu_0` and the solenoidal
constraint :math:`\nabla\cdot\mathbf{B}=0` is enforced by evolving the
magnetic flux potential :math:`A_z` rather than :math:`\mathbf{B}` directly.

Twist-and-shift boundary condition in :math:`z`:

.. math::

    f(x,\,y,\,z+L_z) = f(x,\,y+\hat{s}x L_z,\,z),
    \qquad
    \hat{s} = \frac{r}{q}\frac{dq}{dr}
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Final

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (SI, then normalised)
# ---------------------------------------------------------------------------

#: ITER on-axis toroidal guide field :math:`B_T = 5.3\,\text{T}`.
B_TOROIDAL_SI: Final[float] = 5.3          # Tesla

#: Poloidal-to-toroidal field ratio
#: :math:`\varepsilon = B_p / B_T \ll 1`.
#: Controls the strength of the guide-field anisotropy.  A smaller
#: :math:`\varepsilon` drives faster exponential decay of the singular
#: value spectrum in the :math:`z`-direction (Flute-Mode Lemma, §3.2.1).
EPSILON: Final[float] = 0.1

#: Normalised ion inertial length
#: :math:`d_i = c/\omega_{pi}` (in units of the box half-width).
#: Governs the Hall term strength and the onset of fast reconnection
#: independent of resistivity.
D_ION_INERTIAL: Final[float] = 0.1

#: Effective Lundquist number at the grid scale
#: :math:`S_\text{eff} = \mu_0 L v_A / \eta`.
#: For ITER :math:`S\sim 10^8`; we operate at :math:`S_\text{eff}=10^6`
#: to maintain numerical stability at the target resolution.
S_LUNDQUIST: Final[float] = 1.0e6

#: Adiabatic index :math:`\gamma = 5/3` for a monatomic ideal plasma.
GAMMA: Final[float] = 5.0 / 3.0

#: Magnetic shear parameter
#: :math:`\hat{s} = (r/q)\,dq/dr` at the rational surface :math:`q=m/n`.
#: Enters the twist-and-shift boundary condition.
S_HAT: Final[float] = 0.8          # typical mid-radius value


# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridConfig:
    r"""
    Numerical grid specification for the flux-tube domain.

    The domain spans
    :math:`[0,L_x]\times[0,L_y]\times[0,L_z]`
    with the :math:`z`-axis aligned to the equilibrium guide field.

    Parameters
    ----------
    nx, ny, nz:
        Number of collocation points in each direction.
        Must be even for the 3/2-rule dealiased FFT.
    lx, ly, lz:
        Physical box lengths in normalised units.
    """
    nx: int = 128
    ny: int = 128
    nz: int = 64

    lx: float = 1.0       # perpendicular (radial)
    ly: float = 1.0       # perpendicular (binormal)
    lz: float = 2 * np.pi # parallel (field-aligned)

    def __post_init__(self) -> None:
        for attr, val in [("nx", self.nx), ("ny", self.ny), ("nz", self.nz)]:
            if val % 2 != 0:
                raise ValueError(f"{attr}={val} must be even for real FFT dealiasing.")

    @property
    def dx(self) -> float:
        r"""Grid spacing :math:`\Delta x = L_x / N_x`."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        r"""Grid spacing :math:`\Delta y = L_y / N_y`."""
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        r"""Grid spacing :math:`\Delta z = L_z / N_z`."""
        return self.lz / self.nz

    @property
    def n_total(self) -> int:
        r"""Total number of grid points :math:`N = N_x N_y N_z`."""
        return self.nx * self.ny * self.nz

    @property
    def n_fields(self) -> int:
        r"""Number of evolved fields: :math:`\{\rho,\,v_x,\,v_y,\,v_z,\,A_z,\,B_x,\,P\}`."""
        return 7


# ---------------------------------------------------------------------------
# Dissipation parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DissipationConfig:
    r"""
    Hyper-dissipation coefficients.

    Fourth-order (bi-harmonic) operators are used to model collisionless
    damping at the grid scale without resolving the Debye length.

    The hyper-resistivity :math:`\eta_4` is set such that the effective
    Lundquist number at the grid scale equals :math:`S_\text{eff}`:

    .. math::

        \eta_4 = \frac{v_A L^3}{S_\text{eff} k_{\max}^2}
    """
    #: Hyper-viscosity :math:`\nu_4` (normalised units).
    nu4: float = 5.0e-8
    #: Hyper-resistivity :math:`\eta_4` (normalised units).
    eta4: float = 5.0e-8


# ---------------------------------------------------------------------------
# Time-stepping parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeConfig:
    r"""
    Parameters for the ETDRK4 time integrator.

    The method solves :math:`\partial_t U = \mathcal{L}U + \mathcal{N}(U)`
    where :math:`\mathcal{L}` is the stiff linear operator (hyper-dissipation)
    and :math:`\mathcal{N}` is the nonlinear Hall-MHD right-hand side.

    The Kassam–Trefethen contour integral avoids catastrophic cancellation
    in the ETD coefficients near :math:`z=0`:

    .. math::

        \varphi(\mathcal{L}) = \frac{1}{2\pi i}
        \oint_\Gamma \varphi(z)(z I - \mathcal{L})^{-1}\,dz

    Parameters
    ----------
    dt:
        Timestep :math:`\Delta t` in Alfvén time units.
    n_contour:
        Number of quadrature points :math:`M` on the Kassam–Trefethen
        contour of radius :math:`\delta=1`.
    chunk_size:
        Number of timesteps :math:`N_c` per GPU chunk before async
        transfer to CPU for TDA.
    n_total_steps:
        Total number of timesteps for the Hero Run.
    """
    dt: float = 5.0e-4
    n_contour: int = 32
    chunk_size: int = 25
    n_total_steps: int = 150_000


# ---------------------------------------------------------------------------
# DLRA parameters (Phase 2A, conditional)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DLRAConfig:
    r"""
    Parameters for the Dynamical Low-Rank Approximation engine.

    The DLRA propagates
    :math:`U(t)\approx\sum_{i=1}^R X_i(x,t)\otimes Y_i(y,t)\otimes Z_i(z,t)`
    using the Lubich–Oseledets projector-splitting integrator.

    Rank adaptivity
    ---------------
    The rank :math:`R(t)` is allowed to vary within
    :math:`[R_{\min},\,R_{\max}]`.  After each ETDRK4 substage the
    truncated SVD (randomised Halko–Martinsson–Tropp) is applied with
    threshold:

    .. math::

        \tau(t) = \delta_\text{tol}\,/\,T

    where :math:`T` is the total simulation time, ensuring bounded
    cumulative truncation error (Theorem~3.1 of the proposal).
    """
    r_min: int = 4
    r_max: int = 64
    r_initial: int = 32
    #: Relative tolerance for rank truncation.
    delta_tol: float = 1.0e-4
    #: Oversampling parameter for randomised SVD.
    n_oversample: int = 10


# ---------------------------------------------------------------------------
# Memory budget (Phase 0 verification)
# ---------------------------------------------------------------------------

@dataclass
class MemoryBudget:
    r"""
    Analytical memory estimate for a given grid and DLRA rank.

    All estimates assume ``float64`` (8 bytes per element).

    Full-rank state tensor
    ----------------------

    .. math::

        M_\text{state} = N_x N_y N_z \times N_\text{fields} \times 8\,\text{B}

    ETDRK4 requires 5 copies of the state vector (stage values + RHS
    evaluations), so:

    .. math::

        M_\text{ETDRK4} = 5\,M_\text{state}

    DLRA state (Tucker format, rank :math:`R`)
    ------------------------------------------

    Three factor matrices plus a core tensor:

    .. math::

        M_\text{DLRA} = (N_x + N_y + N_z)R + R^3

    times :math:`N_\text{fields}\times 8\,\text{B}`, giving approximate
    peak with 5 ETDRK4 buffers:

    .. math::

        M_\text{peak,DLRA} \approx 5\,(N_x+N_y+N_z)R\,N_f\times 8\,\text{B}
    """

    grid: GridConfig = field(default_factory=GridConfig)
    dlra: DLRAConfig = field(default_factory=DLRAConfig)

    def state_bytes_fullrank(self) -> int:
        r"""Full-rank state tensor size in bytes."""
        return self.grid.n_total * self.grid.n_fields * 8

    def peak_bytes_fullrank(self) -> int:
        r"""Peak RAM for full-rank ETDRK4 (5 copies + FFT workspace ~1.2×)."""
        return int(self.state_bytes_fullrank() * 5 * 1.2)

    def state_bytes_dlra(self, rank: int | None = None) -> int:
        r"""Tucker-format state size in bytes for a given rank."""
        R = rank or self.dlra.r_initial
        g = self.grid
        n_factor = (g.nx + g.ny + g.nz) * R + R**3
        return n_factor * g.n_fields * 8

    def peak_bytes_dlra(self, rank: int | None = None) -> int:
        r"""Peak RAM for DLRA ETDRK4 (5 copies + tSVD workspace)."""
        return self.state_bytes_dlra(rank) * 6   # 5 buffers + tSVD

    def report(self) -> str:
        r"""Human-readable memory budget summary."""
        gb = 1024**3
        lines = [
            "Memory Budget (Phase 0 verification)",
            "=" * 45,
            f"Grid: {self.grid.nx}×{self.grid.ny}×{self.grid.nz}, "
            f"{self.grid.n_fields} fields, float64",
            f"Full-rank state:       {self.state_bytes_fullrank()/gb:.2f} GB",
            f"Full-rank peak (×5):   {self.peak_bytes_fullrank()/gb:.2f} GB",
            f"DLRA state  (R={self.dlra.r_initial:2d}):  "
            f"{self.state_bytes_dlra()/gb:.3f} GB",
            f"DLRA peak   (R={self.dlra.r_initial:2d}):  "
            f"{self.peak_bytes_dlra()/gb:.3f} GB",
            "-" * 45,
            f"Hardware ceiling:      16.00 GB",
            f"Full-rank feasible:    "
            f"{'YES' if self.peak_bytes_fullrank() < 16*gb else 'NO — chunked streaming required'}",
            f"DLRA feasible:         "
            f"{'YES' if self.peak_bytes_dlra() < 13*gb else 'NO'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canonical config (single source of truth)
# ---------------------------------------------------------------------------

GRID    = GridConfig()
DISSIP  = DissipationConfig()
TIME    = TimeConfig()
DLRA    = DLRAConfig()
MEMORY  = MemoryBudget(grid=GRID, dlra=DLRA)


if __name__ == "__main__":
    print(MEMORY.report())