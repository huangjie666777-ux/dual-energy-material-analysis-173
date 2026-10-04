"""Dual-energy material decomposition: per-pixel non-negative 2-material fit.

Model (linear two-material assumption): the reconstructed attenuation map at
energy e is

    mu_e(x, y) = sum_j A[e, j] * rho_j(x, y)

with A the mass attenuation matrix (mm^2/mg, rows = low/high energy,
columns = materials) and rho_j the material density (mg/mm^3).
Each pixel is solved as an exact 2-variable NNLS problem; negative
components of the unconstrained solution are never simply clipped.
"""

from __future__ import annotations

import numpy as np

from .io_utils import ValidationError

MAX_CONDITION_NUMBER = 10000.0


def validate_materials(names, matrix) -> dict:
    """Validate two unique material names and the 2x2 attenuation matrix."""
    if not isinstance(names, (list, tuple)) or len(names) != 2:
        raise ValidationError("exactly two material names are required")
    cleaned = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("material names must be non-empty strings")
        cleaned.append(name.strip())
    if cleaned[0] == cleaned[1]:
        raise ValidationError("material names must be unique")

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (2, 2):
        raise ValidationError(
            "mass attenuation matrix must be 2x2 "
            "(rows: low/high energy, columns: materials)"
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValidationError(
            "mass attenuation coefficients must be finite and strictly positive"
        )
    condition = float(np.linalg.cond(arr, 2))
    if condition > MAX_CONDITION_NUMBER:
        raise ValidationError(
            f"mass attenuation matrix is ill-conditioned (cond={condition:.3g} > "
            f"{MAX_CONDITION_NUMBER:g}); the two materials are not identifiable"
        )
    return {"names": cleaned, "matrix": arr, "condition_number": condition}


def nnls_2x2(matrix: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Exact vectorized NNLS for a 2x2 matrix.

    observations has shape (2, n); returns shape (2, n) with the
    exact non-negative least-squares solution per column. Negative
    observations are kept as-is (never clipped before the fit).
    """
    ata = matrix.T @ matrix
    atb = matrix.T @ observations
    det = ata[0, 0] * ata[1, 1] - ata[0, 1] * ata[1, 0]
    unconstrained = np.empty_like(atb)
    unconstrained[0] = (ata[1, 1] * atb[0] - ata[0, 1] * atb[1]) / det
    unconstrained[1] = (ata[0, 0] * atb[1] - ata[0, 1] * atb[0]) / det

    def cost(x: np.ndarray) -> np.ndarray:
        return (
            0.5 * (ata[0, 0] * x[0] * x[0]
                   + 2.0 * ata[0, 1] * x[0] * x[1]
                   + ata[1, 1] * x[1] * x[1])
            - (atb[0] * x[0] + atb[1] * x[1])
        )

    best = np.zeros_like(atb)
    best_cost = cost(best)

    interior = (unconstrained[0] >= 0.0) & (unconstrained[1] >= 0.0)
    interior_cost = np.where(interior, cost(unconstrained), np.inf)
    take = interior_cost < best_cost
    best = np.where(take, unconstrained, best)
    best_cost = np.where(take, interior_cost, best_cost)

    for j in (0, 1):
        candidate = np.zeros_like(atb)
        candidate[j] = np.maximum(atb[j] / ata[j, j], 0.0)
        candidate_cost = cost(candidate)
        take = candidate_cost < best_cost
        best = np.where(take, candidate, best)
        best_cost = np.where(take, candidate_cost, best_cost)
    return best


def decompose(matrix: np.ndarray, low_map: np.ndarray, high_map: np.ndarray):
    """Fit non-negative densities and return (densities, residuals).

    densities has shape (2, n, n) in mg/mm^3; residuals has shape
    (2, n, n) and equals matrix @ densities - observations per energy
    (row 0 = low, row 1 = high), in mm^-1.
    """
    if low_map.shape != high_map.shape:
        raise ValidationError(
            f"low/high reconstructions must share shape, got "
            f"{low_map.shape} vs {high_map.shape}"
        )
    observations = np.stack([low_map.ravel(), high_map.ravel()])
    densities = nnls_2x2(matrix, observations)
    predicted = matrix @ densities
    residuals = predicted - observations
    shape = low_map.shape
    return (
        densities.reshape((2,) + shape),
        residuals.reshape((2,) + shape),
    )

