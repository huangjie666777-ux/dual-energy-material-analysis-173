import numpy as np
import pytest

from ctrecon.decomposition import decompose, nnls_2x2, validate_materials
from ctrecon.io_utils import ValidationError

MATRIX = [[0.03, 0.012], [0.012, 0.008]]


def test_valid_materials():
    clean = validate_materials(["aluminum", "pvc"], MATRIX)
    assert clean["names"] == ["aluminum", "pvc"]
    assert clean["matrix"].shape == (2, 2)
    assert clean["condition_number"] > 1.0


@pytest.mark.parametrize("names,matrix", [
    (["a"], MATRIX),
    (["a", "a"], MATRIX),
    (["", "b"], MATRIX),
    (["a", "b"], [[0.03, 0.012]]),
    (["a", "b"], [[0.03, 0.012], [0.012, 0.0]]),
    (["a", "b"], [[0.03, 0.012], [0.012, float("nan")]]),
    (["a", "b"], [[1.0, 1.0], [1.0, 1.0 + 1e-9]]),
])
def test_invalid_materials_rejected(names, matrix):
    with pytest.raises(ValidationError):
        validate_materials(names, matrix)


def test_ill_conditioned_matrix_rejected():
    with pytest.raises(ValidationError, match="ill-conditioned"):
        validate_materials(["a", "b"], [[1.0, 1.0], [1.0, 1.0001]])


def test_nnls_exact_interior_solution():
    matrix = np.asarray(MATRIX)
    rho_true = np.array([2.0, 1.5])
    obs = matrix @ rho_true
    out = nnls_2x2(matrix, obs[:, np.newaxis])
    assert np.allclose(out[:, 0], rho_true, atol=1e-10)


def test_nnls_negative_observation_not_clipped_to_unconstrained():
    matrix = np.asarray(MATRIX)
    # Strongly negative observation: exact NNLS optimum is the origin.
    obs = np.array([[-1.0], [-1.0]])
    out = nnls_2x2(matrix, obs)
    assert np.allclose(out[:, 0], [0.0, 0.0])
    # A naive clip of the unconstrained solution would also give zero here,
    # so use a mixed case where unconstrained has one negative component.
    obs2 = np.array([[0.5], [-0.2]])
    out2 = nnls_2x2(matrix, obs2)
    x = out2[:, 0]
    assert np.all(x >= 0.0)
    # Compare against brute-force grid to confirm exactness.
    grid = np.linspace(0, 2, 4001)
    best = None
    for a in grid:
        for b in (0.0,):
            r = matrix @ np.array([a, b]) - obs2[:, 0]
            c = r @ r
            if best is None or c < best[0]:
                best = (c, a, b)
    for b in grid:
        r = matrix @ np.array([0.0, b]) - obs2[:, 0]
        c = r @ r
        if c < best[0]:
            best = (c, 0.0, b)
    r = matrix @ x - obs2[:, 0]
    assert r @ r <= best[0] + 1e-12


def test_decompose_recovers_densities_and_zero_residual():
    matrix = np.asarray(MATRIX)
    n = 8
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.0, 3.0, size=(2, n, n))
    low = matrix[0, 0] * rho[0] + matrix[0, 1] * rho[1]
    high = matrix[1, 0] * rho[0] + matrix[1, 1] * rho[1]
    densities, residuals = decompose(matrix, low, high)
    assert densities.shape == (2, n, n)
    assert residuals.shape == (2, n, n)
    assert np.allclose(densities, rho, atol=1e-8)
    assert np.allclose(residuals, 0.0, atol=1e-8)


def test_decompose_shape_mismatch():
    with pytest.raises(ValidationError):
        decompose(np.asarray(MATRIX), np.zeros((4, 4)), np.zeros((5, 5)))

