import numpy as np
import pytest

from ctrecon.io_utils import ValidationError
from ctrecon.rois import integrate_rois, parse_rois


def test_parse_valid_rois():
    rois = parse_rois([
        {"name": "a", "x0": 0, "y0": 0, "x1": 4, "y1": 4},
        {"name": "b", "x0": 4, "y0": 4, "x1": 8, "y1": 8},
    ], image_size=8)
    assert [r.name for r in rois] == ["a", "b"]
    assert rois[0].n_pixels == 16


@pytest.mark.parametrize("spec", [
    [],
    [{"name": "a", "x0": 0, "y0": 0, "x1": 2, "y1": 2}] * 9,
    [{"name": "a", "x0": 0, "y0": 0, "x1": 2, "y1": 2},
     {"name": "a", "x0": 2, "y0": 2, "x1": 4, "y1": 4}],
    [{"name": "", "x0": 0, "y0": 0, "x1": 2, "y1": 2}],
    [{"name": "a", "x0": 2, "y0": 0, "x1": 2, "y1": 2}],
    [{"name": "a", "x0": -1, "y0": 0, "x1": 2, "y1": 2}],
    [{"name": "a", "x0": 0, "y0": 0, "x1": 9, "y1": 2}],
    [{"name": "a", "x0": 0, "y0": 0, "x1": 2}],
    ["not-a-dict"],
])
def test_invalid_rois_rejected(spec):
    with pytest.raises(ValidationError):
        parse_rois(spec, image_size=8)


def test_integrate_mass_and_residual():
    densities = np.full((2, 8, 8), 2.0)
    residuals = np.zeros((2, 8, 8))
    residuals[0] = 0.1
    residuals[1] = -0.2
    rois = parse_rois([{"name": "block", "x0": 1, "y0": 2, "x1": 5, "y1": 6}], 8)
    results = integrate_rois(rois, ["m0", "m1"], densities, residuals, 0.5, 2.0)
    (result,) = results
    assert result["n_pixels"] == 16
    # 2 mg/mm^3 * 16 px * (0.5*0.5*2) mm^3/px = 16 mg
    assert result["mass_mg"]["m0"] == pytest.approx(16.0)
    assert result["mass_mg"]["m1"] == pytest.approx(16.0)
    assert result["mean_residual_per_mm"]["low"] == pytest.approx(0.1)
    assert result["mean_residual_per_mm"]["high"] == pytest.approx(-0.2)

