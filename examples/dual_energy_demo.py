"""Generate a two-material dual-energy phantom and print curl commands.

Analytic phantom: an aluminum disk and a PVC disk at different positions.
Per energy e the line integral is the linear two-material mixture

    p_e(theta, t) = sum_j A[e, j] * q_j(theta, t)

where q_j is the density line integral of material j (disk chord formula,
density in mg/mm^3) and A is the mass attenuation matrix in mm^2/mg.

Run: .venv/bin/python examples/dual_energy_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctrecon.synthetic import acquire, disk_sinogram

N_ANGLES, N_DET = 180, 256
DETECTOR_SPACING = 0.5
CENTER_INDEX = 127.5
PIXEL_SPACING = 0.5
OUTPUT_SIZE = 128
SLICE_THICKNESS_MM = 1.0

MATERIAL_NAMES = ["aluminum", "pvc"]
# Rows: low / high energy; columns: aluminum, pvc. Units: mm^2/mg.
ATTENUATION_MATRIX = [
    [0.030, 0.012],
    [0.012, 0.008],
]
# Disk: (center_mm, radius_mm, density mg/mm^3) per material.
DISKS = {
    "aluminum": ((-8.0, 3.0), 7.0, 2.7),
    "pvc": ((7.0, -4.0), 9.0, 1.3),
}


def energy_sinogram(energy_row: np.ndarray) -> np.ndarray:
    total = np.zeros((N_ANGLES, N_DET), dtype=np.float64)
    for j, name in enumerate(MATERIAL_NAMES):
        center, radius, density = DISKS[name]
        q = disk_sinogram(
            N_ANGLES, N_DET, DETECTOR_SPACING, CENTER_INDEX,
            disk_center_mm=center, radius_mm=radius, attenuation=density,
        )
        total += energy_row[j] * q
    return total


def main() -> None:
    matrix = np.asarray(ATTENUATION_MATRIX, dtype=np.float64)
    rng = np.random.default_rng(0)
    out_dir = Path(__file__).resolve().parent
    for label, row in (("low", matrix[0]), ("high", matrix[1])):
        line_integrals = energy_sinogram(row)
        intensity, dark, flat = acquire(line_integrals, rng=rng)
        np.savez(
            out_dir / f"dual_energy_{label}.npz",
            intensity=intensity, dark=dark, flat=flat,
        )
        print("wrote", out_dir / f"dual_energy_{label}.npz")

    rois = [
        {"name": "al_disk", "x0": 36, "y0": 46, "x1": 60, "y1": 70},
        {"name": "pvc_disk", "x0": 62, "y0": 56, "x1": 94, "y1": 88},
    ]
    materials = {
        "names": MATERIAL_NAMES,
        "mass_attenuation_matrix_mm2_per_mg": ATTENUATION_MATRIX,
    }
    print(
        "curl -s -X POST http://127.0.0.1:8000/decompose \
"
        "  -F 'low_file=@examples/dual_energy_low.npz' \
"
        "  -F 'high_file=@examples/dual_energy_high.npz' \
"
        f"  -F 'detector_spacing_mm={DETECTOR_SPACING}' \
"
        f"  -F 'center_index={CENTER_INDEX}' \
"
        f"  -F 'output_size={OUTPUT_SIZE}' \
"
        f"  -F 'pixel_spacing_mm={PIXEL_SPACING}' \
"
        "  -F 'filter=hann' \
"
        f"  -F 'slice_thickness_mm={SLICE_THICKNESS_MM}' \
"
        f"  -F 'materials={json.dumps(materials)}' \
"
        f"  -F 'rois={json.dumps(rois)}' \
"
        "  -o decomposition.zip"
    )


if __name__ == "__main__":
    main()

