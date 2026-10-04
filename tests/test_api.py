import io
import json
import zipfile

import numpy as np
from fastapi.testclient import TestClient

from ctrecon.app import app
from ctrecon.synthetic import acquire, disk_sinogram

client = TestClient(app)
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")


def _upload(intensity, dark, flat, **params):
    buffer = io.BytesIO()
    np.savez(buffer, intensity=intensity, dark=dark, flat=flat)
    files = {"file": ("scan.npz", buffer.getvalue(), "application/octet-stream")}
    return client.post("/reconstruct", files=files, data=params)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "ram-lak" in response.json()["filters"]


def test_reconstruct_endpoint_returns_zip_bundle():
    truth = disk_sinogram(90, 128, 0.5, 63.5, (4.0, 2.0), 8.0, 0.2)
    intensity, dark, flat = acquire(truth)
    response = _upload(
        intensity, dark, flat,
        detector_spacing_mm="0.5",
        center_index="63.5",
        output_size="64",
        pixel_spacing_mm="0.5",
        filter="ram-lak",
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    bundle = zipfile.ZipFile(io.BytesIO(response.content))
    assert set(bundle.namelist()) == {
        "reconstruction.npy", "preview.png", "metadata.json"}

    metadata = json.loads(bundle.read("metadata.json"))
    assert metadata["parameters"]["output_size"] == 64
    assert metadata["ranges"]["image_min"] < metadata["ranges"]["image_max"]
    assert bundle.read("preview.png")[:8] == PNG_MAGIC

    image = np.load(io.BytesIO(bundle.read("reconstruction.npy")),
                    allow_pickle=False)
    assert image.dtype == np.float64
    assert image.shape == (64, 64)


def test_bad_upload_returns_422():
    response = _upload(
        np.ones((4, 8)), np.zeros(7), np.ones(8),
        detector_spacing_mm="1", center_index="3.5",
        output_size="8", pixel_spacing_mm="1", filter="ram-lak",
    )
    assert response.status_code == 422
    assert "error" in response.json()


MATRIX = [[0.03, 0.012], [0.012, 0.008]]


def _dual_energy_npzs():
    from ctrecon.decomposition import nnls_2x2  # noqa: F401  (ensure import path)
    matrix = np.asarray(MATRIX)
    # Two disks of pure materials, analytic density line integrals.
    q_al = disk_sinogram(90, 128, 0.5, 63.5, (-6.0, 2.0), 6.0, 2.7)
    q_pvc = disk_sinogram(90, 128, 0.5, 63.5, (7.0, -3.0), 8.0, 1.3)
    npzs = []
    for row in matrix:
        p = row[0] * q_al + row[1] * q_pvc
        intensity, dark, flat = acquire(p)
        buffer = io.BytesIO()
        np.savez(buffer, intensity=intensity, dark=dark, flat=flat)
        npzs.append(buffer.getvalue())
    return npzs


def _decompose_call(materials=None, rois=None, **params):
    low_npz, high_npz = _dual_energy_npzs()
    files = {
        "low_file": ("low.npz", low_npz, "application/octet-stream"),
        "high_file": ("high.npz", high_npz, "application/octet-stream"),
    }
    data = {
        "detector_spacing_mm": "0.5",
        "center_index": "63.5",
        "output_size": "64",
        "pixel_spacing_mm": "0.5",
        "filter": "hann",
        "slice_thickness_mm": "1.0",
        "materials": json.dumps(materials or {
            "names": ["aluminum", "pvc"],
            "mass_attenuation_matrix_mm2_per_mg": MATRIX,
        }),
        "rois": json.dumps(rois if rois is not None else [
            {"name": "al", "x0": 10, "y0": 18, "x1": 30, "y1": 38},
            {"name": "pvc", "x0": 32, "y0": 24, "x1": 60, "y1": 52},
        ]),
    }
    data.update(params)
    return client.post("/decompose", files=files, data=data)


def test_decompose_endpoint_returns_zip_bundle():
    response = _decompose_call()
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    bundle = zipfile.ZipFile(io.BytesIO(response.content))
    assert set(bundle.namelist()) == {
        "density_aluminum.npy", "density_pvc.npy",
        "residual_low.npy", "residual_high.npy",
        "preview_aluminum.png", "preview_pvc.png",
        "result.json",
    }
    report = json.loads(bundle.read("result.json"))
    assert report["materials"]["names"] == ["aluminum", "pvc"]
    assert len(report["rois"]) == 2
    al_roi = next(r for r in report["rois"] if r["name"] == "al")
    # ROI covers the aluminum disk: aluminum mass dominates, pvc ~ 0.
    assert al_roi["mass_mg"]["aluminum"] > 10.0
    assert al_roi["mass_mg"]["pvc"] < 0.1 * al_roi["mass_mg"]["aluminum"]
    assert abs(al_roi["mean_residual_per_mm"]["low"]) < 0.05

    density = np.load(io.BytesIO(bundle.read("density_aluminum.npy")),
                      allow_pickle=False)
    assert density.dtype == np.float64
    assert density.shape == (64, 64)
    assert density.min() >= 0.0
    assert bundle.read("preview_pvc.png")[:8] == PNG_MAGIC


def test_decompose_rejects_mismatched_shapes():
    low_npz, _ = _dual_energy_npzs()
    buffer = io.BytesIO()
    np.savez(buffer, intensity=np.ones((4, 8)), dark=np.zeros(8),
             flat=np.ones(8) * 2)
    files = {
        "low_file": ("low.npz", low_npz, "application/octet-stream"),
        "high_file": ("high.npz", buffer.getvalue(), "application/octet-stream"),
    }
    data = {
        "detector_spacing_mm": "0.5", "center_index": "63.5",
        "output_size": "64", "pixel_spacing_mm": "0.5",
        "slice_thickness_mm": "1.0",
        "materials": json.dumps({
            "names": ["a", "b"],
            "mass_attenuation_matrix_mm2_per_mg": MATRIX,
        }),
        "rois": json.dumps([{"name": "r", "x0": 0, "y0": 0, "x1": 4, "y1": 4}]),
    }
    response = client.post("/decompose", files=files, data=data)
    assert response.status_code == 422


def test_decompose_rejects_bad_materials_and_rois():
    response = _decompose_call(materials={
        "names": ["only"],
        "mass_attenuation_matrix_mm2_per_mg": MATRIX,
    })
    assert response.status_code == 422
    response = _decompose_call(materials={
        "names": ["a", "b"],
        "mass_attenuation_matrix_mm2_per_mg": [[1.0, 1.0], [1.0, 1.0001]],
    })
    assert response.status_code == 422
    response = _decompose_call(rois=[{"name": "r", "x0": 60, "y0": 0,
                                      "x1": 65, "y1": 4}])
    assert response.status_code == 422
    response = _decompose_call(slice_thickness_mm="-1")
    assert response.status_code == 422

