"""FastAPI application exposing the parallel-beam CT reconstruction API."""

from __future__ import annotations

import io
import json
import zipfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .io_utils import ValidationError, load_npz
from .preview import npy_bytes, render_png
from .reconstruct import supported_filters
from .service import decompose_upload, reconstruct_upload

app = FastAPI(title="Parallel-beam CT FBP reconstruction", version="1.1.0")


@app.exception_handler(ValidationError)
async def _validation_handler(_request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": str(exc)})


def _parse_json_field(raw: str, field: str):
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"form field {field!r} must be valid JSON: {exc}") from exc


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "filters": list(supported_filters())}


@app.post("/reconstruct")
async def reconstruct(
    file: UploadFile = File(..., description="NPZ with intensity, dark, flat"),
    detector_spacing_mm: float = Form(..., alias="detector_spacing_mm"),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
) -> Response:
    payload = await file.read()
    data = load_npz(payload)
    result = reconstruct_upload(
        data,
        {
            "detector_spacing_mm": detector_spacing_mm,
            "center_index": center_index,
            "output_size": output_size,
            "pixel_spacing_mm": pixel_spacing_mm,
            "filter": filter,
        },
    )

    metadata = {
        "parameters": result.params,
        "ranges": {
            "image_min": result.stats["image_min"],
            "image_max": result.stats["image_max"],
            "image_mean": result.stats["image_mean"],
            "sinogram_min": result.stats["sinogram_min"],
            "sinogram_max": result.stats["sinogram_max"],
        },
        "units": {
            "detector_spacing_mm": "millimeter",
            "pixel_spacing_mm": "millimeter",
            "image": "linear attenuation coefficient per millimeter",
        },
        "notes": "preview.png uses a min/max linear stretch for display only; reconstruction.npy is untouched float64 data.",
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("reconstruction.npy", npy_bytes(result.image))
        zf.writestr("preview.png", render_png(result.image))
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True))

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="reconstruction.zip"'},
    )


@app.post("/decompose")
async def decompose(
    low_file: UploadFile = File(..., description="Low-energy NPZ (intensity, dark, flat)"),
    high_file: UploadFile = File(..., description="High-energy NPZ, same shape as low"),
    detector_spacing_mm: float = Form(...),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
    materials: str = Form(..., description='JSON: {"names": [a, b], "mass_attenuation_matrix_mm2_per_mg": [[l0, l1], [h0, h1]]}'),
    slice_thickness_mm: float = Form(...),
    rois: str = Form(..., description='JSON list of {"name", "x0", "y0", "x1", "y1"}'),
) -> Response:
    low_data = load_npz(await low_file.read())
    high_data = load_npz(await high_file.read())

    materials_spec = _parse_json_field(materials, "materials")
    if not isinstance(materials_spec, dict):
        raise ValidationError("'materials' must be a JSON object")
    roi_spec = _parse_json_field(rois, "rois")

    result = decompose_upload(
        low_data,
        high_data,
        {
            "detector_spacing_mm": detector_spacing_mm,
            "center_index": center_index,
            "output_size": output_size,
            "pixel_spacing_mm": pixel_spacing_mm,
            "filter": filter,
        },
        materials_spec.get("names"),
        materials_spec.get("mass_attenuation_matrix_mm2_per_mg"),
        slice_thickness_mm,
        roi_spec,
    )

    names = result.materials["names"]
    report = {
        "parameters": {
            **result.params,
            "slice_thickness_mm": result.slice_thickness_mm,
        },
        "materials": {
            "names": names,
            "mass_attenuation_matrix_mm2_per_mg": result.materials["matrix"].tolist(),
            "matrix_condition_number": result.materials["condition_number"],
        },
        "rois": result.roi_results,
        "units": {
            "density": "mg/mm^3",
            "residual": "linear attenuation per millimeter (predicted - observed)",
            "mass": "milligram",
            "slice_thickness_mm": "millimeter",
        },
        "notes": "density_*.npy / residual_*.npy are untouched float64 data; "
                 "preview_*.png uses a min/max linear stretch for display only.",
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for j, name in enumerate(names):
            zf.writestr(f"density_{name}.npy", npy_bytes(result.densities[j]))
            zf.writestr(f"preview_{name}.png", render_png(result.densities[j]))
        zf.writestr("residual_low.npy", npy_bytes(result.residuals[0]))
        zf.writestr("residual_high.npy", npy_bytes(result.residuals[1]))
        zf.writestr("result.json", json.dumps(report, indent=2, sort_keys=True))

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="decomposition.zip"'},
    )

