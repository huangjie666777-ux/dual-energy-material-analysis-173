"""End-to-end reconstruction and decomposition pipelines shared by HTTP/CLI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import calibrate
from .decomposition import decompose, validate_materials
from .io_utils import LoadedData, ValidationError, validate_params
from .reconstruct import fbp
from .rois import integrate_rois, parse_rois


@dataclass(frozen=True)
class ReconstructionResult:
    image: np.ndarray
    line_integrals: np.ndarray
    params: dict
    stats: dict


@dataclass(frozen=True)
class DecompositionResult:
    densities: np.ndarray        # (2, n, n) mg/mm^3, rows follow material order
    residuals: np.ndarray        # (2, n, n) mm^-1, rows are low/high energy
    low_image: np.ndarray
    high_image: np.ndarray
    params: dict
    materials: dict
    slice_thickness_mm: float
    roi_results: list


def _reconstruct_with_params(data: LoadedData, clean: dict) -> tuple[np.ndarray, np.ndarray]:
    sinogram = calibrate(data)
    image = fbp(
        sinogram,
        detector_spacing=clean["detector_spacing_mm"],
        center_index=clean["center_index"],
        output_size=clean["output_size"],
        pixel_spacing=clean["pixel_spacing_mm"],
        filter_name=clean["filter"],
    )
    return sinogram, image


def reconstruct_upload(data: LoadedData, params: dict) -> ReconstructionResult:
    clean = validate_params(
        params["detector_spacing_mm"],
        params["center_index"],
        params["output_size"],
        params["pixel_spacing_mm"],
        params["filter"],
    )
    sinogram, image = _reconstruct_with_params(data, clean)
    stats = {
        "image_min": float(np.min(image)),
        "image_max": float(np.max(image)),
        "image_mean": float(np.mean(image)),
        "sinogram_min": float(np.min(sinogram)),
        "sinogram_max": float(np.max(sinogram)),
    }
    return ReconstructionResult(
        image=image,
        line_integrals=sinogram,
        params=clean,
        stats=stats,
    )


def validate_slice_thickness(value) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValidationError("slice_thickness_mm must be a finite positive number")
    return float(value)


def decompose_upload(
    low_data: LoadedData,
    high_data: LoadedData,
    params: dict,
    material_names,
    attenuation_matrix,
    slice_thickness_mm,
    roi_spec,
) -> DecompositionResult:
    """Reconstruct both energies and run NNLS decomposition + ROI metering."""
    if low_data.intensity.shape != high_data.intensity.shape:
        raise ValidationError(
            f"low/high sinograms must share shape, got "
            f"{low_data.intensity.shape} vs {high_data.intensity.shape}"
        )
    clean = validate_params(
        params["detector_spacing_mm"],
        params["center_index"],
        params["output_size"],
        params["pixel_spacing_mm"],
        params["filter"],
    )
    materials = validate_materials(material_names, attenuation_matrix)
    thickness = validate_slice_thickness(slice_thickness_mm)
    rois = parse_rois(roi_spec, clean["output_size"])

    _, low_image = _reconstruct_with_params(low_data, clean)
    _, high_image = _reconstruct_with_params(high_data, clean)

    densities, residuals = decompose(materials["matrix"], low_image, high_image)
    roi_results = integrate_rois(
        rois,
        materials["names"],
        densities,
        residuals,
        clean["pixel_spacing_mm"],
        thickness,
    )
    return DecompositionResult(
        densities=densities,
        residuals=residuals,
        low_image=low_image,
        high_image=high_image,
        params=clean,
        materials=materials,
        slice_thickness_mm=thickness,
        roi_results=roi_results,
    )

