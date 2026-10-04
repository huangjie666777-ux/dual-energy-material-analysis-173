"""Rectangular regions of interest and quantitative integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_utils import ValidationError

MAX_ROIS = 8


@dataclass(frozen=True)
class Roi:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def n_pixels(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


def parse_rois(spec, image_size: int) -> list[Roi]:
    """Validate ROI specs: [x0, x1) x [y0, y1), top-left inclusive."""
    if not isinstance(spec, (list, tuple)) or not (1 <= len(spec) <= MAX_ROIS):
        raise ValidationError(f"rois must be a list of 1 to {MAX_ROIS} rectangles")
    rois: list[Roi] = []
    seen: set[str] = set()
    for entry in spec:
        if not isinstance(entry, dict):
            raise ValidationError("each ROI must be an object with name, x0, y0, x1, y1")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("each ROI needs a non-empty string name")
        name = name.strip()
        if name in seen:
            raise ValidationError(f"duplicate ROI name {name!r}")
        seen.add(name)
        try:
            coords = [int(entry[k]) for k in ("x0", "y0", "x1", "y1")]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"ROI {name!r} needs integer x0, y0, x1, y1") from exc
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= image_size) or not (0 <= y0 < y1 <= image_size):
            raise ValidationError(
                f"ROI {name!r} ({x0},{y0})-({x1},{y1}) is empty or out of bounds "
                f"for a {image_size}x{image_size} image"
            )
        rois.append(Roi(name=name, x0=x0, y0=y0, x1=x1, y1=y1))
    return rois


def integrate_rois(
    rois: list[Roi],
    material_names: list[str],
    densities: np.ndarray,
    residuals: np.ndarray,
    pixel_spacing_mm: float,
    slice_thickness_mm: float,
) -> list[dict]:
    """Integrate material mass (mg) and mean per-energy residuals per ROI."""
    pixel_volume_mm3 = pixel_spacing_mm * pixel_spacing_mm * slice_thickness_mm
    results = []
    for roi in rois:
        window_d = densities[:, roi.y0:roi.y1, roi.x0:roi.x1]
        window_r = residuals[:, roi.y0:roi.y1, roi.x0:roi.x1]
        results.append({
            "name": roi.name,
            "pixels": [roi.x0, roi.y0, roi.x1, roi.y1],
            "n_pixels": roi.n_pixels,
            "mass_mg": {
                name: float(window_d[j].sum() * pixel_volume_mm3)
                for j, name in enumerate(material_names)
            },
            "mean_residual_per_mm": {
                "low": float(window_r[0].mean()),
                "high": float(window_r[1].mean()),
            },
        })
    return results

