"""
lib/image_preprocessing.py
---------------------------
Shared 2D image preprocessing for EM-conditioned 3D organelle generation.

IMPORTANT: Both the extraction pipeline (scripts/extract_instances_from_nifti.py)
and the inference pipeline (inference.py, raw-input mode) import from this
module.  This is the single source of truth for prepare_slice() and
normalize_image().  Do not copy these functions elsewhere.
"""
from __future__ import annotations

import numpy as np
import skimage.transform


def prepare_slice(
    arr_2d: np.ndarray,
    sx: float,
    sy: float,
    proj_size: int,
) -> np.ndarray:
    """Resample a 2D image to isotropic pixel size, letterbox-pad to square,
    then resize to (proj_size, proj_size).

    Parameters
    ----------
    arr_2d:
        (H, W) float array.
        Axis 0 (rows / height) has physical spacing *sx*.
        Axis 1 (cols / width)  has physical spacing *sy*.
        Only the ratio sx/sy matters for output geometry.
    sx:
        Physical voxel size along axis 0 (same units as sy; ratio matters).
    sy:
        Physical voxel size along axis 1.
    proj_size:
        Side length in pixels of the square output image.

    Returns
    -------
    (proj_size, proj_size) float32 array.
    Pixel values are **not** normalised here; call normalize_image() separately.

    Raises
    ------
    ValueError
        If arr_2d is not 2-D.
    """
    if arr_2d.ndim != 2:
        raise ValueError(
            f"prepare_slice expects a 2-D array, got shape {arr_2d.shape}"
        )
    if sx <= 0 or sy <= 0:
        raise ValueError(f"Voxel spacings must be positive; got sx={sx}, sy={sy}")

    scale = min(sx, sy)
    out_h = max(1, int(round(arr_2d.shape[0] * sx / scale)))
    out_w = max(1, int(round(arr_2d.shape[1] * sy / scale)))

    # Step 1: Resample to isotropic (physical-unit-consistent) grid
    iso = skimage.transform.resize(
        arr_2d,
        (out_h, out_w),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    # Step 2: Letterbox-pad to square (background = 0)
    side = max(out_h, out_w)
    canvas = np.zeros((side, side), dtype=np.float32)
    pad_h = (side - out_h) // 2
    pad_w = (side - out_w) // 2
    canvas[pad_h : pad_h + out_h, pad_w : pad_w + out_w] = iso

    # Step 3: Resize to target resolution
    return skimage.transform.resize(
        canvas,
        (proj_size, proj_size),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def normalize_image(arr: np.ndarray) -> np.ndarray:
    """Per-image min-max normalization to [0, 1].

    A constant image (hi == lo, including constant-nonzero) is mapped to
    all zeros rather than being left at its original value.  This is the
    correct handling for empty edge slices that barely passed the minimum-
    area filter.

    Parameters
    ----------
    arr:
        Float array of any shape.

    Returns
    -------
    float32 array of the same shape, values in [0, 1].
    """
    lo = float(arr.min())
    hi = float(arr.max())
    if hi > lo:
        return ((arr - lo) / (hi - lo)).astype(np.float32)
    # Constant image (including constant-nonzero) → zeros
    return np.zeros_like(arr, dtype=np.float32)
