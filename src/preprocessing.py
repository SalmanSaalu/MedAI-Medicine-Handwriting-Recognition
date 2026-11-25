"""Image preparation and candidate text-region detection for MedScript AI.

This module deliberately does NOT decide which regions are medicine names.
It returns candidate text crops for TrOCR and human review.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def _paper_background_value(gray: np.ndarray) -> int:
    """Estimate paper intensity robustly; the border may contain shadows."""
    # A median over the complete grayscale image is robust when handwriting
    # occupies a minority of pixels. Border-only estimates can be badly biased
    # by white scan margins or dark page edges.
    return int(np.median(gray))


def _rotate_image(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate without introducing a pure-white border on gray paper."""
    height, width = gray.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_width = int(height * sin_value + width * cos_value)
    new_height = int(height * cos_value + width * sin_value)

    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]

    background = _paper_background_value(gray)
    return cv2.warpAffine(
        gray,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )


def _projection_score(binary: np.ndarray, angle: float) -> float:
    """Score how concentrated foreground pixels are in horizontal bands."""
    height, width = binary.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    row_projection = np.count_nonzero(rotated, axis=1).astype(np.float64)
    smooth = np.convolve(row_projection, np.ones(5) / 5.0, mode="same")
    return float(np.sum(smooth**2))


def estimate_skew_angle(gray: np.ndarray, max_angle: float = 10.0) -> float:
    """Estimate small page skew. Returns 0 for nearly blank images."""
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    height, width = binary.shape
    if width > 1000:
        scale = 1000.0 / width
        binary = cv2.resize(
            binary,
            (1000, max(1, int(height * scale))),
            interpolation=cv2.INTER_NEAREST,
        )

    if np.count_nonzero(binary) < 50:
        return 0.0

    coarse_angles = np.arange(-max_angle, max_angle + 0.001, 0.5)
    coarse_scores = [_projection_score(binary, float(a)) for a in coarse_angles]
    best_coarse = float(coarse_angles[int(np.argmax(coarse_scores))])

    fine_angles = np.arange(best_coarse - 0.5, best_coarse + 0.501, 0.1)
    fine_scores = [_projection_score(binary, float(a)) for a in fine_angles]
    return float(fine_angles[int(np.argmax(fine_scores))])


def prepare_image(pil_image: Image.Image, max_dimension: int = 1800) -> np.ndarray:
    """Convert to grayscale, resize if required, and correct small skew.

    Avoids aggressive thresholding/denoising here because thin handwritten
    strokes may be damaged. Thresholding is applied separately for detection.
    """
    rgb = np.asarray(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape
    largest = max(height, width)
    if largest > max_dimension:
        scale = max_dimension / float(largest)
        gray = cv2.resize(
            gray,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    angle = estimate_skew_angle(gray)
    if abs(angle) >= 0.3:
        gray = _rotate_image(gray, angle)

    return gray


def _make_crop(
    gray: np.ndarray,
    binary: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    index: int,
) -> dict[str, Any] | None:
    image_height, image_width = gray.shape
    x_margin = max(8, int(image_width * 0.012))
    y_margin = max(6, int(image_height * 0.018))

    x1 = max(0, x - x_margin)
    y1 = max(0, y - y_margin)
    x2 = min(image_width, x + width + x_margin)
    y2 = min(image_height, y + height + y_margin)

    crop = gray[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 12:
        return None

    return {
        "index": index,
        "bbox": (x1, y1, x2, y2),
        "image": crop,
    }


def detect_horizontal_lines(gray: np.ndarray) -> list[dict[str, Any]]:
    """Detect horizontal text bands using an ink-row projection.

    This is the recommended first-pass mode for prescriptions where writing
    mostly follows horizontal rows. A detected band can include dosage text
    as well as a medicine name, so outputs must be reviewed.
    """
    if gray is None or gray.size == 0:
        return []

    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    image_height, image_width = binary.shape

    row_projection = np.count_nonzero(binary, axis=1).astype(np.float32)
    peak = float(row_projection.max()) if row_projection.size else 0.0
    if peak <= 0:
        return []

    # Smooth tiny vertical breaks. Keep this window small so adjacent rows
    # are less likely to merge.
    smooth = np.convolve(row_projection, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    threshold = max(1.0, peak * 0.025)
    active = (smooth >= threshold).astype(np.uint8)
    active = cv2.morphologyEx(
        active.reshape(-1, 1),
        cv2.MORPH_CLOSE,
        np.ones((3, 1), dtype=np.uint8),
    ).flatten().astype(bool)

    active_rows = np.flatnonzero(active)
    if active_rows.size == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = previous = int(active_rows[0])
    for row in active_rows[1:]:
        row = int(row)
        if row - previous > 1:
            groups.append((start, previous + 1))
            start = row
        previous = row
    groups.append((start, previous + 1))

    # Merge only one- or two-pixel interruptions.
    merged: list[tuple[int, int]] = []
    for top, bottom in groups:
        if merged and top - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], bottom)
        else:
            merged.append((top, bottom))

    regions: list[dict[str, Any]] = []
    min_band_height = max(7, int(image_height * 0.008))
    min_band_width = max(12, int(image_width * 0.015))

    for top, bottom in merged:
        if bottom - top < min_band_height:
            continue

        ink_region = binary[top:bottom, :]
        points = cv2.findNonZero(ink_region)
        if points is None:
            continue

        x, local_y, width, band_height = cv2.boundingRect(points)
        if width < min_band_width or band_height < min_band_height:
            continue

        # Use the actual ink bounds rather than the full page width.
        region = _make_crop(
            gray, binary, x, top + local_y, width, band_height, len(regions) + 1
        )
        if region is not None:
            regions.append(region)

    return regions


def detect_connected_regions(gray: np.ndarray) -> list[dict[str, Any]]:
    """Experimental connected-region detector for less regular layouts.

    It groups nearby ink components with a small horizontal dilation. It can
    split cursive words or join neighbouring writing; inspect the crop preview.
    This is not a replacement for a trained full-page text detector.
    """
    if gray is None or gray.size == 0:
        return []

    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    image_height, image_width = binary.shape

    kernel_width = max(8, min(32, int(image_width * 0.018)))
    kernel_height = max(2, min(5, int(image_height * 0.008)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, kernel_height)
    )
    grouped = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(
        grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: list[tuple[int, int, int, int]] = []
    min_width = max(10, int(image_width * 0.008))
    min_height = max(7, int(image_height * 0.012))
    min_area = max(45, int(image_height * image_width * 0.00008))

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < min_width or height < min_height or width * height < min_area:
            continue
        # Ignore an accidental near-full-page component.
        if width > image_width * 0.96 and height > image_height * 0.80:
            continue
        boxes.append((x, y, width, height))

    # Reading order: top-to-bottom, then left-to-right. Nearby boxes with
    # slightly different baselines may not be perfectly ordered; review them.
    boxes.sort(key=lambda box: (box[1], box[0]))

    regions: list[dict[str, Any]] = []
    for x, y, width, height in boxes:
        region = _make_crop(gray, binary, x, y, width, height, len(regions) + 1)
        if region is not None:
            regions.append(region)

    return regions


def detect_text_regions(
    gray: np.ndarray, mode: str = "horizontal"
) -> list[dict[str, Any]]:
    """Select one of the available candidate-region detectors."""
    if mode == "horizontal":
        return detect_horizontal_lines(gray)
    if mode == "connected":
        return detect_connected_regions(gray)
    raise ValueError("mode must be 'horizontal' or 'connected'")


def detect_text_lines(gray: np.ndarray) -> list[np.ndarray]:
    """Backward-compatible helper returning only horizontal crop arrays."""
    return [region["image"] for region in detect_horizontal_lines(gray)]
