#!/usr/bin/env python3
"""Batch-segment peanut group photographs into traceable instance crops.

The default segmenter is a deterministic classical pipeline based on
light-background separation, distance-transform peaks, and watershed.  The
optional SAM path keeps those watershed centers as prompts and uses Segment
Anything to refine touching/merged candidates; the classical pipeline remains
the source of the initial instance count and point prompts.

Examples
--------
python process_peanuts.py --input-dir ./raw_peanut_groups \
    --output-dir ./peanut_dataset_processed --image-size 224 --device auto

Preview only three source images:
python process_peanuts.py --input-dir ./raw_peanut_groups \
    --output-dir ./peanut_dataset_processed --only GD1(3).jpg,HN3(3).jpg,YN4(3).jpg
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import re
import shutil
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - optional fallback
    ndi = None

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed as sk_watershed
except Exception:  # pragma: no cover - optional fallback
    peak_local_max = None
    sk_watershed = None


LOG = logging.getLogger("process_peanuts")
SEED = 20260916
_PROMPTABLE_SAM_CACHE: dict[tuple[str, str, str], object] = {}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

REGION_NAMES = {
    "GD": "Guangdong",
    "YN": "Yunnan",
    "HN": "Henan",
    "SD": "Shandong",
    "DB": "Dongbei / Northeast China",
    "HLJ": "Heilongjiang",
    "JL": "Jilin",
    "LN": "Liaoning",
}

MANIFEST_FIELDS = [
    "peanut_id",
    "original_filename",
    "source_image_id",
    "parent_photo_id",
    "source_number",
    "group_label",
    "region_code",
    "peanut_number",
    "transparent_path",
    "cnn_ready_path",
    "centroid_x",
    "centroid_y",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "mask_area_pixels",
    "original_image_width",
    "original_image_height",
    "partial_at_image_edge",
    "segmentation_confidence",
    "qc_status",
    "qc_flags",
    "qc_path",
    "segmenter",
]

QC_FIELDS = MANIFEST_FIELDS + ["source_error"]
ERROR_FIELDS = ["original_filename", "source_image_id", "error_type", "error_message"]


@dataclass(frozen=True)
class SourceInfo:
    original_filename: str
    source_image_id: str
    source_number: str
    group_label: str
    region_code: str


@dataclass(frozen=True)
class SeedPoint:
    """A watershed-derived center used only as a SAM point prompt."""

    x: float
    y: float


@dataclass
class Detection:
    label: int
    mask: np.ndarray
    centroid_x: float
    centroid_y: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    area: int
    confidence: float
    flags: list[str]


def set_deterministic_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def parse_source_info(path: Path) -> SourceInfo:
    """Parse labels without discarding the group number or original stem.

    Expected examples include GD1.jpg, GD1(3).jpg, and filenames using full
    width parentheses such as HLJ2(2）.jpg.  Unknown/nonconforming names keep
    their full stem as group_label and source_image_id.
    """
    stem = path.stem
    match = re.fullmatch(
        r"(?P<region>[A-Za-z]+)(?P<group_num>\d+)(?:[\(\uFF08](?P<source_num>\d+)[\)\uFF09])?",
        stem,
    )
    if not match:
        return SourceInfo(
            original_filename=path.name,
            source_image_id=stem,
            source_number="",
            group_label=stem,
            region_code="",
        )
    region = match.group("region").upper()
    group_label = f"{region}{match.group('group_num')}"
    source_number = match.group("source_num") or ""
    return SourceInfo(
        original_filename=path.name,
        source_image_id=stem,
        source_number=source_number,
        group_label=group_label,
        region_code=region,
    )


def source_number_tag(source_number: str) -> str:
    return source_number.zfill(3) if source_number else "000"


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # OpenCV's Windows path handling can fail on otherwise valid filenames
    # containing full-width Unicode punctuation. Pillow provides a reliable
    # fallback for those paths without changing the decoded RGB pixels.
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except Exception as exc:
        raise ValueError(f"Could not decode image: {path}") from exc


def border_pixels(arr: np.ndarray, fraction: float = 0.025) -> np.ndarray:
    h, w = arr.shape[:2]
    n = max(4, int(round(min(h, w) * fraction)))
    return np.concatenate(
        [arr[:n, :, :].reshape(-1, arr.shape[2]),
         arr[-n:, :, :].reshape(-1, arr.shape[2]),
         arr[:, :n, :].reshape(-1, arr.shape[2]),
         arr[:, -n:, :].reshape(-1, arr.shape[2])],
        axis=0,
    )


def robust_threshold(values: np.ndarray, floor: float, multiplier: float = 4.0) -> float:
    values = values.astype(np.float32)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return max(floor, med + multiplier * max(mad, 1.0))


def light_background_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a cleaned foreground mask for brown/tan objects on light paper."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    border = border_pixels(lab)
    bg = np.median(border, axis=0)

    l_drop = np.maximum(bg[0] - lab[:, :, 0], 0.0)
    chroma = np.sqrt((lab[:, :, 1] - bg[1]) ** 2 + (lab[:, :, 2] - bg[2]) ** 2)
    sat = hsv[:, :, 1]

    border_l_drop = np.maximum(bg[0] - border[:, 0], 0.0)
    border_chroma = np.sqrt((border[:, 1] - bg[1]) ** 2 + (border[:, 2] - bg[2]) ** 2)
    l_threshold = robust_threshold(border_l_drop, floor=4.0)
    chroma_threshold = robust_threshold(border_chroma, floor=5.0)

    # The photographs have soft grey cast-shadows between pods.  A luminance-
    # only threshold therefore joins the entire pile to the paper.  Prefer
    # chroma/saturation to identify the brown/tan pods, and only accept a very
    # dark pixel when it also has non-trivial colour.  The explicit floors are
    # intentional: border JPEG noise can otherwise make robust thresholds too
    # permissive when the border is nearly uniform.
    mask = (
        (chroma > max(10.0, chroma_threshold * 1.8))
        | (sat > 38.0)
        | ((l_drop > max(38.0, l_threshold * 5.0)) & (sat > 14.0))
    )
    mask = (mask.astype(np.uint8) * 255)

    # Keep morphology local.  Large closing kernels bridge the small paper
    # gaps between adjacent peanuts and defeat instance watershed.
    k_open = 3
    k_close = 3
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Remove isolated noise but retain edge-touching and small uncertain
    # objects for QC instead of silently deleting them.
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_component_area = max(250, int(rgb.shape[0] * rgb.shape[1] * 0.00001))
    keep = np.zeros(nlabels, dtype=np.uint8)
    keep[0] = 0
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_component_area).astype(np.uint8)
    mask = (keep[labels] * 255).astype(np.uint8)
    return mask > 0


def fallback_peaks(distance: np.ndarray, foreground: np.ndarray, min_distance: int) -> np.ndarray:
    """OpenCV local-max fallback when scikit-image is unavailable."""
    size = int(min_distance * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(distance.astype(np.float32), kernel)
    positive = distance[foreground]
    if positive.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    threshold = max(6.0, float(np.percentile(positive, 50)) * 0.25)
    peaks = (distance >= dilated - 1e-3) & (distance >= threshold) & foreground
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), 8)
    coords = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0:
            x, y = centroids[i]
            coords.append((int(round(y)), int(round(x))))
    return np.asarray(coords, dtype=np.int32)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask.astype(np.uint8) * 255)
    inv = cv2.bitwise_not(binary)
    flood = inv.copy()
    flood_mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)
    outside = flood == 128
    holes = (inv > 0) & ~outside
    return mask | holes


def fill_safe_small_holes(mask: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Fill tiny background pinholes without absorbing another object.

    A plain binary hole-fill is unsafe after watershed: a neighboring peanut
    can sit inside the closed boundary of a watershed cell and would then be
    copied into the target mask.  Only fill a small hole when the original
    foreground mask is empty throughout that hole.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return mask
    # Restrict the flood fill to a small padded window.  Running it over the
    # full 4000x3000 source for every watershed cell is unnecessarily costly.
    pad = 2
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(mask.shape[1], int(xs.max()) + pad + 1)
    local_mask = mask[y1:y2, x1:x2]
    local_foreground = foreground[y1:y2, x1:x2]
    filled = fill_holes(local_mask)
    candidates = (filled & ~local_mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    result = mask.copy()
    max_hole_area = max(100, int(max(1, local_mask.sum()) * 0.01))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > max_hole_area:
            continue
        hole = labels == i
        if not np.any(local_foreground & hole):
            result[y1:y2, x1:x2][hole] = True
    return result


def mask_touches_neighbor(mask: np.ndarray, foreground: np.ndarray, all_labels: np.ndarray, label: int) -> bool:
    kernel = np.ones((3, 3), np.uint8)
    neighbor = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    other_label = (all_labels > 0) & (all_labels != label) & neighbor
    if not np.any(other_label):
        return False
    # A common boundary with foreground on both sides indicates contact in
    # the source image rather than merely a one-pixel watershed border.
    return bool(np.any(foreground & neighbor & ~mask))


def suspicious_holes(mask: np.ndarray) -> bool:
    binary = (mask.astype(np.uint8) * 255)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    return bool(np.any(hierarchy[0, :, 3] >= 0))


def choose_min_distance(distance: np.ndarray, foreground: np.ndarray) -> int:
    positive = distance[foreground]
    if positive.size == 0:
        return 20
    p80 = float(np.percentile(positive, 80))
    # Peanut centers in the repository photos can be close together.  A
    # conservative spacing gives watershed a chance to place one seed in
    # each lobe while still suppressing duplicate maxima inside one peanut.
    return max(16, min(90, int(round(p80 * 0.42))))


def detect_seed_points(rgb: np.ndarray) -> tuple[list[SeedPoint], np.ndarray]:
    """Find one distance-transform/local-maximum seed per candidate peanut.

    This is deliberately not a watershed instance detector.  The foreground
    mask, distance transform, and local maxima provide only point prompts for
    SAM.  No watershed label image or watershed region boundary leaves this
    function.
    """
    foreground = light_background_mask(rgb)
    if not np.any(foreground):
        return [], foreground

    distance = cv2.distanceTransform((foreground.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    min_distance = choose_min_distance(distance, foreground)
    if peak_local_max is not None:
        threshold_abs = max(6.0, float(np.percentile(distance[foreground], 50)) * 0.25)
        coords = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            labels=foreground.astype(np.uint8),
            exclude_border=False,
        )
    else:
        coords = np.empty((0, 2), dtype=np.int32)
    if coords.size == 0:
        coords = fallback_peaks(distance, foreground, min_distance)

    seeds = [SeedPoint(x=float(x), y=float(y)) for y, x in np.asarray(coords).reshape(-1, 2)]
    return seeds, foreground


def detect_classical_instances(rgb: np.ndarray) -> tuple[list[Detection], np.ndarray, str]:
    foreground = light_background_mask(rgb)
    if not np.any(foreground):
        return [], foreground, "classical"

    distance = cv2.distanceTransform((foreground.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    min_distance = choose_min_distance(distance, foreground)
    if peak_local_max is not None and sk_watershed is not None:
        threshold_abs = max(6.0, float(np.percentile(distance[foreground], 50)) * 0.25)
        coords = peak_local_max(
            distance,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
            labels=foreground.astype(np.uint8),
            exclude_border=False,
        )
        markers = np.zeros(distance.shape, dtype=np.int32)
        for marker_id, (y, x) in enumerate(coords, start=1):
            markers[int(y), int(x)] = marker_id
        if markers.max() == 0:
            coords = fallback_peaks(distance, foreground, min_distance)
            markers = np.zeros(distance.shape, dtype=np.int32)
            for marker_id, (y, x) in enumerate(coords, start=1):
                markers[int(y), int(x)] = marker_id
        labels = sk_watershed(-distance, markers, mask=foreground)
    else:
        coords = fallback_peaks(distance, foreground, min_distance)
        markers = np.zeros(distance.shape, dtype=np.int32)
        for marker_id, (y, x) in enumerate(coords, start=1):
            markers[int(y), int(x)] = marker_id
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        labels = cv2.watershed(bgr, markers)
        labels[~foreground] = 0

    n_labels = int(labels.max())
    if n_labels <= 0:
        return [], foreground, "classical"

    areas = np.asarray([(labels == i).sum() for i in range(1, n_labels + 1)], dtype=np.int64)
    positive_areas = areas[areas > 0]
    median_area = float(np.median(positive_areas)) if positive_areas.size else 1.0
    min_keep = max(500, int(median_area * 0.04))
    detections: list[Detection] = []
    for label in range(1, n_labels + 1):
        raw_mask = labels == label
        area = int(raw_mask.sum())
        if area < min_keep:
            continue
        mask = fill_safe_small_holes(raw_mask, foreground)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        m = cv2.moments(mask.astype(np.uint8))
        cx = float(m["m10"] / m["m00"]) if m["m00"] else float(xs.mean())
        cy = float(m["m01"] / m["m00"]) if m["m00"] else float(ys.mean())
        flags: list[str] = []
        h, w = mask.shape
        partial = x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1
        if partial:
            flags.append("partial_at_image_edge")
        if area < max(800, int(median_area * 0.25)):
            flags.append("extremely_small_mask")
        if area > max(1000, int(median_area * 2.5)):
            flags.append("extremely_large_mask")
        if suspicious_holes(raw_mask):
            flags.append("suspicious_holes")
        if mask_touches_neighbor(raw_mask, foreground, labels, label):
            flags.append("touching_another_peanut")
        fill_ratio = area / max(1, int((x2 - x1) * (y2 - y1)))
        if fill_ratio < 0.18 or fill_ratio > 0.98:
            flags.append("possible_merged_mask" if fill_ratio > 0.98 else "low_shape_confidence")
        contrast = float(np.mean(distance[mask])) / max(1.0, float(np.max(distance)))
        confidence = float(np.clip(0.45 + 0.45 * contrast + 0.10 * min(fill_ratio, 1.0), 0.0, 1.0))
        if confidence < 0.55:
            flags.append("low_confidence_segmentation")
        detections.append(
            Detection(
                label=label,
                mask=mask,
                centroid_x=cx,
                centroid_y=cy,
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                area=int(mask.sum()),
                confidence=confidence,
                flags=sorted(set(flags)),
            )
        )

    return detections, foreground, "classical"


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / max(1, union))


def _segment_seed_points_with_promptable_sam(
    rgb: np.ndarray,
    seeds: list[SeedPoint],
    checkpoint: str,
    device: str,
    model_family: str = "sam",
) -> list[Detection]:
    """Create final instances from one positive SAM point per seed.

    Watershed is intentionally absent from this function.  Each call passes
    exactly one positive point and no box or negative points.  The selected
    SAM mask is the only mask that can become a final Detection.
    """
    LOG.info("SAM seed points detected: %d", len(seeds))
    if not seeds:
        LOG.info("SAM masks after IoU deduplication: 0")
        return []
    if not checkpoint:
        raise ValueError("a SAM segmenter requires --sam-checkpoint")
    try:
        import torch
        if model_family == "mobile_sam":
            from mobile_sam import SamPredictor, sam_model_registry
            model_type = "vit_t"
        else:
            from segment_anything import SamPredictor, sam_model_registry
            model_type = "vit_b"
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            f"{model_family} requested but its optional dependencies are unavailable. "
            "Install torch and the corresponding SAM package, then retry with a valid checkpoint."
        ) from exc

    requested_device = device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        LOG.warning("CUDA requested but unavailable; using CPU")
        requested_device = "cpu"

    cache_key = (model_family, str(Path(checkpoint).resolve()), requested_device)
    predictor = _PROMPTABLE_SAM_CACHE.get(cache_key)
    if predictor is None:
        model = sam_model_registry[model_type](checkpoint=checkpoint)
        model.to(device=requested_device)
        model.eval()
        predictor = SamPredictor(model)
        _PROMPTABLE_SAM_CACHE[cache_key] = predictor
    predictor.set_image(rgb)

    candidates: list[tuple[float, np.ndarray, SeedPoint]] = []
    for seed in seeds:
        point_coords = np.asarray([[seed.x, seed.y]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int32)
        # Exactly one seed point is passed to each SAM call. Do not add boxes,
        # negative points, watershed labels, or neighboring centers here.
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best_index = int(np.argmax(np.asarray(scores)))
        best_mask = np.asarray(masks[best_index], dtype=bool)
        if np.any(best_mask):
            candidates.append((float(scores[best_index]), best_mask, seed))

    # Greedy score-ordered suppression removes duplicate predictions of the
    # same object only.  Non-overlapping masks are never merged together.
    kept: list[tuple[float, np.ndarray, SeedPoint]] = []
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(mask_iou(candidate[1], existing[1]) > 0.7 for existing in kept):
            continue
        kept.append(candidate)
    LOG.info("SAM masks after IoU deduplication: %d", len(kept))

    detections: list[Detection] = []
    for label, (score, mask, _seed) in enumerate(kept, start=1):
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        moments = cv2.moments(mask.astype(np.uint8))
        detections.append(
            Detection(
                label=label,
                mask=mask,
                centroid_x=float(moments["m10"] / moments["m00"]) if moments["m00"] else float(xs.mean()),
                centroid_y=float(moments["m01"] / moments["m00"]) if moments["m00"] else float(ys.mean()),
                bbox_x1=int(xs.min()),
                bbox_y1=int(ys.min()),
                bbox_x2=int(xs.max()) + 1,
                bbox_y2=int(ys.max()) + 1,
                area=int(mask.sum()),
                confidence=float(score),
                flags=[],
            )
        )
    return detections


def detect_instances(
    rgb: np.ndarray,
    segmenter: str = "auto",
    sam_checkpoint: str = "",
    device: str = "auto",
) -> tuple[list[Detection], np.ndarray, str]:
    if segmenter in {"sam", "mobile_sam"}:
        # In SAM mode, classical processing supplies only seed coordinates.
        # Its watershed labels and region boundaries are discarded here.
        seeds, foreground = detect_seed_points(rgb)
        detections = _segment_seed_points_with_promptable_sam(
            rgb,
            seeds,
            sam_checkpoint,
            device,
            model_family=segmenter,
        )
        used = f"distance-seeds+{segmenter}"
        return detections, foreground, used

    if segmenter == "auto" and sam_checkpoint:
        try:
            seeds, foreground = detect_seed_points(rgb)
            detections = _segment_seed_points_with_promptable_sam(
                rgb, seeds, sam_checkpoint, device, model_family="sam"
            )
            used = "distance-seeds+sam"
            return detections, foreground, used
        except Exception as exc:  # optional path must not break auto fallback
            LOG.warning("SAM unavailable; using classical segmentation: %s", exc)
    return detect_classical_instances(rgb)


def sort_visual_order(detections: list[Detection]) -> list[Detection]:
    if not detections:
        return detections
    median_height = float(np.median([d.bbox_y2 - d.bbox_y1 for d in detections]))
    row_tolerance = max(35.0, median_height * 0.55)
    ordered = sorted(detections, key=lambda d: (d.centroid_y, d.centroid_x))
    rows: list[list[Detection]] = []
    row_centers: list[float] = []
    for det in ordered:
        candidates = [i for i, cy in enumerate(row_centers) if abs(det.centroid_y - cy) <= row_tolerance]
        if not candidates:
            rows.append([det])
            row_centers.append(det.centroid_y)
        else:
            i = min(candidates, key=lambda j: abs(det.centroid_y - row_centers[j]))
            rows[i].append(det)
            row_centers[i] = float(np.mean([d.centroid_y for d in rows[i]]))
    row_order = sorted(range(len(rows)), key=lambda i: row_centers[i])
    result: list[Detection] = []
    for i in row_order:
        result.extend(sorted(rows[i], key=lambda d: d.centroid_x))
    return result


def alpha_from_mask(mask: np.ndarray) -> np.ndarray:
    hard = (mask.astype(np.uint8) * 255)
    # A one-pixel Gaussian transition gives real antialiasing while retaining
    # fully transparent background pixels.
    alpha = cv2.GaussianBlur(hard, (0, 0), sigmaX=0.65, sigmaY=0.65)
    alpha[alpha < 8] = 0
    return alpha.astype(np.uint8)


def crop_with_margin(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop exactly to this one mask's tight bounding box.

    The historical function name is retained for callers, but no margin or
    neighboring/unioned region is used in final crop generation.
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("empty mask")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return rgb[y1:y2, x1:x2].copy(), alpha_from_mask(mask[y1:y2, x1:x2])


def save_rgba_and_cnn(rgb_crop: np.ndarray, alpha_crop: np.ndarray, transparent_path: Path, cnn_path: Path, image_size: int) -> None:
    transparent_path.parent.mkdir(parents=True, exist_ok=True)
    cnn_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.dstack([rgb_crop, alpha_crop]).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(transparent_path, format="PNG")

    rgba_image = Image.fromarray(rgba, mode="RGBA")
    scale = min(image_size / rgba_image.width, image_size / rgba_image.height)
    new_size = (max(1, int(round(rgba_image.width * scale))), max(1, int(round(rgba_image.height * scale))))
    resized = rgba_image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    canvas.paste(resized, ((image_size - new_size[0]) // 2, (image_size - new_size[1]) // 2), resized)
    canvas.save(cnn_path, format="PNG")


def draw_preview(rgb: np.ndarray, detections: Sequence[Detection], output_path: Path) -> None:
    image = rgb.copy()
    for number, det in enumerate(detections, start=1):
        contour_mask = (det.mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = (40, 220, 80) if not det.flags else (255, 150, 30)
        cv2.drawContours(image, contours, -1, color, 3)
        cx, cy = int(round(det.centroid_x)), int(round(det.centroid_y))
        cv2.drawMarker(image, (cx, cy), (30, 30, 220), cv2.MARKER_CROSS, 24, 3)
        label = f"p{number:03d}"
        cv2.putText(image, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(image, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(output_path, format="JPEG", quality=92)


def draw_sam_debug_overlay(rgb: np.ndarray, detections: Sequence[Detection], output_path: Path) -> None:
    """Save one differently colored final SAM boundary per deduplicated mask."""
    image = rgb.copy()
    for number, det in enumerate(detections, start=1):
        hue = (number * 137) % 180
        color = tuple(int(v) for v in cv2.cvtColor(
            np.uint8([[[hue, 220, 245]]]), cv2.COLOR_HSV2RGB
        )[0, 0])
        contours, _ = cv2.findContours(
            det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(image, contours, -1, color, 4)
        cx, cy = int(round(det.centroid_x)), int(round(det.centroid_y))
        cv2.drawMarker(image, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(image, f"p{number:03d}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, f"p{number:03d}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(output_path, format="JPEG", quality=94)


def read_csv_rows(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, fieldnames=None))


def atomic_write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_done_source_names(output_dir: Path, manifest_rows: Sequence[dict[str, str]]) -> set[str]:
    done: set[str] = set()
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        by_source.setdefault(row.get("original_filename", ""), []).append(row)
    for source_name, rows in by_source.items():
        preview = output_dir / "previews" / f"{Path(source_name).stem}_annotated.jpg"
        if not preview.exists() or not rows:
            continue
        if all((output_dir / row.get("transparent_path", "__missing__")).exists() and
               (output_dir / row.get("cnn_ready_path", "__missing__")).exists() for row in rows):
            done.add(source_name)
    return done


def row_for_detection(
    info: SourceInfo,
    det: Detection,
    peanut_number: int,
    transparent_path: Path,
    cnn_path: Path,
    width: int,
    height: int,
    qc_path: Path | None,
    segmenter: str,
) -> dict[str, object]:
    tag = source_number_tag(info.source_number)
    peanut_id = f"{info.group_label}_{tag}_p{peanut_number:03d}"
    partial = "partial_at_image_edge" in det.flags
    if partial:
        status = "partial"
    elif "possible_merged_mask" in det.flags or "extremely_large_mask" in det.flags:
        status = "merged"
    elif det.flags:
        status = "uncertain"
    else:
        status = "clean"
    return {
        "peanut_id": peanut_id,
        "original_filename": info.original_filename,
        "source_image_id": info.source_image_id,
        "parent_photo_id": info.source_image_id,
        "source_number": info.source_number,
        "group_label": info.group_label,
        "region_code": info.region_code,
        "peanut_number": peanut_number,
        "transparent_path": transparent_path.relative_to(transparent_path.parents[1]).as_posix() if False else "",
        "cnn_ready_path": "",
        "centroid_x": f"{det.centroid_x:.3f}",
        "centroid_y": f"{det.centroid_y:.3f}",
        "bbox_x1": det.bbox_x1,
        "bbox_y1": det.bbox_y1,
        "bbox_x2": det.bbox_x2,
        "bbox_y2": det.bbox_y2,
        "mask_area_pixels": det.area,
        "original_image_width": width,
        "original_image_height": height,
        "partial_at_image_edge": str(partial).lower(),
        "segmentation_confidence": f"{det.confidence:.4f}",
        "qc_status": status,
        "qc_flags": ";".join(det.flags),
        "qc_path": qc_path.as_posix() if qc_path else "",
        "segmenter": segmenter,
    }


def validate_outputs(output_dir: Path, manifest_rows: Sequence[dict[str, str]], source_paths: Sequence[Path], errors: Sequence[dict[str, str]], image_size: int) -> dict[str, object]:
    ids = [r.get("peanut_id", "") for r in manifest_rows]
    duplicate_ids = sorted({x for x in ids if x and ids.count(x) > 1})
    missing_paths: list[str] = []
    invalid_transparent: list[str] = []
    invalid_cnn: list[str] = []
    transparent_basenames: set[str] = set()
    cnn_basenames: set[str] = set()
    for row in manifest_rows:
        tp = output_dir / row.get("transparent_path", "__missing__")
        cp = output_dir / row.get("cnn_ready_path", "__missing__")
        if not tp.exists() or not cp.exists():
            missing_paths.append(row.get("peanut_id", ""))
            continue
        transparent_basenames.add(tp.name)
        cnn_basenames.add(cp.name)
        try:
            with Image.open(tp) as im:
                if im.mode != "RGBA":
                    invalid_transparent.append(f"{tp}: mode={im.mode}")
                alpha = np.asarray(im.getchannel("A"))
                if not np.any(alpha == 0):
                    invalid_transparent.append(f"{tp}: no alpha=0 pixels")
        except Exception as exc:
            invalid_transparent.append(f"{tp}: {exc}")
        try:
            with Image.open(cp) as im:
                if im.size != (image_size, image_size) or im.mode != "RGB":
                    invalid_cnn.append(f"{cp}: size={im.size} mode={im.mode}")
        except Exception as exc:
            invalid_cnn.append(f"{cp}: {exc}")
    missing_pairs = sorted(transparent_basenames ^ cnn_basenames)
    source_names_with_detections = {r.get("original_filename", "") for r in manifest_rows}
    source_names_with_errors = {r.get("original_filename", "") for r in errors}
    sources_without_result = [p.name for p in source_paths if p.name not in source_names_with_detections and p.name not in source_names_with_errors]
    checks = {
        "duplicate_peanut_ids": duplicate_ids,
        "missing_paths": missing_paths,
        "invalid_transparent": invalid_transparent,
        "invalid_cnn": invalid_cnn,
        "missing_pairs": missing_pairs,
        "sources_without_result": sources_without_result,
        "ok": not any([duplicate_ids, missing_paths, invalid_transparent, invalid_cnn, missing_pairs, sources_without_result]),
    }
    atomic_write_csv(output_dir / "validation.csv", ["check", "value"], [{"check": k, "value": str(v)} for k, v in checks.items()])
    return checks


def make_readme() -> str:
    return """# Peanut instance dataset

This archive was generated by `process_peanuts.py`. The canonical files are
the RGBA images in `transparent/`; the `cnn_ready_224/` images are proportional
letterboxed RGB derivatives on a pure white canvas. The CSV manifest preserves
the original group-photo filename and the parsed group label, source number,
and region code for traceability. `parent_photo_id` is the full source-photo
stem (for example `GD1(3)`), so all crops from one photograph can be kept in
the same train/validation split.

Rows marked `partial`, `uncertain`, or `merged` in `peanuts_manifest.csv` are
not silently discarded. Review their `qc_flags` and the matching files under
`qc/` before including them in a clean training split.
"""


def create_archives(output_dir: Path) -> list[Path]:
    manifest = output_dir / "peanuts_manifest.csv"
    readme = output_dir / "README.md"
    readme.write_text(make_readme(), encoding="utf-8")
    archives: list[Path] = []

    def make_zip(name: str, members: Iterable[Path]) -> Path:
        target = output_dir / name
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for member in members:
                if member.is_file():
                    zf.write(member, member.relative_to(output_dir).as_posix())
        archives.append(target)
        return target

    make_zip("peanut_transparent_github.zip", [manifest, readme, *((output_dir / "transparent").rglob("*.png") if (output_dir / "transparent").exists() else [])])
    make_zip("peanut_cnn_ready_224.zip", [manifest, *((output_dir / "cnn_ready_224").rglob("*.png") if (output_dir / "cnn_ready_224").exists() else [])])
    qc_members = [output_dir / "qc_manifest.csv", output_dir / "source_errors.csv", output_dir / "previews"]
    qc_files: list[Path] = [p for p in qc_members if p.is_file()]
    qc_files.extend(p for root in [output_dir / "qc", output_dir / "previews"] if root.exists() for p in root.rglob("*"))
    make_zip("peanut_qc.zip", qc_files)
    return archives


def process(args: argparse.Namespace) -> int:
    set_deterministic_seed()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for rel in ["transparent", "cnn_ready_224", "qc/partial", "qc/uncertain", "qc/merged", "previews"]:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)

    source_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=lambda p: p.name.casefold())
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        source_paths = [p for p in source_paths if p.name in wanted]
    if args.max_images:
        source_paths = source_paths[: args.max_images]
    if not source_paths:
        LOG.error("No source images found in %s", input_dir)
        return 2

    manifest_path = output_dir / "peanuts_manifest.csv"
    qc_path = output_dir / "qc_manifest.csv"
    errors_path = output_dir / "source_errors.csv"
    manifest_rows = read_csv_rows(manifest_path, MANIFEST_FIELDS)
    qc_rows = read_csv_rows(qc_path, QC_FIELDS)
    errors = read_csv_rows(errors_path, ERROR_FIELDS)
    done_sources = load_done_source_names(output_dir, manifest_rows) if args.resume else set()

    LOG.info("Device=%s; segmenter=%s; sources selected=%d", args.device, args.segmenter, len(source_paths))
    for source_path in source_paths:
        if source_path.name in done_sources:
            LOG.info("Skipping completed source (resume): %s", source_path.name)
            continue
        info = parse_source_info(source_path)
        LOG.info("Processing %s -> group_label=%s region_code=%s source_number=%s", info.original_filename, info.group_label, info.region_code or "", info.source_number or "")
        try:
            errors = [e for e in errors if e.get("original_filename") != source_path.name]
            rgb = read_rgb(source_path)
            detections, _, segmenter_used = detect_instances(
                rgb,
                segmenter=args.segmenter,
                sam_checkpoint=args.sam_checkpoint,
                device=args.device,
            )
            detections = sort_visual_order(detections)
            if not detections:
                raise ValueError("no individual peanut detections")

            preview_path = output_dir / "previews" / f"{source_path.stem}_annotated.jpg"
            draw_preview(rgb, detections, preview_path)
            if "sam" in segmenter_used:
                draw_sam_debug_overlay(
                    rgb,
                    detections,
                    output_dir / "previews" / f"{source_path.stem}_sam_masks_debug.jpg",
                )
            h, w = rgb.shape[:2]
            source_tag = source_number_tag(info.source_number)
            for number, det in enumerate(detections, start=1):
                peanut_id = f"{info.group_label}_{source_tag}_p{number:03d}"
                transparent_path = output_dir / "transparent" / info.group_label / f"{peanut_id}.png"
                cnn_path = output_dir / "cnn_ready_224" / info.group_label / f"{peanut_id}.png"
                # IMPORTANT: final crops use only this deduplicated SAM mask.
                # The watershed labels array is never read in crop generation;
                # no watershed region or union of neighboring masks is used.
                crop_rgb, crop_alpha = crop_with_margin(rgb, det.mask)
                save_rgba_and_cnn(crop_rgb, crop_alpha, transparent_path, cnn_path, args.image_size)
                status = "partial" if "partial_at_image_edge" in det.flags else ("merged" if "possible_merged_mask" in det.flags or "extremely_large_mask" in det.flags else ("uncertain" if det.flags else "clean"))
                qc_file = output_dir / "qc" / status / f"{peanut_id}.png" if status != "clean" else None
                if qc_file:
                    qc_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(transparent_path, qc_file)
                row = {
                    "peanut_id": peanut_id,
                    "original_filename": info.original_filename,
                    "source_image_id": info.source_image_id,
                    "parent_photo_id": info.source_image_id,
                    "source_number": info.source_number,
                    "group_label": info.group_label,
                    "region_code": info.region_code,
                    "peanut_number": number,
                    "transparent_path": transparent_path.relative_to(output_dir).as_posix(),
                    "cnn_ready_path": cnn_path.relative_to(output_dir).as_posix(),
                    "centroid_x": f"{det.centroid_x:.3f}",
                    "centroid_y": f"{det.centroid_y:.3f}",
                    "bbox_x1": det.bbox_x1,
                    "bbox_y1": det.bbox_y1,
                    "bbox_x2": det.bbox_x2,
                    "bbox_y2": det.bbox_y2,
                    "mask_area_pixels": det.area,
                    "original_image_width": w,
                    "original_image_height": h,
                    "partial_at_image_edge": str("partial_at_image_edge" in det.flags).lower(),
                    "segmentation_confidence": f"{det.confidence:.4f}",
                    "qc_status": status,
                    "qc_flags": ";".join(det.flags),
                    "qc_path": qc_file.relative_to(output_dir).as_posix() if qc_file else "",
                    "segmenter": segmenter_used,
                }
                manifest_rows.append(row)
                if status != "clean":
                    qc_rows.append({**row, "source_error": ""})
            atomic_write_csv(manifest_path, MANIFEST_FIELDS, manifest_rows)
            atomic_write_csv(qc_path, QC_FIELDS, qc_rows)
        except Exception as exc:
            LOG.error("Failed %s: %s", source_path.name, exc)
            errors = [e for e in errors if e.get("original_filename") != source_path.name]
            errors.append({
                "original_filename": source_path.name,
                "source_image_id": info.source_image_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            atomic_write_csv(errors_path, ERROR_FIELDS, errors)
            if args.debug:
                traceback.print_exc()

    atomic_write_csv(manifest_path, MANIFEST_FIELDS, manifest_rows)
    atomic_write_csv(qc_path, QC_FIELDS, qc_rows)
    atomic_write_csv(errors_path, ERROR_FIELDS, errors)
    checks = validate_outputs(output_dir, manifest_rows, source_paths, errors, args.image_size)
    if args.archives:
        archives = create_archives(output_dir)
    else:
        archives = []

    statuses = [r.get("qc_status", "") for r in manifest_rows]
    LOG.info("Source photographs processed: %d", len({r.get('original_filename') for r in manifest_rows}))
    LOG.info("Individual peanuts detected: %d", len(manifest_rows))
    LOG.info("Successful clean samples: %d", statuses.count("clean"))
    LOG.info("Partial edge peanuts: %d", statuses.count("partial"))
    LOG.info("QC / uncertain: %d", sum(s in {"uncertain", "merged"} for s in statuses))
    LOG.info("Transparent PNGs: %d", sum(1 for r in manifest_rows if (output_dir / r.get("transparent_path", "__missing__")).exists()))
    LOG.info("CNN-ready PNGs: %d", sum(1 for r in manifest_rows if (output_dir / r.get("cnn_ready_path", "__missing__")).exists()))
    LOG.info("Missing pairs: %d", len(checks["missing_pairs"]))
    LOG.info("Validation: %s", "PASS" if checks["ok"] else "FAIL")
    if archives:
        for archive in archives:
            LOG.info("Archive: %s", archive)
    return 0 if checks["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing group photographs")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--image-size", type=int, default=224, help="CNN output size; default 224")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--segmenter", choices=["auto", "classical", "sam", "mobile_sam"], default="auto")
    parser.add_argument("--sam-checkpoint", default="", help="SAM checkpoint for --segmenter sam/mobile_sam")
    parser.add_argument("--only", default="", help="Comma-separated source filenames for a preview/sample run")
    parser.add_argument("--max-images", type=int, default=0, help="Process at most N images")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--archives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        raise SystemExit(process(build_parser().parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
