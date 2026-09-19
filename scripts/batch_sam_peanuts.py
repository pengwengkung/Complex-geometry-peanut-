#!/usr/bin/env python3
"""Resume-safe batch export for the validated single-point SAM pipeline."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

from process_peanuts import (
    _segment_seed_points_with_promptable_sam,
    alpha_from_mask,
    detect_seed_points,
    draw_sam_debug_overlay,
    read_rgb,
    save_rgba_and_cnn,
    sort_visual_order,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MANIFEST_FIELDS = [
    "output_filename",
    "region_label",
    "parent_photo",
    "peanut_index_in_photo",
    "sam_mask_score",
    "mask_area_px",
]
FILENAME_RE = re.compile(
    r"^(?P<region>[A-Za-z]+)(?P<number>\d+)(?:[\(\uFF08](?P<photo_number>\d+)[\)\uFF09])?$"
)


def parse_photo_label(path: Path) -> tuple[str, str]:
    """Return region and a filesystem-safe photo id from a source filename."""
    stem = path.stem
    match = FILENAME_RE.fullmatch(stem)
    if match:
        region = match.group("region").upper()
    else:
        leading_letters = re.match(r"^[A-Za-z]+", stem)
        region = leading_letters.group(0).upper() if leading_letters else "UNKNOWN"
    photo_id = re.sub(r"[\(\)\uFF08\uFF09]", "_", stem).strip("_")
    return region, photo_id


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def append_error(path: Path, source_name: str, exc: BaseException) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{source_name}\t{type(exc).__name__}: {exc}\n")


def photo_rows_complete(output_dir: Path, rows: list[dict[str, str]], parent_photo: str) -> bool:
    photo_rows = [row for row in rows if row.get("parent_photo") == parent_photo]
    if not photo_rows:
        return False
    for row in photo_rows:
        filename = row.get("output_filename", "")
        region = row.get("region_label", "")
        transparent = output_dir / "transparent_pngs" / region / filename
        cnn = output_dir / "cnn_ready_pngs" / region / filename
        if not transparent.is_file() or not cnn.is_file():
            return False
    return True


def crop_mask_with_padding(rgb: np.ndarray, mask: np.ndarray, padding_px: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Crop only one SAM mask's bbox, with a small fixed border."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("SAM returned an empty mask")
    height, width = mask.shape
    x1 = max(0, int(xs.min()) - padding_px)
    y1 = max(0, int(ys.min()) - padding_px)
    x2 = min(width, int(xs.max()) + 1 + padding_px)
    y2 = min(height, int(ys.max()) + 1 + padding_px)
    local_mask = mask[y1:y2, x1:x2]
    return rgb[y1:y2, x1:x2].copy(), alpha_from_mask(local_mask)


def print_progress(completed: int, total: int, source_name: str, crop_count: int, failed: list[str], skipped: bool) -> None:
    action = "skipped-existing" if skipped else "processed"
    print(
        f"[{completed}/{total}] {action} {source_name} | "
        f"peanuts={crop_count} | failures={len(failed)}",
        flush=True,
    )


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transparent_pngs").mkdir(exist_ok=True)
    (output_dir / "cnn_ready_pngs").mkdir(exist_ok=True)
    (output_dir / "debug_overlays").mkdir(exist_ok=True)

    source_paths = sorted(
        [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.casefold(),
    )
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        source_paths = [path for path in source_paths if path.name in wanted]
    if not source_paths:
        raise ValueError(f"No source images found in {input_dir}")

    manifest_path = output_dir / "manifest.csv"
    errors_path = output_dir / "errors.txt"
    errors_path.touch(exist_ok=True)
    rows = read_manifest(manifest_path)
    failed: list[str] = []
    completed = 0

    print(f"Starting batch: {len(source_paths)} photos", flush=True)
    for source_path in source_paths:
        region, photo_id = parse_photo_label(source_path)
        parent_photo = source_path.name
        if photo_rows_complete(output_dir, rows, parent_photo):
            completed += 1
            print_progress(completed, len(source_paths), parent_photo, sum(1 for row in rows if row.get("parent_photo") == parent_photo), failed, skipped=True)
            continue

        # Remove incomplete rows before retrying this parent photo, preventing
        # duplicate manifest entries after an interrupted or failed attempt.
        rows = [row for row in rows if row.get("parent_photo") != parent_photo]
        write_manifest(manifest_path, rows)
        try:
            rgb = read_rgb(source_path)
            seeds, _ = detect_seed_points(rgb)
            detections = _segment_seed_points_with_promptable_sam(
                rgb,
                seeds,
                args.checkpoint,
                args.device,
                model_family="sam",
            )
            detections = sort_visual_order(detections)
            if not detections:
                raise ValueError("no deduplicated SAM masks were produced")

            debug_path = output_dir / "debug_overlays" / f"{region}_{photo_id}_sam_masks_debug.jpg"
            draw_sam_debug_overlay(rgb, detections, debug_path)

            photo_rows: list[dict[str, object]] = []
            for index, detection in enumerate(detections, start=1):
                filename = f"{region}_{photo_id}_p{index:03d}.png"
                transparent_path = output_dir / "transparent_pngs" / region / filename
                cnn_path = output_dir / "cnn_ready_pngs" / region / filename
                # Final crops use only this individual SAM mask's bbox. The
                # watershed labels array is never read in final crop generation.
                crop_rgb, crop_alpha = crop_mask_with_padding(rgb, detection.mask, args.padding_px)
                save_rgba_and_cnn(crop_rgb, crop_alpha, transparent_path, cnn_path, args.image_size)
                photo_rows.append({
                    "output_filename": filename,
                    "region_label": region,
                    "parent_photo": parent_photo,
                    "peanut_index_in_photo": index,
                    "sam_mask_score": f"{detection.confidence:.6f}",
                    "mask_area_px": int(detection.area),
                })

            rows.extend(photo_rows)
            write_manifest(manifest_path, rows)
            completed += 1
            print_progress(completed, len(source_paths), parent_photo, len(photo_rows), failed, skipped=False)
        except Exception as exc:  # one bad source must not stop the batch
            failed.append(parent_photo)
            append_error(errors_path, parent_photo, exc)
            completed += 1
            print_progress(completed, len(source_paths), parent_photo, 0, failed, skipped=False)

    region_counts = Counter(row.get("region_label", "UNKNOWN") for row in rows)
    print("\nFinal totals", flush=True)
    print(f"photos processed: {len(source_paths) - len(failed)}", flush=True)
    print(f"photos failed: {len(failed)}", flush=True)
    if failed:
        for source_name in failed:
            print(f"  - {source_name}", flush=True)
    print(f"total individual peanut crops: {len(rows)}", flush=True)
    print("crops by region:", flush=True)
    for region, count in sorted(region_counts.items()):
        print(f"  {region}: {count}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"errors: {errors_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--padding-px", type=int, default=8)
    parser.add_argument("--only", default="", help="Optional comma-separated source filenames for a focused test")
    return parser


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(run(build_parser().parse_args()))
