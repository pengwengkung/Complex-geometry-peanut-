"""Create reproducible crop-level and merchant-held-out dataset splits.

The input manifest has one row per segmented peanut crop.  A merchant is
identified from the parent-photo stem: GD1.jpg, GD1(2).jpg, and GD1(3).jpg
are treated as the same merchant, GD1.  The one malformed full-width closing
parenthesis in the source filenames is normalized before parsing.

Outputs are CSV files with one row per crop and a ``split`` column.  The
leave-merchants-out split is deliberately grouped by merchant; the random
split is deliberately not grouped, because it represents an ordinary
image/crop-level 80/20 validation split stratified only by region.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


MERCHANT_RE = re.compile(r"^([A-Za-z]+\d+)(?:\(\d+\))?$")
BASE_COLUMNS = [
    "output_filename",
    "region_label",
    "parent_photo",
    "merchant_id",
    "peanut_index_in_photo",
    "split",
    "sam_mask_score",
    "mask_area_px",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    return parser.parse_args()


def merchant_from_parent_photo(parent_photo: str) -> str:
    """Return a stable merchant ID from the source parent-photo filename."""

    stem = Path(parent_photo).stem
    normalized = stem.replace("（", "(").replace("）", ")")
    match = MERCHANT_RE.fullmatch(normalized)
    if match:
        return match.group(1)
    # Keep an unparseable source photo separate rather than guessing that it
    # belongs to another merchant.
    return f"UNKNOWN_{normalized}"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    required = {
        "output_filename",
        "region_label",
        "parent_photo",
        "peanut_index_in_photo",
        "sam_mask_score",
        "mask_area_px",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    for row in rows:
        row["merchant_id"] = merchant_from_parent_photo(row["parent_photo"])
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def random_region_split(
    rows: list[dict[str, str]], test_fraction: float, seed: int
) -> list[dict[str, str]]:
    """Split individual crop rows, with a separate shuffle inside each region."""

    import random

    rng = random.Random(seed)
    by_region: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_region[row["region_label"]].append(index)

    test_indices: set[int] = set()
    for region in sorted(by_region):
        indices = by_region[region]
        rng.shuffle(indices)
        # Ceiling gives every region a test example and stays at least as
        # close to 20% as possible for small regions.
        test_count = max(1, math.ceil(test_fraction * len(indices)))
        test_indices.update(indices[:test_count])

    result = []
    for index, row in enumerate(rows):
        copy = dict(row)
        copy["split"] = "test" if index in test_indices else "train"
        result.append(copy)
    return result


def choose_region_holdout(
    merchant_counts: dict[str, int], test_fraction: float
) -> set[str]:
    """Choose whole merchants near the requested row fraction for one region.

    A region with one merchant cannot have that merchant in both train and
    test.  We keep that sole merchant in train so that the training set does
    not lose the region completely, and report the limitation in the README.
    """

    merchants = sorted(merchant_counts)
    if len(merchants) <= 1:
        return set()

    target_rows = max(1, round(test_fraction * sum(merchant_counts.values())))
    candidates: list[tuple[tuple[int, int, tuple[str, ...]], set[str]]] = []
    # For the regions in this dataset there are at most six merchants, so an
    # exhaustive search is small and gives a transparent, deterministic choice.
    for mask in range(1, (1 << len(merchants)) - 1):
        selected = {
            merchants[index]
            for index in range(len(merchants))
            if mask & (1 << index)
        }
        selected_rows = sum(merchant_counts[merchant] for merchant in selected)
        score = (
            abs(selected_rows - target_rows),
            abs(len(selected) - max(1, round(test_fraction * len(merchants)))),
            tuple(sorted(selected)),
        )
        candidates.append((score, selected))
    return min(candidates, key=lambda item: item[0])[1]


def leave_merchants_out_split(
    rows: list[dict[str, str]], test_fraction: float
) -> tuple[list[dict[str, str]], set[str]]:
    by_region_merchant: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_region_merchant[row["region_label"]][row["merchant_id"]] += 1

    held_out: set[str] = set()
    held_out_by_region: dict[str, set[str]] = {}
    for region in sorted(by_region_merchant):
        selected = choose_region_holdout(
            dict(by_region_merchant[region]), test_fraction
        )
        held_out_by_region[region] = selected
        held_out.update(selected)

    result = []
    for row in rows:
        copy = dict(row)
        copy["split"] = "test" if row["merchant_id"] in held_out else "train"
        result.append(copy)
    return result, held_out


def distinct_count(rows: list[dict[str, str]], column: str) -> int:
    return len({row[column] for row in rows})


def make_split_summary(
    strategy: str, rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    output = []
    for split in ("train", "test"):
        selected = [row for row in rows if row["split"] == split]
        output.append(
            {
                "strategy": strategy,
                "split": split,
                "merchant_count": distinct_count(selected, "merchant_id"),
                "image_count": len(selected),
                "image_fraction": round(len(selected) / len(rows), 6),
                "regions_present": ";".join(sorted({row["region_label"] for row in selected})),
            }
        )
    return output


def make_region_summary(
    strategy: str, rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    regions = sorted({row["region_label"] for row in rows})
    result = []
    for region in regions:
        all_rows = [row for row in rows if row["region_label"] == region]
        train = [row for row in all_rows if row["split"] == "train"]
        test = [row for row in all_rows if row["split"] == "test"]
        train_merchants = {row["merchant_id"] for row in train}
        test_merchants = {row["merchant_id"] for row in test}
        result.append(
            {
                "strategy": strategy,
                "region_label": region,
                "total_images": len(all_rows),
                "train_images": len(train),
                "test_images": len(test),
                "test_image_fraction": round(len(test) / len(all_rows), 6),
                "total_merchants": len({row["merchant_id"] for row in all_rows}),
                "train_merchants": len(train_merchants),
                "test_merchants": len(test_merchants),
                "merchant_overlap": len(train_merchants & test_merchants),
            }
        )
    return result


def make_merchant_assignments(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_merchant: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_merchant[(row["region_label"], row["merchant_id"])].append(row)
    return [
        {
            "region_label": region,
            "merchant_id": merchant,
            "image_count": len(merchant_rows),
            "split": merchant_rows[0]["split"],
        }
        for (region, merchant), merchant_rows in sorted(by_merchant.items())
    ]


def write_readme(
    output_dir: Path,
    random_rows: list[dict[str, str]],
    merchant_rows: list[dict[str, str]],
    seed: int,
    test_fraction: float,
) -> None:
    random_summary = make_split_summary("random_stratified_80_20", random_rows)
    merchant_summary = make_split_summary("leave_merchants_out", merchant_rows)
    random_region = make_region_summary("random_stratified_80_20", random_rows)
    merchant_region = make_region_summary("leave_merchants_out", merchant_rows)
    single_merchant_regions = [
        item["region_label"]
        for item in merchant_region
        if item["total_merchants"] == 1
    ]
    lines = [
        "# Dataset split definitions",
        "",
        "These files contain one row per segmented peanut crop. Filter the `split` column for `train` or `test`.",
        "",
        "## Files",
        "",
        "- `random_stratified_80_20.csv`: individual crops shuffled within each region; this is the ordinary random comparison split.",
        "- `leave_merchants_out.csv`: all crops from a merchant stay together; no merchant is in both train and test.",
        "- `split_summary.csv`: train/test merchant and crop totals.",
        "- `region_balance.csv`: region-by-region counts and merchant overlap.",
        "- `leave_merchants_out_merchants.csv`: the complete merchant assignment for the grouped split.",
        "",
        "## Reproducibility",
        "",
        f"The random split uses seed `{seed}` and a requested test fraction of `{test_fraction:.2f}`.",
        "The random split is stratified only by `region_label`; it does not keep merchants or parent photos together.",
        "",
        "A merchant ID is parsed from the parent-photo stem: `GD1.jpg`, `GD1(2).jpg`, and `GD1(3).jpg` become merchant `GD1`. Full-width parentheses are normalized before parsing. Unparseable source names are kept as separate `UNKNOWN_...` merchants.",
        "",
        "For the leave-merchants-out split, whole merchants are selected within each region to get as close as possible to 20% of that region's crops while leaving at least one merchant in training. Regions with only one merchant cannot appear in both train and test without breaking the no-overlap rule; those regions are kept in training and are shown explicitly in `region_balance.csv`.",
        "",
        "## Summary",
        "",
        "| strategy | split | merchants | crops | crop fraction |",
        "|---|---|---:|---:|---:|",
    ]
    for item in random_summary + merchant_summary:
        lines.append(
            f"| {item['strategy']} | {item['split']} | {item['merchant_count']} | {item['image_count']} | {item['image_fraction']:.4f} |"
        )
    lines += [
        "",
        "## Region balance",
        "",
        "See `region_balance.csv` for exact values. The leave-merchants-out selection is region-aware for regions with at least two merchants; one-merchant regions are the unavoidable exception described above.",
        "",
        "## CNN-ready pixel handling",
        "",
        "The locally generated CNN-ready set was checked separately: all 2,921 files were 224x224 RGB PNGs with 8-bit pixel values. They are stored as ordinary image values from 0 to 255, which is the correct file format; they are not supposed to store ImageNet-normalized floating-point values.",
        "",
        "When loading these files for an ImageNet-pretrained backbone, convert the pixels to 0-1 and then apply the standard channel normalization: mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). Apply this once at model input, not once when saving the PNG and again when loading it.",
        "",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1")

    rows = load_manifest(args.manifest)
    random_rows = random_region_split(rows, args.test_fraction, args.seed)
    merchant_rows, held_out = leave_merchants_out_split(rows, args.test_fraction)

    # Guardrails: the grouped split must never leak a merchant across sides.
    train_merchants = {row["merchant_id"] for row in merchant_rows if row["split"] == "train"}
    test_merchants = {row["merchant_id"] for row in merchant_rows if row["split"] == "test"}
    overlap = train_merchants & test_merchants
    if overlap:
        raise AssertionError(f"Merchant leakage in leave-out split: {sorted(overlap)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "random_stratified_80_20.csv", random_rows, BASE_COLUMNS)
    write_csv(args.output_dir / "leave_merchants_out.csv", merchant_rows, BASE_COLUMNS)

    summary = make_split_summary("random_stratified_80_20", random_rows)
    summary += make_split_summary("leave_merchants_out", merchant_rows)
    write_csv(
        args.output_dir / "split_summary.csv",
        summary,
        ["strategy", "split", "merchant_count", "image_count", "image_fraction", "regions_present"],
    )

    region_summary = make_region_summary("random_stratified_80_20", random_rows)
    region_summary += make_region_summary("leave_merchants_out", merchant_rows)
    write_csv(
        args.output_dir / "region_balance.csv",
        region_summary,
        [
            "strategy",
            "region_label",
            "total_images",
            "train_images",
            "test_images",
            "test_image_fraction",
            "total_merchants",
            "train_merchants",
            "test_merchants",
            "merchant_overlap",
        ],
    )
    write_csv(
        args.output_dir / "leave_merchants_out_merchants.csv",
        make_merchant_assignments(merchant_rows),
        ["region_label", "merchant_id", "image_count", "split"],
    )
    write_readme(args.output_dir, random_rows, merchant_rows, args.seed, args.test_fraction)

    print(f"Manifest rows: {len(rows)}")
    print(f"Unique merchants: {len({row['merchant_id'] for row in rows})}")
    print(f"Leave-out held merchants: {len(held_out)} ({', '.join(sorted(held_out)) or 'none'})")
    for strategy, split_rows in (
        ("random_stratified_80_20", random_rows),
        ("leave_merchants_out", merchant_rows),
    ):
        print(strategy)
        for item in make_split_summary(strategy, split_rows):
            print(
                f"  {item['split']}: {item['merchant_count']} merchants, "
                f"{item['image_count']} crops ({item['image_fraction']:.4f})"
            )


if __name__ == "__main__":
    main()
