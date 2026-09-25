"""Reproduce the frozen-ImageNet-embedding baseline on the standard split.

This script follows the paper's public description:

1. Load ImageNet-pretrained InceptionV3 (default) or ResNet50.
2. Freeze it and keep its 2048-value penultimate representation.
3. Train scikit-learn kNN, SVM, Random Forest, Logistic Regression, and a
   small one-hidden-layer neural network on those fixed representations.
4. Measure macro one-vs-rest ROC-AUC with 10-fold CV, leave-one-out CV, and
   ten 66%-training repeated random splits.

Important reproducibility notes:

* The paper describes the models and validation families but does not publish
  every classifier hyperparameter.  The explicit defaults below are therefore
  reproducible baseline choices, not claims about undisclosed Orange defaults.
* By default, CV uses the ``train`` side of the standard 80/20 split.  The
  previous step's 20% side remains untouched.  Use --partition all only when
  you intentionally want paper-style CV over every available crop.
* The current local data have eight region labels and 2,921 segmented crops,
  while the paper describes five regions and about 6,500 pods.  The script
  reports this mismatch in run_metadata.json so an AUC comparison is honest.

Example from the repository root in VS Code's terminal (PowerShell):

  python scripts/reproduce_peng_baseline.py `
      --cnn-root C:/path/to/cnn_ready_pngs `
      --split-csv splits/random_stratified_80_20.csv `
      --out-dir baseline_results

The CNN-ready PNGs are loaded with the ImageNet mean and standard deviation
defined by torchvision's official pretrained weights.  The PNG files remain
ordinary 8-bit images; normalization happens only when the model reads them.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    Inception_V3_Weights,
    ResNet50_Weights,
    inception_v3,
    resnet50,
)


PAPER_AUC_REFERENCE = 0.986
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class RunData:
    X: np.ndarray
    y: np.ndarray
    class_names: list[str]
    filenames: list[str]
    metadata: dict[str, object]


class PeanutImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform: Callable):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-root", type=Path, required=True,
                        help="Folder containing region subfolders of CNN-ready PNGs.")
    parser.add_argument("--split-csv", type=Path,
                        default=Path("splits/random_stratified_80_20.csv"))
    parser.add_argument("--partition", choices=("train", "all"), default="train",
                        help="CV data: standard-split train side, or all rows.")
    parser.add_argument("--exclude-label", action="append", default=[],
                        help="Exclude a label before validation; repeat for multiple labels. "
                             "Useful for UNKNOWN labels that are not paper regions.")
    parser.add_argument("--backbone", choices=("inception_v3", "resnet50"),
                        default="inception_v3")
    parser.add_argument("--out-dir", type=Path, default=Path("baseline_results"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Use 0 on Windows unless you deliberately need workers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter-mlp", type=int, default=100)
    parser.add_argument("--loo-progress-every", type=int, default=250)
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-extract embeddings instead of using the saved cache.")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split_rows(path: Path, partition: str) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if partition != "all":
        rows = [row for row in rows if row["split"] == partition]
    if not rows:
        raise ValueError(f"No rows selected from {path} for partition={partition!r}")
    required = {"output_filename", "region_label"}
    if partition != "all":
        required.add("split")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Split file is missing columns: {sorted(missing)}")
    if len({row["output_filename"] for row in rows}) != len(rows):
        raise ValueError("The selected split contains duplicate crop filenames.")
    return rows


def resolve_image_paths(cnn_root: Path, rows: list[dict[str, str]]) -> list[Path]:
    """Resolve the repository's region/output_filename layout, with a safe fallback."""

    paths: list[Path] = []
    missing: list[str] = []
    for row in rows:
        filename = row["output_filename"]
        preferred = cnn_root / row["region_label"] / filename
        flat = cnn_root / filename
        if preferred.is_file():
            paths.append(preferred)
        elif flat.is_file():
            paths.append(flat)
        else:
            missing.append(str(preferred))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Could not find {len(missing)} CNN-ready files. First paths tried:\n{preview}"
        )
    return paths


def build_backbone(name: str):
    if name == "inception_v3":
        weights = Inception_V3_Weights.DEFAULT
        # torchvision's official InceptionV3 weights require the auxiliary
        # head to be enabled at construction time.  It is inactive in eval
        # mode, and we replace only the final classifier below.
        model = inception_v3(weights=weights)
        # InceptionV3's fc input is the 2048-value penultimate representation.
        model.fc = nn.Identity()
        return model, weights.transforms(), "InceptionV3"
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    # ResNet50's fc input is also a 2048-value penultimate representation.
    model.fc = nn.Identity()
    return model, weights.transforms(), "ResNet50"


def extract_embeddings(
    paths: list[Path],
    labels: list[str],
    filenames: list[str],
    backbone_name: str,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    no_cache: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    cache_path = out_dir / f"embeddings_{backbone_name}.npz"
    cache_key = [str(path.resolve()) for path in paths]
    if cache_path.is_file() and not no_cache:
        cached = np.load(cache_path, allow_pickle=False)
        cached_paths = cached["paths"].astype(str).tolist()
        if cached_paths == cache_key:
            X = cached["X"].astype(np.float32, copy=False)
            if X.ndim == 2 and X.shape[1] == 2048:
                print(f"Using cached embeddings: {cache_path}")
                return X, {"cache": str(cache_path), "extracted": False}

    model, transform, model_label = build_backbone(backbone_name)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)

    dataset = PeanutImageDataset(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    X_batches: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, (images, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            embeddings = model(images)
            if hasattr(embeddings, "logits"):
                embeddings = embeddings.logits
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            if embeddings.shape[1] != 2048:
                raise RuntimeError(
                    f"Expected 2048 features from {model_label}, got {embeddings.shape[1]}"
                )
            X_batches.append(embeddings.detach().cpu().numpy().astype(np.float32))
            if batch_number == 1 or batch_number % 10 == 0:
                print(f"Embedding batch {batch_number}/{len(loader)}")

    X = np.concatenate(X_batches, axis=0)
    np.savez_compressed(cache_path, X=X, paths=np.asarray(cache_key))
    print(f"Extracted {len(X)} x {X.shape[1]} embeddings in {time.perf_counter() - started:.1f}s")
    return X, {"cache": str(cache_path), "extracted": True}


def make_classifiers(seed: int, max_iter_mlp: int) -> OrderedDict[str, object]:
    """Explicit, modest sklearn baselines; every estimator is cloned per fold."""

    return OrderedDict(
        [
            (
                "kNN",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)),
                    ]
                ),
            ),
            (
                "SVM",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("classifier", SVC(C=1.0, kernel="rbf", gamma="scale", probability=False)),
                    ]
                ),
            ),
            (
                "Random Forest",
                RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
            ),
            (
                "Logistic Regression",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
                    ]
                ),
            ),
            (
                "Feedforward NN",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("classifier", MLPClassifier(
                            hidden_layer_sizes=(64,),
                            activation="relu",
                            solver="adam",
                            max_iter=max_iter_mlp,
                            random_state=seed,
                            early_stopping=False,
                        )),
                    ]
                ),
            ),
        ]
    )


def score_matrix(estimator, X_test: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)
    return estimator.decision_function(X_test)


def auc_from_scores(y_true: np.ndarray, scores: np.ndarray, n_classes: int) -> float:
    # Every requested split is stratified, so every class is represented in the
    # aggregated scores.  The result is macro one-vs-rest AUC across region labels.
    return float(
        roc_auc_score(
            y_true,
            scores,
            labels=np.arange(n_classes),
            multi_class="ovr",
            average="macro",
        )
    )


def evaluate_10fold(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    seed: int,
) -> list[float]:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    for train_idx, test_idx in cv.split(X, y):
        fitted = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted.fit(X[train_idx], y[train_idx])
        fold_scores.append(auc_from_scores(y[test_idx], score_matrix(fitted, X[test_idx]), n_classes))
    return fold_scores


def evaluate_loo(
    name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    progress_every: int,
) -> list[float]:
    """Return one pooled AUC for LOOCV; individual one-row folds have no AUC."""

    loo = LeaveOneOut()
    pooled_scores = np.zeros((len(y), n_classes), dtype=np.float64)
    for fold_number, (train_idx, test_idx) in enumerate(loo.split(X, y), start=1):
        fitted = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted.fit(X[train_idx], y[train_idx])
        pooled_scores[test_idx] = score_matrix(fitted, X[test_idx])[0]
        if fold_number == 1 or fold_number % progress_every == 0 or fold_number == len(y):
            print(f"  {name} LOOCV: {fold_number}/{len(y)}")
    return [auc_from_scores(y, pooled_scores, n_classes)]


def evaluate_repeated_random(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    seed: int,
) -> list[float]:
    cv = StratifiedShuffleSplit(n_splits=10, train_size=0.66, random_state=seed)
    scores: list[float] = []
    for train_idx, test_idx in cv.split(X, y):
        fitted = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted.fit(X[train_idx], y[train_idx])
        scores.append(auc_from_scores(y[test_idx], score_matrix(fitted, X[test_idx]), n_classes))
    return scores


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_iter_mlp < 1:
        raise ValueError("--batch-size and --max-iter-mlp must be positive")
    set_seeds(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_split_rows(args.split_csv, args.partition)
    if args.exclude_label:
        excluded = set(args.exclude_label)
        rows = [row for row in rows if row["region_label"] not in excluded]
        print(f"Excluded labels: {sorted(excluded)}")
    class_counts = Counter(row["region_label"] for row in rows)
    too_small = {label: count for label, count in class_counts.items() if count < 10}
    if too_small:
        raise ValueError(
            "10-fold cross-validation requires at least 10 crops per label. "
            f"Too-small labels: {too_small}. Use --partition all, or explicitly "
            "exclude labels that are not valid paper regions with --exclude-label."
        )
    paths = resolve_image_paths(args.cnn_root, rows)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform([row["region_label"] for row in rows])
    filenames = [row["output_filename"] for row in rows]
    labels = [row["region_label"] for row in rows]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Selected crops: {len(rows)}")
    print(f"Regions: {dict(zip(label_encoder.classes_, np.bincount(y))) }")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")

    X, embedding_metadata = extract_embeddings(
        paths=paths,
        labels=labels,
        filenames=filenames,
        backbone_name=args.backbone,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        no_cache=args.no_cache,
    )
    if X.shape != (len(rows), 2048):
        raise RuntimeError(f"Unexpected embedding matrix shape: {X.shape}")

    classifiers = make_classifiers(args.seed, args.max_iter_mlp)
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    n_classes = len(label_encoder.classes_)
    for classifier_name, estimator in classifiers.items():
        print(f"\n=== {classifier_name} ===")
        for validation_method in ("10-fold", "leave-one-out", "repeated-random-66"):
            started = time.perf_counter()
            if validation_method == "10-fold":
                scores = evaluate_10fold(estimator, X, y, n_classes, args.seed)
            elif validation_method == "leave-one-out":
                scores = evaluate_loo(
                    classifier_name,
                    estimator,
                    X,
                    y,
                    n_classes,
                    args.loo_progress_every,
                )
            else:
                scores = evaluate_repeated_random(estimator, X, y, n_classes, args.seed)
            median_auc = float(np.median(scores))
            print(
                f"{validation_method}: median AUC={median_auc:.6f}; "
                f"range={min(scores):.6f}-{max(scores):.6f}; "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )
            summary_rows.append(
                {
                    "backbone": args.backbone,
                    "partition": args.partition,
                    "classifier": classifier_name,
                    "validation_method": validation_method,
                    "median_auc": round(median_auc, 8),
                    "mean_auc": round(float(np.mean(scores)), 8),
                    "min_auc": round(float(np.min(scores)), 8),
                    "max_auc": round(float(np.max(scores)), 8),
                    "n_scores": len(scores),
                    "paper_reference_auc": PAPER_AUC_REFERENCE,
                    "below_paper_reference": median_auc < PAPER_AUC_REFERENCE,
                }
            )
            for score_number, score in enumerate(scores, start=1):
                detail_rows.append(
                    {
                        "backbone": args.backbone,
                        "partition": args.partition,
                        "classifier": classifier_name,
                        "validation_method": validation_method,
                        "repeat_or_fold": score_number,
                        "auc": round(float(score), 8),
                    }
                )

    write_csv(
        args.out_dir / "auc_summary.csv",
        summary_rows,
        [
            "backbone",
            "partition",
            "classifier",
            "validation_method",
            "median_auc",
            "mean_auc",
            "min_auc",
            "max_auc",
            "n_scores",
            "paper_reference_auc",
            "below_paper_reference",
        ],
    )
    write_csv(
        args.out_dir / "auc_details.csv",
        detail_rows,
        [
            "backbone",
            "partition",
            "classifier",
            "validation_method",
            "repeat_or_fold",
            "auc",
        ],
    )

    metadata = {
        "backbone": args.backbone,
        "partition": args.partition,
        "excluded_labels": args.exclude_label,
        "split_csv": str(args.split_csv.resolve()),
        "cnn_root": str(args.cnn_root.resolve()),
        "selected_crops": len(rows),
        "region_labels": label_encoder.classes_.tolist(),
        "region_counts": {
            name: int(count)
            for name, count in zip(label_encoder.classes_, np.bincount(y))
        },
        "feature_dimension": int(X.shape[1]),
        "device": str(device),
        "torch_version": torch.__version__,
        "image_preprocessing": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "source": "torchvision official ImageNet weight transform",
        },
        "classifier_settings": {
            "kNN": "StandardScaler + 5-neighbor distance-weighted kNN",
            "SVM": "StandardScaler + RBF SVC, C=1, gamma=scale, decision scores",
            "Random Forest": "200 trees, sklearn defaults otherwise",
            "Logistic Regression": "StandardScaler + sklearn logistic regression, max_iter=1000",
            "Feedforward NN": f"StandardScaler + one hidden ReLU layer of 64, max_iter={args.max_iter_mlp}",
        },
        "validation_settings": {
            "10-fold": "StratifiedKFold(n_splits=10, shuffle=True, random_state=seed); median of fold AUCs",
            "leave-one-out": "LeaveOneOut; one pooled out-of-fold multiclass AUC because one-row folds cannot have an AUC",
            "repeated-random-66": "10 StratifiedShuffleSplit repetitions, train_size=0.66; median of repetition AUCs",
        },
        "paper_comparison": {
            "reported_reference": ">0.986 median AUC for direct image embeddings",
            "paper_dataset_description": "about 6500 pods, five regions",
            "current_dataset_note": "current local segmented set is smaller and has eight labels; comparisons are directional, not literal replication",
        },
        "embedding_cache": embedding_metadata,
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {args.out_dir / 'auc_summary.csv'}")
    print(f"Saved: {args.out_dir / 'auc_details.csv'}")
    print(f"Saved: {args.out_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
