"""Create validated train/validation/test CSVs without image leakage."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "RxHandBD" / "annotations.csv"
DEFAULT_OUTPUT = ROOT / "data" / "splits"
DEFAULT_MANIFEST_DIR = DEFAULT_OUTPUT
SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}
MANIFEST_FILES = {
    "train": "train_manifest.csv",
    "validation": "validation_manifest.csv",
    "test": "test_manifest.csv",
}
PREFERENCE = {"test": 0, "validation": 1, "train": 2}


def resolve_project_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def read_table(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    frame = pd.read_csv(
        csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    required = {"image_path", "text"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{csv_path.name} must contain image_path,text. "
            f"Found: {list(frame.columns)}"
        )
    frame = frame[["image_path", "text"]].copy()
    frame["image_path"] = frame["image_path"].str.strip()
    frame["text"] = frame["text"].str.strip()
    if (frame["image_path"] == "").any() or (frame["text"] == "").any():
        raise ValueError(f"{csv_path} contains blank paths or labels.")
    if frame["image_path"].duplicated().any():
        examples = frame.loc[
            frame["image_path"].duplicated(False), "image_path"
        ].head(10).tolist()
        raise ValueError(f"{csv_path} has duplicate paths: {examples}")
    return frame


def resolve_image_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_images(frame: pd.DataFrame) -> dict[str, str]:
    missing: list[str] = []
    unreadable: list[str] = []
    digest_by_path: dict[str, str] = {}
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for value, text in zip(frame["image_path"], frame["text"]):
        path = resolve_image_path(value)
        if not path.is_file():
            missing.append(value)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            unreadable.append(value)
            continue

        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        value_hash = digest.hexdigest()
        digest_by_path[value] = value_hash
        groups[value_hash].append((value, text))

    if missing or unreadable:
        raise ValueError(
            f"Image check failed: {len(missing)} missing and "
            f"{len(unreadable)} unreadable images. "
            f"Missing examples: {missing[:10]}; unreadable examples: {unreadable[:10]}"
        )

    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    conflict_groups = [
        items for items in duplicate_groups
        if len({text.casefold() for _, text in items}) > 1
    ]
    print(f"Images checked and readable: {len(frame)}")
    print(f"Exact duplicate image groups: {len(duplicate_groups)}")
    print(f"Duplicate groups with conflicting labels: {len(conflict_groups)}")
    return digest_by_path


def load_manifest(
    annotations: pd.DataFrame, manifest_dir: Path
) -> dict[str, pd.DataFrame]:
    annotation_labels = dict(zip(annotations["image_path"], annotations["text"]))
    membership: dict[str, str] = {}
    splits: dict[str, pd.DataFrame] = {}

    for split_name, filename in MANIFEST_FILES.items():
        part = read_table(manifest_dir / filename)
        for image_path, text in zip(part["image_path"], part["text"]):
            if image_path in membership:
                raise ValueError(
                    f"Manifest overlap: {image_path} occurs in "
                    f"{membership[image_path]} and {split_name}."
                )
            membership[image_path] = split_name
            if image_path not in annotation_labels:
                raise ValueError(
                    f"Manifest image is absent from annotations.csv: {image_path}"
                )
            if annotation_labels[image_path] != text:
                raise ValueError(
                    f"Manifest label differs for {image_path}: "
                    f"{annotation_labels[image_path]!r} vs {text!r}"
                )
        splits[split_name] = part

    annotation_paths = set(annotation_labels)
    manifest_paths = set(membership)
    missing = sorted(annotation_paths - manifest_paths)
    extra = sorted(manifest_paths - annotation_paths)
    if missing or extra:
        raise ValueError(
            "Manifest must cover each annotation exactly once. "
            f"Missing: {len(missing)}, extra: {len(extra)}. "
            f"Missing examples: {missing[:10]}; extra examples: {extra[:10]}"
        )
    return splits


def resolve_duplicate_content(
    splits: dict[str, pd.DataFrame],
    digest_by_path: dict[str, str],
    report_path: Path,
) -> tuple[dict[str, pd.DataFrame], int, int, int]:
    groups: dict[str, list[tuple[str, str, str, int]]] = defaultdict(list)
    for split_name, frame in splits.items():
        for row_index, row in enumerate(frame.itertuples(index=False)):
            groups[digest_by_path[row.image_path]].append(
                (split_name, row.image_path, row.text, row_index)
            )

    remove_paths: set[str] = set()
    report_rows: list[dict[str, str]] = []
    duplicate_group_count = 0
    conflict_group_count = 0
    duplicate_copies_removed = 0
    conflict_rows_removed = 0

    for digest, entries in groups.items():
        if len(entries) < 2:
            continue
        duplicate_group_count += 1
        labels = {entry[2].casefold() for entry in entries}
        conflicting = len(labels) > 1

        if conflicting:
            conflict_group_count += 1
            conflict_rows_removed += len(entries)
            remove_paths.update(entry[1] for entry in entries)
            for split_name, image_path, text, _ in entries:
                report_rows.append(
                    {
                        "sha256": digest,
                        "split": split_name,
                        "image_path": image_path,
                        "text": text,
                        "action": "excluded_conflicting_labels",
                        "reason": "Identical image bytes have different labels; review manually.",
                    }
                )
            continue

        winner = min(
            entries,
            key=lambda entry: (PREFERENCE[entry[0]], entry[3]),
        )
        duplicate_copies_removed += len(entries) - 1
        for split_name, image_path, text, _ in entries:
            if image_path == winner[1]:
                action = "kept"
                reason = "One copy retained; held-out split is preferred."
            else:
                action = "excluded_duplicate"
                reason = f"Identical labeled image retained in {winner[0]} only."
                remove_paths.add(image_path)
            report_rows.append(
                {
                    "sha256": digest,
                    "split": split_name,
                    "image_path": image_path,
                    "text": text,
                    "action": action,
                    "reason": reason,
                }
            )

    filtered = {
        split_name: frame[
            ~frame["image_path"].isin(remove_paths)
        ].reset_index(drop=True)
        for split_name, frame in splits.items()
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        report_rows,
        columns=["sha256", "split", "image_path", "text", "action", "reason"],
    ).to_csv(report_path, index=False)
    return filtered, duplicate_group_count, duplicate_copies_removed, conflict_rows_removed


def make_random_split(
    annotations: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    shuffled = annotations.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    return {
        "train": shuffled.iloc[:train_end].reset_index(drop=True),
        "validation": shuffled.iloc[train_end:validation_end].reset_index(drop=True),
        "test": shuffled.iloc[validation_end:].reset_index(drop=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate data, de-duplicate exact images, and create splits."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--validation", type=float, default=0.10)
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Ignore supplied membership CSVs and split rows randomly.",
    )
    args = parser.parse_args()

    csv_path = resolve_project_path(args.csv)
    output_dir = resolve_project_path(args.output_dir)
    manifest_dir = resolve_project_path(args.manifest_dir)
    annotations = read_table(csv_path)
    if len(annotations) < 10:
        raise ValueError(f"Only {len(annotations)} rows found; need at least 10.")

    digest_by_path = validate_images(annotations)
    manifest_paths = [manifest_dir / name for name in MANIFEST_FILES.values()]
    manifest_exists = [path.is_file() for path in manifest_paths]

    if not args.random_split and all(manifest_exists):
        splits = load_manifest(annotations, manifest_dir)
        source = f"preserved membership from {manifest_dir}"
    elif not args.random_split and any(manifest_exists):
        missing = [
            str(path) for path, exists in zip(manifest_paths, manifest_exists)
            if not exists
        ]
        raise FileNotFoundError(
            "Split manifest is incomplete. Missing: " + ", ".join(missing)
        )
    else:
        if not (0 < args.train < 1 and 0 < args.validation < 1):
            raise ValueError("train and validation fractions must be between 0 and 1.")
        if args.train + args.validation >= 1:
            raise ValueError("train + validation must be less than 1.")
        splits = make_random_split(
            annotations, args.train, args.validation, args.seed
        )
        source = "random row-level split"
        print(
            "WARNING: annotations.csv contains no writer or prescription IDs; "
            "random rows cannot guarantee separation by writer or prescription."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    splits, duplicate_groups, duplicate_copies, conflict_rows = (
        resolve_duplicate_content(
            splits,
            digest_by_path,
            output_dir / "duplicate_resolution.csv",
        )
    )
    if any(frame.empty for frame in splits.values()):
        raise ValueError(
            "At least one split became empty after duplicate resolution. "
            "Review duplicate_resolution.csv."
        )

    for split_name, frame in splits.items():
        frame.to_csv(output_dir / SPLIT_FILES[split_name], index=False)

    retained = sum(len(frame) for frame in splits.values())
    print(f"Input annotation rows: {len(annotations)}")
    print(f"Exact duplicate groups resolved: {duplicate_groups}")
    print(f"Same-label duplicate copies removed: {duplicate_copies}")
    print(f"Rows excluded for conflicting labels: {conflict_rows}")
    print(f"Unique image records retained: {retained}")
    for split_name in SPLIT_FILES:
        print(f"{split_name.title():12s}{len(splits[split_name])}")
    print(f"Split source: {source}")
    print(f"CSV splits saved to: {output_dir}")
    print(f"Review duplicate decisions in: {output_dir / 'duplicate_resolution.csv'}")
    print("Proceed to training only if this command exits successfully.")
    print("Keep test.csv untouched until final model evaluation.")


if __name__ == "__main__":
    main()
