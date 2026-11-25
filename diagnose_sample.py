"""Diagnose TrOCR predictions on one exact RXHandBD sample.

Run from the MedScript-AI project root:
    python diagnose_sample.py --image "data/RxHandBD/Train_Set/P2509.jpg"

Optional:
    python diagnose_sample.py --image "..." --label "Calbo-D"

The script compares the original image with the application's horizontal-line
crops, using greedy and beam-search decoding. It also reports which checkpoint
is loaded and whether the image appears in train/validation/test CSV files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps

from src.preprocessing import prepare_image, detect_text_regions
from src.recognizer import HandwritingRecognizer, _model_source, _processor_source

ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "data" / "RxHandBD" / "annotations.csv"
SPLITS_DIR = ROOT / "data" / "splits"
DEBUG_DIR = ROOT / "diagnostic_outputs"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def find_ground_truth(image_path: Path, supplied_label: str | None) -> str | None:
    if supplied_label:
        return supplied_label
    if not ANNOTATIONS.is_file():
        return None
    df = pd.read_csv(ANNOTATIONS, dtype=str).fillna("")
    if not {"image_path", "text"}.issubset(df.columns):
        print("WARNING: annotations.csv does not have image_path,text columns")
        return None
    for _, row in df.iterrows():
        if resolve_path(row["image_path"]) == image_path:
            return row["text"].strip()
    return None


def report_split_membership(image_path: Path) -> None:
    print("\nDATASET SPLIT MEMBERSHIP")
    found = False
    for split_name in ("train", "validation", "test"):
        csv_path = SPLITS_DIR / f"{split_name}.csv"
        if not csv_path.is_file():
            print(f"  {split_name}: CSV not found ({csv_path})")
            continue
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        if "image_path" not in df.columns:
            print(f"  {split_name}: missing image_path column")
            continue
        matches = df["image_path"].apply(lambda value: resolve_path(value) == image_path)
        if matches.any():
            found = True
            label = df.loc[matches, "text"].iloc[0] if "text" in df.columns else "(no label column)"
            print(f"  {split_name}: FOUND | label={label}")
    if not found:
        print("  Image was not found in any generated split CSV.")
        print("  It may be absent from the annotation CSV or the splits may need regeneration.")


def predict(recognizer: HandwritingRecognizer, image: Image.Image, beams: int) -> str:
    pixel_values = recognizer.processor(
        images=image.convert("RGB"), return_tensors="pt"
    ).pixel_values.to(recognizer.device)
    kwargs = {"max_new_tokens": 64, "num_beams": beams}
    if beams > 1:
        kwargs["early_stopping"] = True
    generated = recognizer.model.generate(pixel_values, **kwargs)
    return recognizer.processor.batch_decode(
        generated, skip_special_tokens=True
    )[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image path relative to project root or absolute")
    parser.add_argument("--label", default=None, help="Optional known ground-truth transcription")
    args = parser.parse_args()

    image_path = resolve_path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("PROJECT ROOT:", ROOT)
    print("IMAGE:", image_path)
    print("MODEL CHECKPOINT:", _model_source())
    print("PROCESSOR CHECKPOINT:", _processor_source())
    print("CUDA AVAILABLE:", torch.cuda.is_available())

    ground_truth = find_ground_truth(image_path, args.label)
    print("GROUND TRUTH:", ground_truth if ground_truth else "Not found; provide --label if known")
    report_split_membership(image_path)

    recognizer = HandwritingRecognizer(num_beams=1, max_new_tokens=64)
    original = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    print("\nTEST A — EXACT ORIGINAL IMAGE, NO OPENCV PREPROCESSING")
    for beams in (1, 4):
        print(f"  num_beams={beams}: {predict(recognizer, original, beams)!r}")

    print("\nTEST B — APPLICATION PREPROCESSING + HORIZONTAL DETECTION")
    gray = prepare_image(original)
    regions = detect_text_regions(gray, mode="horizontal")
    print("  Detected regions:", len(regions))

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(DEBUG_DIR / "preprocessed_full_image.png")
    if not regions:
        print("  No regions detected. No full-page fallback was run.")
    for index, region in enumerate(regions, start=1):
        crop = Image.fromarray(region["image"]).convert("RGB")
        crop_path = DEBUG_DIR / f"horizontal_crop_{index:02d}.png"
        crop.save(crop_path)
        print(f"  Crop {index}; bbox={region['bbox']}; saved={crop_path}")
        for beams in (1, 4):
            print(f"    num_beams={beams}: {predict(recognizer, crop, beams)!r}")

    print("\nDIAGNOSTIC OUTPUTS:", DEBUG_DIR)
    print("If the model checkpoint path above is microsoft/trocr-base-handwritten,")
    print("the app is not loading your local fine-tuned checkpoint.")


if __name__ == "__main__":
    main()
