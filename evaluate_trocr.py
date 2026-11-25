"""Evaluate the saved model on a chosen split and decoding beam count."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from train_trocr import (
    ROOT,
    OUTPUT_DIR,
    MAX_LABEL_LENGTH,
    VALIDATION_NUM_BEAMS,
    MedicineDataset,
    build_collate_fn,
    levenshtein_distance,
)

BATCH_SIZE = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "splits" / "test.csv",
        help="Split CSV to evaluate; defaults to held-out test.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "test_predictions.csv",
        help="Output CSV for per-image predictions.",
    )
    parser.add_argument("--num-beams", type=int, default=VALIDATION_NUM_BEAMS)
    args = parser.parse_args()

    if args.num_beams < 1:
        raise ValueError("--num-beams must be at least 1.")
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {OUTPUT_DIR}. "
            "Run train_trocr.py first."
        )

    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(OUTPUT_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(OUTPUT_DIR).to(device)
    model.eval()
    model.config.use_cache = True

    dataset = MedicineDataset(csv_path)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=build_collate_fn(processor),
    )

    output_rows = []
    total_edits = 0
    total_chars = 0
    exact = 0

    with torch.inference_mode():
        sample_offset = 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                generated = model.generate(
                    pixel_values,
                    max_new_tokens=MAX_LABEL_LENGTH,
                    num_beams=args.num_beams,
                )
            predictions = processor.batch_decode(
                generated, skip_special_tokens=True
            )
            label_ids = labels.detach().cpu().clone()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            references = processor.batch_decode(
                label_ids, skip_special_tokens=True
            )

            for offset, (reference, prediction) in enumerate(
                zip(references, predictions)
            ):
                reference = reference.strip()
                prediction = prediction.strip()
                distance = levenshtein_distance(reference, prediction)
                total_edits += distance
                total_chars += max(1, len(reference))
                is_exact = reference.casefold() == prediction.casefold()
                exact += int(is_exact)

                image_path = Path(
                    dataset.records[sample_offset + offset]["image_path"]
                )
                try:
                    display_path = str(image_path.relative_to(ROOT))
                except ValueError:
                    display_path = str(image_path)
                output_rows.append(
                    {
                        "image_path": display_path,
                        "ground_truth": reference,
                        "prediction": prediction,
                        "edit_distance": distance,
                        "exact_match": is_exact,
                    }
                )
            sample_offset += len(references)

    if not output_rows:
        raise ValueError(f"No valid records found in {csv_path}.")

    cer = total_edits / max(1, total_chars)
    exact_accuracy = exact / len(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Device: {device}")
    print(f"CSV: {csv_path}")
    print(f"Num beams: {args.num_beams}")
    print(f"Images: {len(dataset)}")
    print(f"Character Error Rate (CER): {cer:.4f} ({cer:.2%})")
    print(f"Exact-match accuracy: {exact_accuracy:.2%}")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
