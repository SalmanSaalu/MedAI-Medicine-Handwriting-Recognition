"""Fine-tune microsoft/trocr-base-handwritten on RXHandBD word images.

Run this on a CUDA GPU (for example, a Colab GPU runtime), not on the
user's 8 GB CPU-only Windows desktop.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
VAL_CSV = ROOT / "data" / "splits" / "validation.csv"
BASE_MODEL = "microsoft/trocr-base-handwritten"
OUTPUT_DIR = ROOT / "models" / "trocr-medicine"
METRICS_CSV = ROOT / "training_metrics.csv"

SEED = 42
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 3e-6
WEIGHT_DECAY = 0.01
MAX_LABEL_LENGTH = 64
NUM_WORKERS = 2
VALIDATION_NUM_BEAMS = 3
AUGMENT_TRAIN_IMAGES = True


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class MedicineDataset(Dataset):
    def __init__(self, csv_path: Path, augment: bool = False):
        self.records: list[dict[str, str | Path]] = []
        self.augment = augment
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"image_path", "text"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"{csv_path.name} must contain columns image_path,text. "
                    f"Found: {reader.fieldnames}"
                )

            for row_number, row in enumerate(reader, start=2):
                image_value = (row.get("image_path") or "").strip()
                text_value = (row.get("text") or "").strip()
                if not image_value or not text_value:
                    continue

                image_path = resolve_image_path(image_value)
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image referenced on CSV row {row_number} not found: {image_path}"
                    )

                self.records.append(
                    {"image_path": image_path, "text": text_value}
                )

        if not self.records:
            raise ValueError(f"No valid records were found in {csv_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        record = self.records[index]
        image = Image.open(record["image_path"])
        image = ImageOps.exif_transpose(image).convert("L").convert("RGB")
        if self.augment:
            # Small changes mimic capture and writing variation without changing labels.
            if random.random() < 0.7:
                image = image.rotate(
                    random.uniform(-2.0, 2.0),
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor=(255, 255, 255),
                )
            if random.random() < 0.5:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.95, 1.05))
        return image, str(record["text"])


def build_collate_fn(processor: TrOCRProcessor):
    def collate_fn(batch: list[tuple[Image.Image, str]]) -> dict[str, torch.Tensor]:
        images, texts = zip(*batch)
        pixel_values = processor(
            images=list(images), return_tensors="pt"
        ).pixel_values

        labels = processor.tokenizer(
            list(texts),
            max_length=MAX_LABEL_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

    return collate_fn


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance with O(min(len(a), len(b))) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, char_ref in enumerate(reference, start=1):
        current = [i]
        for j, char_hyp in enumerate(hypothesis, start=1):
            cost = 0 if char_ref == char_hyp else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def calculate_cer(references: list[str], predictions: list[str]) -> float:
    total_edits = 0
    total_characters = 0
    for reference, prediction in zip(references, predictions):
        total_edits += levenshtein_distance(reference, prediction)
        total_characters += max(1, len(reference))
    return total_edits / max(1, total_characters)


def check_label_lengths(
    dataset: MedicineDataset,
    processor: TrOCRProcessor,
    split_name: str,
) -> None:
    too_long: list[str] = []
    for record in dataset.records:
        text = str(record["text"])
        token_ids = processor.tokenizer(
            text, add_special_tokens=True, truncation=False
        ).input_ids
        if len(token_ids) > MAX_LABEL_LENGTH:
            too_long.append(f"{record['image_path']}: {len(token_ids)} tokens")

    if too_long:
        raise ValueError(
            f"{len(too_long)} {split_name} labels exceed MAX_LABEL_LENGTH="
            f"{MAX_LABEL_LENGTH}. The tokenizer would truncate them. "
            f"Examples: {too_long[:10]}"
        )


def validate(
    model: VisionEncoderDecoderModel,
    loader: DataLoader,
    processor: TrOCRProcessor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    sample_total = 0
    references: list[str] = []
    predictions: list[str] = []

    # `use_cache` is disabled during gradient-checkpointed training in some
    # configurations. Re-enable it for generation-based validation.
    previous_use_cache = model.config.use_cache
    model.config.use_cache = True

    with torch.inference_mode():
        for validation_step, batch in enumerate(loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
            ):
                outputs = model(pixel_values=pixel_values, labels=labels)

            batch_count = pixel_values.size(0)
            loss_total += float(outputs.loss.item()) * batch_count
            sample_total += batch_count

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=MAX_LABEL_LENGTH,
                num_beams=VALIDATION_NUM_BEAMS,
                early_stopping=True,
            )
            predictions.extend(
                text.strip()
                for text in processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            )

            label_ids = labels.detach().cpu().clone()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            references.extend(
                text.strip()
                for text in processor.batch_decode(
                    label_ids, skip_special_tokens=True
                )
            )
            if validation_step % 100 == 0 or validation_step == len(loader):
                print(
                    f"Validation batch {validation_step}/{len(loader)}",
                    flush=True,
                )

    model.config.use_cache = previous_use_cache
    exact_match = sum(
        ref.casefold() == pred.casefold()
        for ref, pred in zip(references, predictions)
    ) / max(1, len(references))

    return {
        "validation_loss": loss_total / max(1, sample_total),
        "cer": calculate_cer(references, predictions),
        "exact_match_accuracy": exact_match,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Run fine-tuning in a cloud GPU environment "
            "such as Google Colab with a GPU runtime. Use the local CPU machine "
            "for Streamlit inference, not this training script."
        )

    set_seed()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Loading processor and pretrained TrOCR...", flush=True)
    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    # These token IDs match the microsoft/trocr-base-handwritten checkpoint.
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.use_cache = False

    print("Loading and checking training/validation datasets...", flush=True)
    train_dataset = MedicineDataset(TRAIN_CSV, augment=AUGMENT_TRAIN_IMAGES)
    val_dataset = MedicineDataset(VAL_CSV)
    check_label_lengths(train_dataset, processor, "training")
    check_label_lengths(val_dataset, processor, "validation")
    collate_fn = build_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUMULATION)
    total_updates = updates_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_updates * 0.05))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        remaining = total_updates - current_step
        return max(0.0, float(remaining) / float(max(1, total_updates - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    model.to(device)

    print(f"Training images: {len(train_dataset)}", flush=True)
    print(f"Validation images: {len(val_dataset)}", flush=True)
    print(f"Batch size: {BATCH_SIZE}; accumulation: {GRAD_ACCUMULATION}", flush=True)
    print(f"Optimizer updates: {total_updates}", flush=True)

    best_accuracy = -1.0
    best_cer = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0
        train_samples = 0

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            batch_count = pixel_values.size(0)

            # Scale the final, possibly smaller, accumulation group correctly.
            group_start = (step // GRAD_ACCUMULATION) * GRAD_ACCUMULATION
            group_size = min(
                GRAD_ACCUMULATION, len(train_loader) - group_start
            )

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(pixel_values=pixel_values, labels=labels)
                raw_loss = outputs.loss
                loss = raw_loss / group_size

            scaler.scale(loss).backward()
            train_loss_total += float(raw_loss.item()) * batch_count
            train_samples += batch_count

            update_now = (
                (step + 1) % GRAD_ACCUMULATION == 0
                or (step + 1) == len(train_loader)
            )
            if update_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 25 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {step + 1}/{len(train_loader)} | "
                    f"Loss {raw_loss.item():.4f}", flush=True
                )

        train_loss = train_loss_total / max(1, train_samples)
        print("Running validation...", flush=True)
        metrics = validate(model, val_loader, processor, device)
        row = {
            "epoch": epoch + 1,
            "training_loss": train_loss,
            **metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(METRICS_CSV, index=False)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}", flush=True)
        print(f"Training loss:          {train_loss:.4f}", flush=True)
        print(f"Validation loss:        {metrics['validation_loss']:.4f}", flush=True)
        print(f"Validation CER:         {metrics['cer']:.4f}", flush=True)
        print(f"Exact-match accuracy:   {metrics['exact_match_accuracy']:.2%}", flush=True)

        accuracy = metrics["exact_match_accuracy"]
        cer = metrics["cer"]
        accuracy_improved = accuracy > best_accuracy
        tied_accuracy_better_cer = accuracy == best_accuracy and cer < best_cer

        if accuracy_improved:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if accuracy_improved or tied_accuracy_better_cer:
            best_cer = cer
            model.config.use_cache = True
            model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
            processor.save_pretrained(OUTPUT_DIR)
            model.config.use_cache = False
            print(f"Best exact-match checkpoint saved to: {OUTPUT_DIR}", flush=True)
        else:
            print(
                f"Validation exact match did not improve for "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epoch(s).",
                flush=True,
            )

        torch.cuda.empty_cache()
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping after {epoch + 1} epochs: validation exact match did not "
                f"improve for {EARLY_STOPPING_PATIENCE} consecutive epochs.",
                flush=True,
            )
            break

    print("\nFine-tuning complete.", flush=True)
    print(f"Best validation exact match: {best_accuracy:.2%}", flush=True)
    print(f"CER at selected checkpoint: {best_cer:.4f}", flush=True)
    print(f"Best model: {OUTPUT_DIR}", flush=True)
    print(f"Epoch metrics: {METRICS_CSV}", flush=True)


if __name__ == "__main__":
    main()
