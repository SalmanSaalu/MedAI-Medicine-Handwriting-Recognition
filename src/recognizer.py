"""TrOCR inference wrapper aligned with the RXHandBD training image policy.

Training policy: EXIF correction -> grayscale -> RGB -> TrOCRProcessor.
Inference uses the same conversion for single images and detected crops.
Do not apply OpenCV deskewing/thresholding here; those are separate experiments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "microsoft/trocr-base-handwritten"
FINETUNED_MODEL = ROOT / "models" / "trocr-medicine"


def _model_source() -> str:
    required = [FINETUNED_MODEL / "config.json", FINETUNED_MODEL / "model.safetensors"]
    return str(FINETUNED_MODEL) if all(path.is_file() for path in required) else BASE_MODEL


class HandwritingRecognizer:
    def __init__(self, num_beams: int = 3, max_new_tokens: int = 64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_beams = max(1, int(num_beams))
        self.max_new_tokens = max(8, int(max_new_tokens))

        if self.device.type == "cpu":
            threads = max(1, int(os.environ.get("MEDSCRIPT_TORCH_THREADS", "2")))
            torch.set_num_threads(threads)

        source = _model_source()
        self.model_source = source
        print(f"Loading TrOCR model and processor: {source}")
        print(f"Inference device: {self.device}")
        self.processor = TrOCRProcessor.from_pretrained(source)
        self.model = VisionEncoderDecoderModel.from_pretrained(source).to(self.device)
        self.model.eval()

    @staticmethod
    def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            raise TypeError("image must be a PIL Image or NumPy array")

        # Exactly match MedicineDataset.__getitem__ in train_trocr.py.
        pil_image = ImageOps.exif_transpose(pil_image).convert("L").convert("RGB")
        return pil_image

    def _generate(self, pixel_values: torch.Tensor) -> list[str]:
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
        }
        if self.num_beams > 1:
            kwargs["early_stopping"] = True
        generated_ids = self.model.generate(pixel_values, **kwargs)
        return [
            text.strip()
            for text in self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        ]

    @torch.inference_mode()
    def recognize(self, image: Image.Image | np.ndarray) -> str:
        pil_image = self._to_pil(image)
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        return self._generate(pixel_values)[0]

    @torch.inference_mode()
    def recognize_many(
        self,
        images: Sequence[Image.Image | np.ndarray],
        batch_size: int = 2,
    ) -> list[str]:
        outputs: list[str] = []
        batch_size = max(1, int(batch_size))
        for start in range(0, len(images), batch_size):
            batch = [self._to_pil(image) for image in images[start : start + batch_size]]
            pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values
            outputs.extend(self._generate(pixel_values.to(self.device)))
        return outputs
