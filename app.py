from __future__ import annotations

import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.preprocessing import prepare_image, detect_text_regions
from src.recognizer import HandwritingRecognizer

ROOT = Path(__file__).resolve().parent
ANNOTATIONS_CSV = ROOT / "data" / "RxHandBD" / "annotations.csv"

st.set_page_config(
    page_title="MedScript AI",
    page_icon="📝",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_recognizer():
    recognizer = HandwritingRecognizer(num_beams=3, max_new_tokens=64)
    return recognizer


@st.cache_data(show_spinner=False)
def load_annotation_vocabulary() -> tuple[str, ...]:
    if not ANNOTATIONS_CSV.is_file():
        return ()
    frame = pd.read_csv(ANNOTATIONS_CSV, encoding="utf-8-sig")
    if "text" not in frame.columns:
        return ()

    labels: dict[str, str] = {}
    for value in frame["text"].dropna():
        label = str(value).strip()
        if label:
            labels.setdefault(label.casefold(), label)
    return tuple(labels.values())


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def nearby_annotation_labels(
    prediction: str, vocabulary: tuple[str, ...]
) -> list[tuple[str, int]]:
    normalized = prediction.strip().casefold()
    if not normalized:
        return []

    matches: list[tuple[int, str, str]] = []
    for label in vocabulary:
        candidate = label.casefold()
        if candidate == normalized or abs(len(candidate) - len(normalized)) > 2:
            continue
        distance = _edit_distance(normalized, candidate)
        if 1 <= distance <= 2:
            matches.append((distance, candidate, label))

    return [
        (label, distance)
        for distance, _, label in sorted(matches)[:5]
    ]


if "medscript_results" not in st.session_state:
    st.session_state["medscript_results"] = None
if "current_image_hash" not in st.session_state:
    st.session_state["current_image_hash"] = None

st.title("MedScript AI")
st.subheader("Handwritten Medicine Name Recognition")
st.warning(
    "Research prototype only. Recognition may be incorrect. Do not use its output "
    "to prescribe, dispense, or administer medicines without qualified human verification."
)

uploaded_file = st.file_uploader(
    "Upload a handwritten medicine sheet",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(image_bytes).hexdigest()

    if st.session_state["current_image_hash"] != current_hash:
        st.session_state["medscript_results"] = None
        st.session_state["current_image_hash"] = current_hash

    original = ImageOps.exif_transpose(
        Image.open(io.BytesIO(image_bytes))
    ).convert("RGB")

    st.image(original, caption="Uploaded handwriting", use_container_width=True)

    st.markdown("### Detection settings")
    mode_label = st.selectbox(
        "Choose how to process the upload",
        [
            "Single word crop (recommended for RXHandBD images)",
            "Horizontal text lines (full-page heuristic)",
            "Connected text fragments (experimental)",
        ],
        help=(
            "Single-word mode sends the whole uploaded image to TrOCR, matching its "
            "word-image training data. Full-page modes use heuristic region detection "
            "and may split, merge, or miss text."
        ),
    )
    if mode_label.startswith("Single"):
        mode = "single"
    elif mode_label.startswith("Horizontal"):
        mode = "horizontal"
    else:
        mode = "connected"

    if st.button("Recognize handwriting", type="primary"):
        st.session_state["medscript_results"] = None

        try:
            with st.spinner("Preparing image and detecting text regions..."):
                recognizer = load_recognizer()
                st.session_state["recognizer_model_source"] = recognizer.model_source
                if mode == "single":
                    # Preserve the whole word image and match the training policy:
                    # EXIF correction followed by grayscale-to-RGB in the recognizer.
                    gray = np.asarray(original.convert("L"))
                    regions = [
                        {
                            "index": 1,
                            "bbox": (0, 0, original.width, original.height),
                            "image": gray,
                        }
                    ]
                else:
                    gray = prepare_image(original)
                    regions = detect_text_regions(gray, mode=mode)

                # Draw every detected region so missed/merged crops are visible.
                preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                for region in regions:
                    x1, y1, x2, y2 = region["bbox"]
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 180, 0), 2)

                with st.expander("Segmentation diagnostics", expanded=True):
                    st.write(f"Candidate regions detected: **{len(regions)}**")
                    st.image(gray, caption="Deskewed grayscale image", use_container_width=True)
                    if regions:
                        st.image(preview, caption="Detected-region bounding boxes", use_container_width=True)
                        columns = st.columns(2)
                        for idx, region in enumerate(regions):
                            crop_pil = Image.fromarray(region["image"]).convert("RGB")
                            with columns[idx % 2]:
                                st.image(
                                    crop_pil,
                                    caption=f"Candidate region {idx + 1} | bbox={region['bbox']}",
                                    use_container_width=True,
                                )

            if not regions:
                st.error(
                    "No candidate text regions were detected. The full page was NOT "
                    "sent to TrOCR because it is trained to recognize cropped text, "
                    "not to find text on a page. Try a clearer image or crop the text manually."
                )
            else:
                with st.spinner(f"Recognizing {len(regions)} candidate regions..."):
                    crop_images = [Image.fromarray(r["image"]).convert("RGB") for r in regions]
                    predictions = recognizer.recognize_many(crop_images, batch_size=1)

                vocabulary = load_annotation_vocabulary()
                results = []
                for region, crop_image, prediction in zip(regions, crop_images, predictions):
                    results.append(
                        {
                            "Region": region["index"],
                            "Bounding box": str(region["bbox"]),
                            "Image": crop_image,
                            "Recognized text": prediction,
                            "Annotation-label suggestions": nearby_annotation_labels(prediction, vocabulary),
                        }
                    )
                st.session_state["medscript_results"] = results

        except Exception as error:
            st.session_state["medscript_results"] = None
            st.error("Recognition failed. Check the terminal for details.")
            st.exception(error)

results = st.session_state["medscript_results"]
if results:
    st.divider()
    st.header("Recognition results")
    st.caption(
        "Nearby-label suggestions use the full annotation vocabulary, including "
        "validation/test labels. Use them for human review, not to score held-out accuracy."
    )
    model_source = st.session_state.get("recognizer_model_source")
    if model_source:
        st.caption(f"Model checkpoint: {model_source}")
    st.caption(
        "These are candidate text regions, not verified medicine names. "
        "Some crops may contain dosage or frequency instructions. Review and edit each result."
    )

    reviewed_rows = []
    for result in results:
        with st.container(border=True):
            st.markdown(f"**Candidate region {result['Region']}**")
            st.caption(f"Bounding box: {result['Bounding box']}")
            st.image(result["Image"], use_container_width=True)
            suggestions = result.get("Annotation-label suggestions", [])
            if suggestions:
                suggestion_text = ", ".join(
                    f"{label} ({distance} edit{'s' if distance != 1 else ''})"
                    for label, distance in suggestions
                )
                st.info(
                    "Nearby annotated labels to review: " + suggestion_text
                    + ". Compare them with the handwriting; suggestions are not applied automatically."
                )
            corrected_text = st.text_input(
                "Model prediction — edit after checking the original handwriting",
                value=result["Recognized text"],
                key=f"review_{st.session_state['current_image_hash']}_{result['Region']}",
            )
            reviewed_rows.append(
                {
                    "Region": result["Region"],
                    "Bounding box": result["Bounding box"],
                    "Model prediction": result["Recognized text"],
                    "Human-reviewed text": corrected_text,
                }
            )

    csv_data = pd.DataFrame(reviewed_rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download reviewed results (CSV)",
        data=csv_data,
        file_name="medscript_results.csv",
        mime="text/csv",
        key=f"download_{st.session_state['current_image_hash']}",
    )
