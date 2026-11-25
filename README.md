# 💊 MedScript AI – Handwritten Medicine Recognition

MedScript AI is an AI-powered research prototype designed to recognize handwritten medicine names from prescription images using a fine-tuned **TrOCR (Transformer-based Optical Character Recognition) model**.

The project aims to simplify the interpretation of handwritten medicine names by combining deep learning-based handwriting recognition with image processing and a manual review interface.

The model is fine-tuned using the RxHandBD handwritten medicine dataset. Training, dataset preparation, and model evaluation are performed using Google Colab with GPU acceleration, while the local Streamlit application provides an interface for uploading prescription images, viewing candidate text regions, and reviewing OCR predictions.

> **Research Prototype Disclaimer:** This application is intended strictly for research and educational purposes. OCR predictions may contain plausible but incorrect medicine names. Every result must be independently verified by a qualified healthcare professional against the original prescription. This application is not a clinical decision-support system.

---

## 🌐 Project Information

- **Project Name:** MedScript AI
- **Project Type:** AI / Deep Learning / OCR
- **Primary Model:** Fine-tuned TrOCR
- **Dataset:** RxHandBD
- **Training Platform:** Google Colab (GPU)
- **Local Interface:** Streamlit
- **Model Training Success Rate:** 68%

### 📄 Project Results & Documentation

- **Project Trained Model and ImageSet folder:** [View MedScript AI Training model files and the zip of the source data images](https://drive.google.com/drive/folders/1Iuwi0Bb5CuypzYjrXAQQc9X_tFZSk7Qo?usp=sharing)
- **Project Result:**[View MedScript AI Project Results](https://docs.google.com/document/d/12KzcMEfDO3_eaiDxwUFsR-CSiU2tPjw1/edit?usp=sharing&ouid=112150240852770674706&rtpof=true&sd=true)

<blockquote>
The reported model training success rate is 68%. This is a project-reported training result and should not be interpreted as independently verified clinical accuracy or guaranteed medicine recognition accuracy.
</blockquote>

---

## ✨ Key Features

- **Handwritten Medicine Recognition:** Recognizes handwritten medicine names using a fine-tuned TrOCR model.
- **Transformer-Based OCR:** Uses Microsoft's TrOCR handwritten recognition architecture.
- **Custom Model Fine-Tuning:** Fine-tunes a pretrained model using the RxHandBD medicine handwriting dataset.
- **Dataset Validation:** Validates image files and annotations before model training.
- **Duplicate Image Detection:** Identifies byte-identical images to reduce data leakage between dataset splits.
- **Dataset Partitioning:** Creates training, validation, and testing datasets.
- **GPU-Based Training:** Uses Google Colab GPU resources for computationally intensive model training.
- **Model Evaluation:** Evaluates recognition performance using Character Error Rate (CER) and Exact Match Accuracy.
- **Candidate Region Detection:** Identifies potential handwriting regions in uploaded images using heuristic image processing.
- **Interactive Streamlit Interface:** Allows users to upload prescription images and inspect candidate text regions.
- **Manual Prediction Review:** Allows users to review and edit OCR-generated predictions.
- **Local Model Inference:** Supports running the trained model locally without requiring a GPU.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| OCR Architecture | Microsoft TrOCR |
| Transformer Framework | Hugging Face Transformers |
| Image Processing | OpenCV, Pillow |
| Data Processing | Pandas, NumPy |
| Model Evaluation | Character Error Rate (CER), Exact Match Accuracy |
| Web Interface | Streamlit |
| Training Environment | Google Colab |
| Hardware Acceleration | CUDA / GPU |
| Dataset | RxHandBD |
| Version Control | Git and GitHub |

---

## 🧠 Model Architecture

MedScript AI uses Microsoft's pretrained **TrOCR Base Handwritten** model as the foundation for handwritten medicine recognition.

**Base Model:** `microsoft/trocr-base-handwritten`

The pretrained model is fine-tuned using annotated medicine handwriting images to adapt its recognition capabilities to medicine names.

### Model Workflow

1. Input prescription or handwritten medicine image.
2. Image preprocessing and candidate handwriting region detection.
3. Extract candidate image regions.
4. Pass the image regions to the fine-tuned TrOCR model.
5. Generate predicted medicine names.
6. Display predictions in the Streamlit interface.
7. Allow manual review and correction.

### Training and Inference

- Training is performed in Google Colab using GPU acceleration.
- The trained checkpoint is saved in `models/trocr-medicine/`.
- The local application loads the fine-tuned checkpoint when available.
- If a fine-tuned model is unavailable, the application can use the original pretrained TrOCR model.

---

## 📊 Model Training and Results

The model is trained using annotated handwritten medicine images from the RxHandBD dataset.

**Reported Model Training Success Rate: 68%**

The training pipeline includes:

- Dataset annotation validation.
- Image-file verification.
- Duplicate-image detection.
- Training, validation, and testing split generation.
- TrOCR fine-tuning.
- Validation-based model evaluation.
- Final held-out test evaluation.

### Evaluation Metrics

| Metric | Description |
|---|---|
| Character Error Rate (CER) | Measures character-level differences between predicted and actual medicine names. |
| Exact Match Accuracy | Measures the percentage of predictions that exactly match the ground-truth medicine names. |
| Training Success Rate | Project-reported training result of 68%. |

**Important:** The 68% training success rate is a reported project result. It should not be confused with validation exact-match accuracy or final test accuracy. The final held-out test results should be reported separately.

---

## 🚀 Getting Started

The project has two main workflows:

1. Model training and evaluation using Google Colab.
2. Local inference using the Streamlit application.

### Part 1: Train the Model Using Google Colab

#### Prerequisites

- Google account.
- Google Colab.
- GPU-enabled runtime.
- `MedScript_Colab_Training_fixed.zip` training archive.
- Google Drive storage.

#### Step 1: Upload the Training Archive

Upload `MedScript_Colab_Training_fixed.zip` to the root directory of your Google Drive.

#### Step 2: Enable GPU Runtime

Open Google Colab and select:

Runtime → Change runtime type → T4 GPU (or another available GPU).

#### Step 3: Mount Google Drive and Extract the Project

Run the following cell:

```python
from google.colab import drive
from pathlib import Path
from zipfile import ZipFile

drive.mount("/content/drive")

zip_path = Path(
    "/content/drive/MyDrive/MedScript_Colab_Training_fixed.zip"
)

if not zip_path.is_file():
    raise FileNotFoundError(
        f"Training ZIP not found: {zip_path}"
    )

with ZipFile(zip_path) as archive:
    archive.extractall("/content")

%cd /content/MedScript-Colab

%pip install -r requirements-colab.txt
```

#### Step 4: Verify GPU Availability

```python
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "Enable GPU runtime in Google Colab."
    )

print("GPU:", torch.cuda.get_device_name(0))
```

#### Step 5: Prepare the Dataset and Train the Model

```python
import subprocess
import sys

# Validate annotations and create dataset splits
subprocess.run(
    [sys.executable, "split_dataset.py"],
    check=True
)

# Fine-tune TrOCR
subprocess.run(
    [sys.executable, "train_trocr.py"],
    check=True
)
```

The training process generates the fine-tuned model checkpoint inside:

`models/trocr-medicine/`

#### Step 6: Evaluate the Model

Evaluate the model using the validation dataset:

```python
subprocess.run([
    sys.executable,
    "evaluate_trocr.py",
    "--csv", "data/splits/validation.csv",
    "--output", "validation_predictions.csv",
    "--num-beams", "3"
], check=True)
```

The evaluation generates predictions and calculates CER and Exact Match Accuracy.

#### Step 7: Save Training Results to Google Drive

The trained model, dataset splits, and evaluation outputs can be saved to Google Drive for future use.

```python
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import shutil

model_source = Path("models/trocr-medicine")

if not (model_source / "model.safetensors").is_file():
    raise FileNotFoundError(
        "Trained model file not found."
    )

run_name = datetime.now(
    ZoneInfo("Asia/Kolkata")
).strftime("run_%Y%m%d_%H%M")

save_dir = (
    Path("/content/drive/MyDrive/MedScript-Training-Results")
    / run_name
)

save_dir.mkdir(parents=True, exist_ok=True)

shutil.copytree(
    model_source,
    save_dir / "trocr-medicine",
    dirs_exist_ok=True
)

shutil.copytree(
    "data/splits",
    save_dir / "splits",
    dirs_exist_ok=True
)

for filename in (
    "training_metrics.csv",
    "validation_predictions.csv"
):
    source = Path(filename)

    if source.is_file():
        shutil.copy2(source, save_dir / filename)

print("Training results saved to:", save_dir)
```

#### Step 8: Final Test Evaluation

After completing model selection and validation, evaluate the final model using the untouched test dataset.

```python
subprocess.run([
    sys.executable,
    "evaluate_trocr.py",
    "--csv", "data/splits/test.csv",
    "--output", "test_predictions.csv",
    "--num-beams", "3"
], check=True)
```

Do not use test results to tune the model.

---

### Part 2: Run the Local Streamlit Application

#### Prerequisites

- Python 3.11 or Python 3.12.
- Git.
- Trained TrOCR model (recommended).
- Minimum 8 GB RAM recommended for the local CPU inference environment.

#### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MedScript-AI
```

Replace the repository URL and directory name with the actual GitHub repository details.

#### Step 2: Create a Virtual Environment

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1
```

#### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install CPU-based PyTorch:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

#### Step 4: Add the Trained Model

Copy the complete trained model and processor directory from Google Drive into:

```text
models/
└── trocr-medicine/
    ├── config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    ├── tokenizer_config.json
    └── ...
```

The folder should contain the complete model and processor files generated during training.

#### Step 5: Launch the Application

```bash
streamlit run app.py
```

Open the application in your browser using the local Streamlit URL, usually:

`http://localhost:8501`

#### Step 6: Upload and Review Prescription Images

1. Upload a handwritten prescription image.
2. Inspect the candidate handwriting regions.
3. Review the OCR-generated medicine names.
4. Correct any incorrectly recognized text manually.
5. Verify every result against the original prescription with a qualified healthcare professional.

---

## 📁 Project Structure

```text
MedScript-AI/
│
├── app.py
├── train_trocr.py
├── split_dataset.py
├── evaluate_trocr.py
│
├── requirements.txt
├── requirements-colab.txt
│
├── data/
│   ├── RxHandBD/
│   │   ├── annotations.csv
│   │   └── Train_Set/
│   │
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── duplicate_resolution.csv
│
├── models/
│   └── trocr-medicine/
│       ├── config.json
│       ├── model.safetensors
│       └── processor files
│
└── README.md
```

### Important Project Components

| File / Directory | Purpose |
|---|---|
| `app.py` | Streamlit interface for image upload, candidate-region detection, OCR predictions, and manual review. |
| `train_trocr.py` | Fine-tunes the pretrained TrOCR model using GPU acceleration. |
| `split_dataset.py` | Validates annotations, identifies exact-image duplicates, and creates dataset partitions. |
| `evaluate_trocr.py` | Calculates CER and Exact Match Accuracy and generates prediction reports. |
| `data/RxHandBD/` | Contains annotated handwritten medicine images and their corresponding labels. |
| `data/splits/` | Stores training, validation, and test CSV files and duplicate-resolution records. |
| `models/trocr-medicine/` | Stores the fine-tuned TrOCR model and processor files. |

---

## 📂 Dataset Information

The project uses the RxHandBD handwritten medicine dataset.

The annotation file follows this format:

```csv
image_path,text
data/RxHandBD/Train_Set/P2509.jpg,Calbo-D
data/RxHandBD/Train_Set/P1282.jpg,Algin
data/RxHandBD/Train_Set/P0692.jpg,Neuroxen
```

### Dataset Preparation

The dataset preparation pipeline performs:

- Annotation format validation.
- Missing-image detection.
- Image file verification.
- Exact byte-identical image detection.
- Duplicate-resolution handling.
- Training, validation, and testing partition generation.

Same-label duplicates are retained in only one split. Identical images with conflicting labels are excluded for manual review.

Duplicate-resolution decisions are recorded in:

`data/splits/duplicate_resolution.csv`

**Dataset limitation:** The current annotations do not contain writer IDs or prescription IDs. Therefore, the generated splits cannot guarantee writer-independent evaluation.

---

## ⚙️ Model Evaluation and Limitations

Although TrOCR provides a strong pretrained handwriting-recognition foundation, handwritten medicine recognition remains challenging.

Some of the key challenges include:

- Different handwriting styles between individuals.
- Similar-looking medicine names.
- Inconsistent handwriting size and alignment.
- Poor image quality and lighting.
- Overlapping or closely spaced handwritten text.
- Variations in spelling and character formation.
- Difficulty identifying medicine names from complex prescription layouts.

The candidate-region detection component currently uses heuristic image processing. It may miss handwriting regions or group unrelated text together.

The recognition output must therefore be treated as a prediction rather than a confirmed medicine name.

---

## 🔮 Future Improvements

- Improve recognition accuracy through additional domain-specific training data.
- Explore handwriting augmentation techniques.
- Improve image preprocessing and handwriting segmentation.
- Support complete prescription-level text extraction.
- Improve recognition of multiple medicine names from a single image.
- Experiment with alternative OCR architectures.
- Introduce confidence-based candidate ranking.
- Explore dictionary-assisted medicine-name verification without automatically overriding model predictions.
- Improve model generalization across different handwriting styles.
- Develop a more comprehensive evaluation dataset with writer-independent partitions.
- Improve the Streamlit interface and manual correction workflow.

---

## 🛡️ Safety and Security

- This is a research and educational prototype.
- OCR-generated medicine names may be incorrect.
- Never use predictions as the sole basis for medication decisions.
- Every prediction must be verified against the original prescription by a qualified healthcare professional.
- Do not upload sensitive patient prescriptions to untrusted environments.
- Protect patient information and training datasets.
- Do not interpret the reported 68% training success rate as clinical reliability.

---

## 👨‍💻 Author

**SalmanSaalu**

GitHub: https://github.com/SalmanSaalu

---

If you find this project interesting, consider starring the repository and sharing feedback or suggestions for improvement.
