# Deep Learning-Powered Virtual Elastin Staining for Lung Adenocarcinoma Grading

This repository contains the official PyTorch implementation for the models described in the paper: **"Deep Learning-Powered Virtual Elastin Staining Improves Objective Grading and Prognostic Stratification of Lung Adenocarcinoma."**

Our work introduces a deep learning framework to improve the objectivity and accuracy of lung adenocarcinoma grading. By computationally generating virtual elastin fluorescence images from standard H&E slides, our model can better distinguish between non-invasive and invasive histological patterns, leading to more reliable prognostic grading.


## Abstract

> The histological grading of lung adenocarcinoma is critical for predicting patient prognosis and guiding therapy, yet it is limited by significant inter-observer variability. Distinguishing non-invasive lepidic patterns from invasive subtypes is a primary diagnostic challenge, hinging on the visualization of the underlying elastic fiber framework, which is often unclear in standard hematoxylin and eosin (H&E) stains. We developed a deep learning framework to improve grading objectivity by computationally visualizing elastic fibers from routine H&E slides. Our results show that integrating H&E with high-fidelity synthesized elastin fluorescence images achieves a final grading accuracy of 94.2%, substantially exceeding the 79.6% accuracy of an H&E-only model. When applied to The Cancer Genome Atlas (TCGA), the AI-assigned grades significantly stratified overall patient survival (log-rank p = 0.047), demonstrating the model's prognostic relevance.

![Figure2](https://github.com/user-attachments/assets/407467e7-cef2-42c8-aea8-398c8b268b82)
Figure 2 from the paper, illustrating the multi-modal deep learning framework.

## Key Features

- **Virtual Staining:** Leverages a Generative Adversarial Network (Pix2Pix-HD) to synthesize high-fidelity eosin-based elastin fluorescence (EBEF) images from standard H&E slides.
- **Multi-modal Classification:** A dual-stream deep learning model that integrates H&E and EBEF modalities to accurately classify seven key histological patterns of lung adenocarcinoma.
- **Automated Grading:** A fully automated pipeline that quantifies pattern proportions from whole-slide images to assign a prognostic grade (Grade 1, 2, or 3).
- **Benchmarking:** Includes code for single-modality (H&E-only, EBEF-only) models to benchmark and demonstrate the performance gain from multi-modal fusion.

## Repository Structure

This repository is organized into three main directories, each containing the code for a specific model architecture discussed in the paper.

| Directory | Description                                                                                             |
| :-------- | :------------------------------------------------------------------------------------------------------ |
| **/HE/**  | Contains training and inference scripts for the **H&E-only** single-modality model.                   |
| **/Fluo/**| Contains training and inference scripts for the **EBEF-only** (fluorescence) single-modality model.     |
| **/Dual/**| Contains training and inference scripts for the **dual-stream** model that fuses H&E and EBEF inputs. |

*Note: The code for the Pix2Pix-HD image-to-image translation model used to generate virtual EBEF images is not included in this repository but can be implemented using publicly available frameworks.*

## Getting Started

### Prerequisites

- Python 3.11+
- PyTorch 2.3+
- A CUDA-enabled GPU is highly recommended for training (e.g., NVIDIA RTX 4090 as used in the study).

## Usage

This repository contains scripts for three distinct models: H&E-only, Fluorescence-only, and the dual-stream H&E+Fluorescence model. All scripts are configured by editing variables directly in the Python files rather than using command-line arguments. The usage instructions for each are detailed below.

---

### H&E-Only Model (`HE/`)

This is the baseline model trained and evaluated exclusively on standard Hematoxylin and Eosin (H&E) stained images.

#### 1. Data Preparation

The `HE` model scripts expect different data structures for training and inference.

*   **For Training:** Data must be organized into class subdirectories.
    ```
    /path/to/he_training_data/
    ├── class_a/
    │   ├── tile_001.png
    │   └── ...
    └── class_b/
        └── ...
    ```

*   **For Inference:** Tiles should be grouped into subfolders, where each subfolder represents a single Whole-Slide Image (WSI).
    ```
    /path/to/he_inference_data/
    ├── WSI_01/
    │   ├── tile_001.png
    │   └── ...
    └── WSI_02/
        └── ...
    ```

#### 2. Training

**Step 1: Configure `HE/train.py`**
Open the script and modify the path variables and hyperparameters.

```python
# --- HE/train.py ---
if __name__ == "__main__":
    TRAIN_IMAGE_PATH = "/path/to/he_training_data"
    VAL_IMAGE_PATH = "/path/to/he_validation_data"
    CHECKPOINT_DIR = "checkpoints_he"
    # ... other settings ...
