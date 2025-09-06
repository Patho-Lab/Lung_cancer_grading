# Deep Learning-Powered Virtual Elastin Staining for Lung Adenocarcinoma Grading

This repository contains the official PyTorch implementation for the models described in the paper: **"Deep Learning-Powered Virtual Elastin Staining Improves Objective Grading and Prognostic Stratification of Lung Adenocarcinoma."**

Our work introduces a deep learning framework to improve the objectivity and accuracy of lung adenocarcinoma grading. By computationally generating virtual elastin fluorescence images from standard H&E slides, our model can better distinguish between non-invasive and invasive histological patterns, leading to more reliable prognostic grading.

[Link to Full Paper](#) <!-- Add a link to your publication here -->

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

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/virtual-elastin-staining.git
    cd virtual-elastin-staining
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You will need to create a `requirements.txt` file listing all dependencies, e.g., `torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`, etc.)*

## Usage

### 1. Data Preparation

The models expect image tiles extracted from co-registered whole-slide images. You will need to prepare your data in the following structure:
/path/to/data/
├── train/
│ ├── he/
│ │ ├── tile_001.png
│ │ ├── tile_002.png
│ │ └── ...
│ └── ebef/
│ ├── tile_001.png
│ ├── tile_002.png
│ └── ...
└── validation/
├── he/
│ ├── tile_101.png
│ └── ...
└── ebef/
├── tile_101.png
└── ...
code
Code
-   **Image Format:** Paired H&E and EBEF tiles must have identical filenames.
-   **Image Size:** The models were trained on 512x512 pixel tiles.
-   **Ground Truth:** Training and validation labels should be organized in a CSV file or inferred from the folder structure, depending on your data loader implementation.

### 2. Training the Models

You can train each model using the scripts provided in their respective directories.

#### H&E-Only Model
```bash
python HE/train.py \
    --data_dir /path/to/data/ \
    --he_dir_train train/he/ \
    --he_dir_val validation/he/ \
    --epochs 150 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir ./models/he_only/
EBEF-Only Model
code
Bash
python Fluo/train.py \
    --data_dir /path/to/data/ \
    --ebef_dir_train train/ebef/ \
    --ebef_dir_val validation/ebef/ \
    --epochs 150 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir ./models/ebef_only/
Dual-Stream Model
code
Bash
python Dual/train.py \
    --data_dir /path/to/data/ \
    --he_dir_train train/he/ \
    --he_dir_val validation/he/ \
    --ebef_dir_train train/ebef/ \
    --ebef_dir_val validation/ebef/ \
    --epochs 150 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir ./models/dual_stream/
3. Inference
Use the inference scripts to classify new image tiles using a trained model checkpoint.
Dual-Stream Inference Example
code
Bash
python Dual/inference.py \
    --he_tile /path/to/new_he_tile.png \
    --ebef_tile /path/to/new_ebef_tile.png \
    --model_weights ./models/dual_stream/best_model.pth
For the practical workflow described in the paper, the --ebef_tile would be a virtual EBEF image generated by a pre-trained GAN (e.g., Pix2Pix-HD).
Performance Highlights
Our study demonstrates the superiority of the dual-stream approach, especially when using high-fidelity virtual EBEF images.
Tile-Level Classification Accuracy (Real EBEF):
H&E + EBEF (Dual): 89.3%
H&E-Only: 87.8%
EBEF-Only: 84.0%
Automated Grading Accuracy (Virtual EBEF):
H&E + Pix2Pix-HD: 94.2%
H&E-Only: 79.6%
Prognostic Value: The AI-assigned grades significantly stratified patient survival in the independent TCGA-LUAD cohort (log-rank p = 0.047).
Citation
If you use this code or our findings in your research, please cite our paper:
code
Bibtex
@article{your_study_2024,
  title   = {Deep Learning-Powered Virtual Elastin Staining Improves Objective Grading and Prognostic Stratification of Lung Adenocarcinoma},
  author  = {Author, A. and Author, B. and et al.},
  journal = {Journal Name},
  year    = {2024},
  volume  = {XX},
  pages   = {XX-XX}
}
License
This project is licensed under the MIT License. See the LICENSE file for details.
