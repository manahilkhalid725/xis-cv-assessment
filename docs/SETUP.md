# Installation and Setup Guide

This document provides complete instructions for setting up the environment and executing the end-to-end computer vision pipeline.

## 1. Prerequisites
- **Python**: Version `3.10` or higher (`3.12.6` was used during development).
- **Git**: Installed and configured on your system path.
- **Operating System**: Windows / Linux / macOS (tested on Windows 11).

---

## 2. Installation Steps

### Step 2.1: Clone the Repository
Clone the repository to your local system and navigate to the project root folder:
```bash
git clone https://github.com/manahilkhalid725/xis-cv-assessment.git
cd xis-cv-assessment
```

### Step 2.2: Create and Activate Virtual Environment (Recommended)
Creating a clean virtual environment prevents package conflicts:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (Command Prompt)
venv\Scripts\activate

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source venv/bin/activate
```

### Step 2.3: Install Dependencies
Install all required Python libraries using pip:
```bash
pip install -r requirements.txt
```

---

## 3. Running the Pipeline

The pipeline consists of three sequential steps, each relying on the output of the previous:

### Step 1: Camera Calibration
Executes intrinsic calibration on the chessboard images in `Dataset/Board`.
```bash
python calibration/calibrate.py
```
- *Input*: `Dataset/Board/`
- *Output*: `calibration/camera_params.json` (intrinsic matrix and distortion parameters).
- *Visual Verification*: `calibration/debug_corners/` (saves images displaying detected and refined sub-pixel corners).

---

### Step 2: Dataset Labeling & Model Training
Auto-labels the raw remote images using Mask R-CNN, partitions the dataset, and trains our custom U-Net model.

#### 1. Generate Ground-Truth Masks:
```bash
python dataset/label_images.py
```
- *Input*: `Dataset/Remote/` and `calibration/camera_params.json`
- *Output*: Saves undistorted images in `dataset/processed/images/` and clean binary masks in `dataset/processed/masks/`.

#### 2. Partition Dataset (70/20/10 Split):
```bash
python dataset/prepare_dataset.py
```
- *Input*: `dataset/processed/`
- *Output*: Prepares training, validation, and testing subfolders in `dataset/split/`.

#### 3. Train the Custom U-Net Model:
```bash
python models/train.py
```
- *Input*: `dataset/split/`
- *Output*: Saves the best model weights to `models/best_model.pth` and visualizes performance trends in `models/training_curves.png`.

---

### Step 3: Accuracy Validation & Measurement
Runs inference on the test dataset and calculates real-world measurements using background corner clustering.
```bash
python measurement/validate_accuracy.py
```
- *Input*: `dataset/split/test/` and `models/best_model.pth`
- *Output*: Computes metric sizes and prints the validation summary (Mean Absolute Error and Mean Percentage Error).
- *Visual overlays*: `measurement/measured_*.png` (annotated test images displaying bounding boxes and millimeter dimensions).
- *Tabulated results*: Saves `measurement/validation_table.md` and `measurement/validation_summary.json`.
