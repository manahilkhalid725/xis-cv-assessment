# Real-World Metric Measurement Pipeline from Calibrated Imagery

This repository contains the complete end-to-end computer vision pipeline for **intrinsic camera calibration**, **deep-learning semantic segmentation**, and **real-world metric measurement** (width and height in millimeters) of a custom target object (TV remote control) from raw imagery.

This repository contains my submission for the XIS AI & Computer Vision engineering technical assessment.

---

## Key System Capabilities
- **Sub-pixel Camera Calibration**: Calculates the intrinsic camera matrix ($K$) and lens distortion coefficients ($D$) using checkerboard images, achieving an excellent Mean Reprojection Error of **0.1933 pixels**.
- **AI-Assisted Labeling**: Automatically extracts binary ground-truth segmentation masks for the remote control using a pre-trained COCO Mask R-CNN model on undistorted images.
- **Custom Segmentation Model**: Trains a lightweight custom **U-Net** semantic segmentation model in PyTorch from scratch, achieving a Validation Intersection over Union (IoU) of **0.7001** on CPU.
- **Occlusion-Resistant Metric Scaling**: Derives the pixel-to-millimeter ratio dynamically on occluded backgrounds using Shi-Tomasi corner detection and pairwise distance histogram peak detection.
- **Metric Accuracy Validation**: Fit min-area bounding boxes to segmentations and validated against physical ground-truth measurements on 8 held-out test images, achieving a Mean Absolute Error of **38.67 mm (width)** and **52.50 mm (height)**. See `docs/MEASUREMENT_REPORT.md` for the full per-image error table and a discussion of the main error sources (segmentation incompleteness, perspective tilt, focus blur).

---

## Large File Hosting (Mandatory)

In compliance with assessment regulations (Section 2.2), all large datasets, calibration images, and weights are hosted on Google Drive. The GitHub repository remains lightweight.

- **Calibration Board Images**: [Google Drive Link - Calibration Images](https://drive.google.com/drive/folders/15DT0QDeWkSBMC1EWYlzoME2j4cLnO0eO?usp=drive_link)
- **Raw Target Dataset Images**: [Google Drive Link - Raw Images](https://drive.google.com/drive/folders/1QU95tKtk1VBPkyQnURSmYurN7Dw6o7GD?usp=drive_link)
- **Labeled Dataset Export (Images & Masks)**: [Google Drive Link - Labeled Dataset](https://drive.google.com/drive/folders/1dbNcYgPD0QlK1jtWLfTRe4F6FbRP3qwr?usp=drive_link)
- **Trained Model Weights (`best_model.pth`)**: [Google Drive Link - Trained Model Weights](https://drive.google.com/drive/folders/1IkgS37WiIKSzPJg1hLF4p-080oylgq9U?usp=drive_link)
- **Main Project Folder**: [Google Drive Link - Parent Directory](https://drive.google.com/drive/folders/1apJCUc11crEcya3Pveg1KBxtfB-w7d28?usp=drive_link)

---

## Quick-Start Guide

Get the pipeline up and running in minutes:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Intrinsic Camera Calibration
```bash
python calibration/calibrate.py
```

### 3. Generate Ground-Truth Masks & Split Dataset
```bash
python dataset/label_images.py
python dataset/prepare_dataset.py
```

### 4. Train the Custom U-Net Model
```bash
python models/train.py
```

### 5. Execute Metric Bounding Box Measurements & Validate Accuracy
```bash
python measurement/validate_accuracy.py
```
This validation script automatically evaluates predictions on the test partition, saves visualization overlays to `measurement/measured_*.png`, and saves metric statistics to `measurement/validation_table.md`.

---

## Repository Structure

```text
project-root/
│
├── calibration/              # Camera calibration code and params
│   ├── camera_params.json    # Saved intrinsics (K, D) and reprojection errors
│   ├── calibrate.py          # Main calibration calculation script
│   └── undistort.py          # Wrapper module to remove lens distortion
│
├── dataset/                  # Dataset preparation and split scripts
│   ├── label_images.py       # AI-assisted labeling via Mask R-CNN
│   ├── prepare_dataset.py    # Train/val/test data partitioner (70/20/10)
│   └── dataset_stats.py      # Computes statistics of the splits
│
├── models/                   # Custom PyTorch segmentation architecture
│   ├── unet.py               # Custom lightweight U-Net class implementation
│   ├── train.py              # PyTorch train loop and metric tracker
│   └── best_model.pth        # Trained weights (Hosted on Google Drive)
│
├── inference/                # Model prediction pipelines
│   └── predict.py            # Predicts segmentation mask on single new image
│
├── measurement/              # Metric scaling and accuracy validation
│   ├── measure.py            # Peak-based scale derivation & bounding boxes
│   ├── validate_accuracy.py  # Validation testing and MAE/MPE calculator
│   ├── validation_table.md   # Tabulated error report (MAE/MPE)
│   ├── validation_summary.json # Raw numerical errors and ratio configurations
│
├── docs/                     # Mandatory assessment reports
│   ├── CALIBRATION_REPORT.md # Calibration metrics & parameters
│   ├── DATASET_CARD.md       # Target choice and class distributions
│   ├── TRAINING_REPORT.md    # Model architecture and training configs
│   ├── MEASUREMENT_REPORT.md # Conversion math, error tables, limitations
│   └── SETUP.md              # Installation and run instructions
│
├── requirements.txt          # Python library dependencies
└── README.md                 # Project quick-start and Drive links
```

For detailed guides, please refer to the files in the [docs/](docs/) folder.
