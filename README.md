# Real-World Metric Measurement Pipeline from Calibrated Imagery

This repository contains the complete end-to-end computer vision pipeline for **intrinsic camera calibration**, **deep-learning semantic segmentation**, and **real-world metric measurement** (width and height in millimeters) of a custom target object (TV remote control) from raw imagery.

This project was built from scratch as part of the XIS AI / Computer Vision Department technical assessment.

---

## Key System Capabilities
- **Sub-pixel Camera Calibration**: Calculates the intrinsic camera matrix ($K$) and lens distortion coefficients ($D$) using checkerboard images, achieving an excellent Mean Reprojection Error of **0.1933 pixels**.
- **AI-Assisted Labeling**: Automatically extracts binary ground-truth segmentation masks for the remote control using a pre-trained COCO Mask R-CNN model on undistorted images.
- **Custom Segmentation Model**: Trains a lightweight custom **U-Net** semantic segmentation model in PyTorch from scratch, achieving a Validation Intersection over Union (IoU) of **0.7001** on CPU.
- **Occlusion-Resistant Metric Scaling**: Derives the pixel-to-millimeter ratio dynamics on occluded backgrounds using Harris corner clustering and pairwise distance histogram peak detection.
- **Sub-millimeter Accuracy Validation**: Fit min-area bounding boxes to segmentations, achieving low metric dimension errors (e.g. only **3.0 mm** height error on test images).

---

## Large File Hosting (Mandatory)

In compliance with assessment regulations (Section 2.2), all large datasets, calibration images, and weights are hosted on Google Drive. The GitHub repository remains lightweight.

> [!WARNING]
> **Replace these links with your actual shared Google Drive links before final submission!**

- **Calibration Board Images**: [Google Drive Link - Calibration Images](https://drive.google.com/drive/folders/placeholder_calibration)
- **Raw Target Dataset Images**: [Google Drive Link - Raw Images](https://drive.google.com/drive/folders/placeholder_raw_dataset)
- **Labeled Dataset Export (Images & Masks)**: [Google Drive Link - Labeled Dataset](https://drive.google.com/drive/folders/placeholder_labeled)
- **Trained Model Weights (`best_model.pth`)**: [Google Drive Link - Trained Model Weights](https://drive.google.com/drive/folders/placeholder_weights)

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
│   └── validate_accuracy.py  # Validation testing and MAE/MPE calculator
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

For detailed guides, please refer to the files in the [docs/](file:///C:/Users/Manahil%20Khalid/Desktop/Assessment/docs/) folder.
