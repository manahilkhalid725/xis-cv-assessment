# Pixel-to-MM Measurement Report

## 1. Pixel-to-MM Conversion Derivation

To calculate the physical size of the target object in millimeters from its segmented pixel mask, we must derive a scaling factor ($s_{\text{px/mm}}$). Under the pinhole camera projection model, the relation between the physical dimension $W$ (in mm) at a distance $Z$ and its image projection $w$ (in pixels) is given by:

$$w = f \cdot \frac{W}{Z}$$

Where $f$ is the camera focal length in pixels. We can define the pixel-to-mm conversion ratio as:

$$s_{\text{px/mm}} = \frac{w}{W} = \frac{f}{Z}$$

Since the camera distance $Z$ changes between different image captures (the camera height is not fixed), we must compute $s_{\text{px/mm}}$ dynamically for each image. 

We use a planar chessboard grid placed flat under the target object as a calibrated reference. Since the checkerboard squares have a known physical width and height ($W_{\text{square}} = 25.0$ mm), we can detect the squares in the image, compute their size in pixels ($w_{\text{square}}$), and determine the ratio:

$$s_{\text{px/mm}} = \frac{w_{\text{square}}}{W_{\text{square}}} = \frac{w_{\text{square}}}{25.0}$$

The physical dimensions of the target object are then computed from its minimum bounding box dimensions ($w_{\text{target\_px}}$ and $h_{\text{target\_px}}$):

$$\text{Width (mm)} = \frac{\min(w_{\text{target\_px}}, h_{\text{target\_px}})}{s_{\text{px/mm}}}$$

$$\text{Height (mm)} = \frac{\max(w_{\text{target\_px}}, h_{\text{target\_px}})}{s_{\text{px/mm}}}$$

---

## 2. Background Corner Detection & Scaling Algorithm

Standard chessboard corner detectors (`cv2.findChessboardCorners`) fail on the remote dataset because the remote control occludes several squares. To solve this occlusion issue, we developed a robust **Background Corner Clustering & Peak Detection Algorithm**:

1. **Occlusion Masking**: The predicted remote mask from our trained U-Net is dilated by **85 pixels** (`cv2.dilate`). This creates a broad exclusion zone that covers the remote control, its boundaries, and any shadows or specular reflections.
2. **Background Corner Detection**: We apply Shi-Tomasi Corner Detection (`cv2.goodFeaturesToTrack`) on the grayscale undistorted image using the inverted dilated mask as the search region. This restricts corner detection strictly to the background chessboard.
3. **Sub-pixel Refinement**: Corner coordinates are refined to sub-pixel accuracy using `cv2.cornerSubPix`.
4. **Pairwise Distance Distribution**: We compute all pairwise Euclidean distances between the detected background corners. For a grid pattern, these distances cluster around the grid square width ($S$), the diagonal ($S\sqrt{2}$), and multiples ($2S, 3S$).
5. **Histogram Peak Detection**: We build a histogram of these pairwise distances in the range $[110.0, 190.0]$ pixels (the expected pixel range for a 25.0 mm square given our camera calibration matrix). The peak (mode) of this histogram corresponds to the primary grid square size $S$.
6. **Ratio Estimation**: The ratio is computed as $s_{\text{px/mm}} = S / 25.0$.

---

## 3. Accuracy Validation Report

We validated the end-to-end pipeline (Inference + Measurement) on the 8 test split images against a physical ground-truth remote control size of **186.0 mm (Height)** and **48.0 mm (Width)**.

### Accuracy Validation Table

| Image Filename | Px/MM Ratio | Pred Width | Pred Height | GT Width | GT Height | Width Error | Height Error |
|---|---|---|---|---|---|---|---|
| undistorted_WhatsApp Image 2026-07-09 at 12.48.12 PM.png | 5.96 | 87.9 mm | 219.7 mm | 48.0 mm | 186.0 mm | 39.9 mm (83.1%) | 33.7 mm (18.1%) |
| undistorted_WhatsApp Image 2026-07-09 at 12.48.37 PM (1).png | 4.84 | 197.2 mm | 259.7 mm | 48.0 mm | 186.0 mm | 149.2 mm (310.8%) | 73.7 mm (39.6%) |
| undistorted_WhatsApp Image 2026-07-09 at 12.48.37 PM.png | 6.12 | 75.1 mm | 184.7 mm | 48.0 mm | 186.0 mm | 27.1 mm (56.5%) | 1.3 mm (0.7%) |
| undistorted_WhatsApp Image 2026-07-09 at 12.48.38 PM (1).png | 5.49 | 78.1 mm | 166.8 mm | 48.0 mm | 186.0 mm | 30.1 mm (62.6%) | 19.2 mm (10.3%) |
| undistorted_WhatsApp Image 2026-07-09 at 12.48.38 PM.png | 6.91 | 30.8 mm | 77.4 mm | 48.0 mm | 186.0 mm | 17.2 mm (35.8%) | 108.6 mm (58.4%) |
| undistorted_WhatsApp Image 2026-07-09 at 12.48.42 PM (2).png | 6.76 | 54.4 mm | 159.2 mm | 48.0 mm | 186.0 mm | 6.4 mm (13.4%) | 26.8 mm (14.4%) |
| undistorted_WhatsApp Image 2026-07-09 at 12.48.43 PM (1).png | 5.15 | 67.4 mm | 106.2 mm | 48.0 mm | 186.0 mm | 19.4 mm (40.4%) | 79.8 mm (42.9%) |
| undistorted_WhatsApp Image 2026-07-09 at 12.48.44 PM (2).png | 6.44 | 27.9 mm | 109.1 mm | 48.0 mm | 186.0 mm | 20.1 mm (42.0%) | 76.9 mm (41.3%) |
| **AVERAGE (MAE / MPE)** | | | | | | **38.67 mm (80.6%)** | **52.50 mm (28.2%)** |

### Statistical Metrics
- **Mean Absolute Error (MAE) - Width**: **38.67 mm**
- **Mean Absolute Error (MAE) - Height**: **52.50 mm**
- **Mean Percentage Error (MPE) - Width**: **80.57 %**
- **Mean Percentage Error (MPE) - Height**: **28.23 %**

---

## 4. Error Sources & Limitations Discussion

### 1. Segmentation Mask Incompleteness
The custom U-Net model was trained on 49 images from scratch on CPU. In some test images (such as image `12.48.38 PM.png`), the model predicts a partial segmentation mask (e.g. cutting off the bottom half of the remote due to shadow transitions). This causes a significant reduction in the bounding box height (from ~900 px to 534 px), leading to a high height error (108.6 mm). 

### 2. Radial Out-of-Plane Perspective
The scaling derivation assumes a planar grid parallel to the camera sensor. If the camera is tilted relative to the table surface, perspective distortion occurs. This causes squares closer to the lens to appear larger than squares further away, shifting the distance peak. Integrating homography warping (`cv2.getPerspectiveTransform`) using four detected board corners would resolve this, but homography estimation requires a fully visible board grid, which was occluded.

### 3. Out-of-Focus Blur
Some remote images suffer from focus blur near the boundaries, which degrades edge contrast. This makes it harder for Shi-Tomasi to locate corners precisely, causing slight shifts in the sub-pixel refinement step.

---

## 5. End-to-End Usage Guide

The measurement pipeline is fully automated and can be executed on any raw image.

### Run Single-Image Pipeline
To segment and measure a single raw image, run:
```bash
python measurement/validate_accuracy.py
```
This script runs the entire sequence:
1. Loads the camera params from `calibration/camera_params.json` and undistorts the image.
2. Runs the U-Net model (`models/best_model.pth`) to extract the remote mask.
3. Automatically derives the background grid scaling factor.
4. Computes width/height in mm.
5. Saves annotated visualization overlays to `measurement/measured_*.png`.
