# Dataset Card

## 1. Object Selection
The target object chosen for the segmentation and measurement pipeline is a **TV Remote Control**.

### Justification
- **Geometry**: The remote control is flat, rigid, and has a well-defined rectangular geometry. This structure makes it ideal for 2D bounding box fitting and metric measurement verification.
- **Availability**: A TV remote is universally available, enabling easy reproducibility of the setup.
- **Labeling Ease**: It has high-contrast boundaries against typical table or chessboard surfaces, allowing clean automated segmentation and manual validation.
- **Physical Reference Dimensions**: 
  - Height: **186.0 mm**
  - Width: **48.0 mm**

---

## 2. Data Collection Strategy
- **Camera Setup**: Captured using a smartphone camera.
- **Environment**: The remote control was placed at different positions, angles, and orientations on top of a standard A4 chessboard calibration grid. 
- **Image Diversity**:
  - Varied lighting conditions (direct light, indoor ambient light, shadows).
  - Varied heights (scale changes).
  - Varied angles of rotation.
  - Varied background positions relative to the checkerboard pattern.
- **Total Captured Images**: 71 images of the remote control.

---

## 3. Labeling Tool & Processing Pipeline
To eliminate human bias and ensure pixel-level accuracy, we developed a semi-automated **AI-Assisted Labeling Pipeline** (`dataset/label_images.py`):
1. **Lens Undistortion**: Every raw captured image is first undistorted using the intrinsic calibration parameters computed from Step 1 (`cv2.undistort`).
2. **COCO Model Inference**: The undistorted image is run through a pre-trained **Mask R-CNN ResNet-50 FPN** model (from `torchvision`), which was trained on the COCO dataset and includes the `remote` class (class ID 75).
3. **Confidence Filtering**: Only remote detections with a confidence score $>0.40$ are processed.
4. **Post-Processing**:
   - The raw mask probability map is thresholded at 0.5 to form a binary mask.
   - Morphological operations (ellipse kernel of size $7 \times 7$, `MORPH_CLOSE` followed by `MORPH_OPEN`) are applied to fill internal holes and smooth jagged boundary pixels.
   - Connected component analysis is executed to keep **only the largest component**, successfully filtering out any background noise or false positive detections.

All 71 images were successfully labeled automatically, achieving an average detection confidence score of **97.8%**.

---

## 4. Dataset Splits & Statistics

The dataset was randomly partitioned into Train (70%), Validation (20%), and Test (10%) splits using a fixed random seed (`42`) for full reproducibility:

| Split | Image Count | Mask Count | Percentage |
|---|---|---|---|
| **Train** | 49 | 49 | 69.0% |
| **Validation** | 14 | 14 | 19.7% |
| **Test** | 8 | 8 | 11.3% |
| **Total** | **71** | **71** | **100.0%** |

### Object Statistics
- **Total Labeled Objects**: 71
- **Mean Object Bounding Box (Pixels)**: $658.23 \times 641.93$ px
- **Mean Object Aspect Ratio**: 1.20
- **Mean Mask Area (Pixels)**: 229,595.86 px²
