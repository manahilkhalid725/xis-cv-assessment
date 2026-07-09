# Camera Calibration Report

## Method Overview
Camera calibration was performed to calculate the camera's intrinsic parameters ($K$) and distortion coefficients ($D$), which are essential for removing lens distortion (both radial and tangential) prior to any metric calculations. The calibration follows the standard pinhole camera model with lens distortion:

$$\mathbf{x}_{\text{undistorted}} = f(\mathbf{x}_{\text{distorted}}, D)$$

We used a planar chessboard pattern as the calibration target. A set of 20 images of the chessboard was captured from varied angles, distances, and rotations to ensure proper coverage of the lens volume. 

The calibration script transposes and rotates any portrait-oriented images to landscape format, ensuring a uniform shape of $1600 \times 1204$ pixels. 

### Corner Detection & Sub-pixel Refinement
1. **Initial Search**: Grayscale versions of the images are searched using `cv2.findChessboardCorners` for a grid size of $7 \times 7$ inner corners (corresponding to an $8 \times 8$ chessboard).
2. **Flags**: Adaptive thresholding and image normalization flags (`cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE`) are enabled to handle variations in lighting.
3. **Sub-pixel Localization**: For every successful grid detection, the coordinates are refined to sub-pixel accuracy using `cv2.cornerSubPix` with a window size of $11 \times 11$ pixels.
4. **Parameter Calculation**: The 3D-to-2D correspondences are passed to `cv2.calibrateCamera` to compute the calibration parameters.

---

## Calibration Image Statistics
- **Total Calibration Images**: 20
- **Successful Corner Detections**: 12 (60% success rate; skipped images were due to extreme out-of-focus blur or board clipping)
- **Calibration Image Dimensions**: $1600 \times 1204$ pixels

---

## Intrinsic Calibration Parameters

### Camera Intrinsic Matrix ($K$)
The intrinsic matrix represents the optical properties of the camera:
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Where:
- $f_x, f_y$ are the focal lengths in pixels.
- $c_x, c_y$ is the principal point (optical center) in pixels.

For our camera:
$$K = \begin{bmatrix} 1328.0162 & 0 & 809.1828 \\ 0 & 1326.7395 & 593.1157 \\ 0 & 0 & 1 \end{bmatrix}$$

### Lens Distortion Coefficients ($D$)
The radial and tangential distortion coefficients are represented by $D = [k_1, k_2, p_1, p_2, k_3]$:
- **Radial Coefficients ($k_1, k_2, k_3$)**: Correct barrel and pincushion distortion.
- **Tangential Coefficients ($p_1, p_2$)**: Correct decentering distortion caused by the physical sensor not being perfectly parallel to the lens.

For our camera:
$$D = \begin{bmatrix} 0.2189 & -0.9617 & -0.0036 & 0.0037 & 1.7345 \end{bmatrix}$$

---

## Reprojection Error & Quality Assessment
The quality of the calibration was validated using the **Mean Reprojection Error (MRE)**. The reprojection error measures the distance (in pixels) between the detected corners and the 3D grid points projected back onto the image plane using the computed calibration matrices:

$$\text{MRE} = \frac{1}{N} \sum_{i=1}^N \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|_2$$

- **Mean Reprojection Error**: **0.1933 pixels**

### Interpretation
An MRE of **0.1933 pixels** is well below the target threshold of **0.3 pixels**, signifying an **excellent** calibration. This low error level guarantees that lens distortion can be successfully corrected, paving the way for sub-millimeter level measurement accuracy.

---

## Undistortion Verification
The parameters are stored in `calibration/camera_params.json` and are loaded by the `CameraUndistorter` class in `calibration/undistort.py`. 
Any input image from this camera is undistorted using `cv2.undistort` with the optimal camera matrix computed by `cv2.getOptimalNewCameraMatrix` (using an alpha of `0` to crop any black boundary edges). This step is mandatory before any pixels-to-metric conversion is carried out.
