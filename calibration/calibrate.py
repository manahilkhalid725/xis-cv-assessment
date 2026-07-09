import cv2
import numpy as np
import glob
import os
import json

def calibrate_camera(board_dir, grid_size=(7, 7), square_size=25.0, output_json="camera_params.json", debug_dir="debug_corners"):
    """
    Performs camera calibration using chessboard images.
    
    Parameters:
        board_dir (str): Directory containing checkerboard calibration images.
        grid_size (tuple): Number of inner corners (width, height).
        square_size (float): Physical size of a square in millimeters.
        output_json (str): Filename to save the computed camera parameters.
        debug_dir (str): Directory to save corner detection visualizations (if not None).
    """
    # Prepare object points (e.g. (0,0,0), (1,0,0), (2,0,0) ....,(6,6,0))
    # Real world coordinates in mm
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Find all jpeg and jpg images
    search_path = os.path.join(board_dir, "*")
    images = glob.glob(search_path)
    # Filter for standard image extensions
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not images:
        print(f"Error: No calibration images found in {board_dir}")
        return None

    print(f"Found {len(images)} calibration images. Starting corner detection...")

    if debug_dir and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    img_shape = None
    successful_detections = 0

    # Criteria for subpixel refinement
    subpixel_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # We ensure all images are in landscape orientation (width > height)
    # so that the sensor orientation is uniform for camera calibration.
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue

        # Force landscape orientation (width > height)
        h, w = img.shape[:2]
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = (w, h)
        elif img_shape != (w, h):
            print(f"Warning: Shape mismatch for {fname} after rotation. Expected {img_shape}, got {(w, h)}. Skipping.")
            continue

        # Find the chess board corners
        # cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE are used for robust corner finding
        ret, corners = cv2.findChessboardCorners(gray, grid_size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            successful_detections += 1
            objpoints.append(objp)

            # Refine corner locations to sub-pixel accuracy (Crucial for scientific rigor!)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpixel_criteria)
            imgpoints.append(corners_refined)

            # Draw and save the corners for verification
            if debug_dir:
                img_drawn = img.copy()
                cv2.drawChessboardCorners(img_drawn, grid_size, corners_refined, ret)
                cv2.imwrite(os.path.join(debug_dir, f"detected_{os.path.basename(fname)}"), img_drawn)
        else:
            print(f"Warning: Chessboard corners not found in {os.path.basename(fname)}")

    print(f"Successfully detected chessboard corners in {successful_detections} / {len(images)} images.")

    if successful_detections < 10:
        print("Warning: Fewer than 10 successful corner detections. Calibration accuracy may be low.")
        if successful_detections == 0:
            print("Error: Chessboard corner detection failed on all images. Cannot calibrate.")
            return None

    print("Running camera calibration...")
    # Run camera calibration
    ret_val, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    # Compute Reprojection Error (Crucial validation metric!)
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        total_points += 1

    mean_error = total_error / total_points
    print(f"Mean Reprojection Error: {mean_error:.4f} pixels")

    # Format parameters for JSON output
    camera_params = {
        "reprojection_error": float(mean_error),
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "grid_size": list(grid_size),
        "square_size_mm": float(square_size),
        "num_calibration_images": len(images),
        "successful_detections": successful_detections
    }

    # Save to file
    with open(output_json, 'w') as f:
        json.dump(camera_params, f, indent=4)
    print(f"Saved camera parameters to {output_json}")

    return camera_params

if __name__ == "__main__":
    board_directory = r"C:\Users\Manahil Khalid\Desktop\Assessment\Dataset\Board"
    output_file = r"C:\Users\Manahil Khalid\Desktop\Assessment\calibration\camera_params.json"
    debug_directory = r"C:\Users\Manahil Khalid\Desktop\Assessment\calibration\debug_corners"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    calibrate_camera(board_directory, grid_size=(7, 7), square_size=25.0, output_json=output_file, debug_dir=debug_directory)
