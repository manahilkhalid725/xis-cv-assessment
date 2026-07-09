import cv2
import numpy as np
import os
import json
import sys

# Ensure import of CameraUndistorter works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from calibration.undistort import CameraUndistorter

def estimate_pixel_to_mm_ratio(img_undistorted, remote_mask, square_size_mm=25.0, debug_path=None):
    """
    Estimates the pixel-to-mm ratio by detecting chessboard corners in the background 
    (outside the remote mask) and finding the primary grid spacing peak in the pairwise distance histogram.
    """
    gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
    
    # 1. Create a search mask for the background (outside the remote)
    # Dilate the remote mask by 85 pixels to completely exclude remote buttons and boundaries
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (85, 85))
    dilated_mask = cv2.dilate(remote_mask, kernel_dilate)
    background_mask = cv2.bitwise_not(dilated_mask)
    
    # 2. Detect corners in the background
    max_corners = 400
    quality_level = 0.04
    min_distance = 20
    
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance, mask=background_mask)
    
    if corners is None or len(corners) < 10:
        print("Warning: Too few corners detected in background. Falling back to default ratio.")
        # Fallback ratio based on calibration average: 126px for 25mm = 5.04 px/mm
        fallback_ratio = 5.04
        return fallback_ratio, None
        
    corners = corners.squeeze(1)
    
    # Refine corner locations to sub-pixel accuracy
    subpixel_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (5, 5), (-1, -1), subpixel_criteria)
    
    # 3. Calculate pairwise distances between corners
    # The grid structure of the chessboard causes adjacent corners to have very similar distances.
    # We collect distances in the range [110.0, 190.0] pixels, which covers the expected size
    # of a square at the camera's height.
    pairwise_dists = []
    for i in range(len(corners)):
        pt = corners[i]
        diffs = corners - pt
        pt_dists = np.sqrt(np.sum(diffs**2, axis=1))
        # Exclude self and look for checkerboard spacing range
        valid = pt_dists[(pt_dists > 110.0) & (pt_dists < 190.0)]
        pairwise_dists.extend(valid)
        
    pairwise_dists = np.array(pairwise_dists)
    
    if len(pairwise_dists) < 5:
        print("Warning: Not enough valid pairwise distances. Falling back to default ratio.")
        return 5.04, None
        
    # 4. Find the peak of the histogram (the mode) in the range [115, 185] pixels
    # We use a bin width of 4 pixels to cluster the distances.
    bins = np.arange(115, 185, 4)
    hist, bin_edges = np.histogram(pairwise_dists, bins=bins)
    
    max_bin_idx = np.argmax(hist)
    grid_spacing_px = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx+1]) / 2.0
    
    # Refine estimate by taking the mean of distances falling into the peak bin
    bin_min = bin_edges[max_bin_idx]
    bin_max = bin_edges[max_bin_idx+1]
    in_bin_dists = pairwise_dists[(pairwise_dists >= bin_min) & (pairwise_dists <= bin_max)]
    if len(in_bin_dists) > 0:
        grid_spacing_px = np.mean(in_bin_dists)
        
    # Compute pixel-to-mm ratio
    px_to_mm = grid_spacing_px / square_size_mm
    
    # Visualization for debugging
    debug_img = img_undistorted.copy()
    if debug_path:
        for pt in corners:
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
        cv2.putText(debug_img, f"Spacing: {grid_spacing_px:.1f} px | Ratio: {px_to_mm:.2f} px/mm", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(debug_path, debug_img)
        
    return px_to_mm, debug_img

def measure_object(img_undistorted, mask, px_to_mm):
    """
    Measures the real-world width and height of the segmented object in mm.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Get the largest contour (the remote control)
    c = max(contours, key=cv2.contourArea)
    
    # Fit a minimum area bounding box
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # Dimensions in pixels
    width_px = rect[1][0]
    height_px = rect[1][1]
    
    # Convert to mm
    mm_per_px = 1.0 / px_to_mm
    width_mm = width_px * mm_per_px
    height_mm = height_px * mm_per_px
    
    # Identify physical width (smaller) and height (larger)
    real_width_mm = min(width_mm, height_mm)
    real_height_mm = max(width_mm, height_mm)
    
    return {
        "box": box,
        "width_px": width_px,
        "height_px": height_px,
        "width_mm": real_width_mm,
        "height_mm": real_height_mm,
        "center": rect[0],
        "angle": rect[2]
    }

def draw_measurements(img, measurement, px_to_mm):
    """
    Draws the minimum bounding box and real-world dimensions on the image.
    """
    annotated_img = img.copy()
    box = measurement["box"]
    
    # Draw bounding box in green
    cv2.drawContours(annotated_img, [box], 0, (0, 255, 0), 2)
    
    # Put label text on the image
    w_mm = measurement["width_mm"]
    h_mm = measurement["height_mm"]
    text = f"Remote: {w_mm:.1f}mm x {h_mm:.1f}mm"
    
    # Get box center to place text
    center = measurement["center"]
    cv2.putText(annotated_img, text, (int(center[0]) - 150, int(center[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
    # Draw scale bar
    px_for_50mm = int(50.0 * px_to_mm)
    h_img, w_img = img.shape[:2]
    # Scale bar location (bottom left)
    cv2.line(annotated_img, (50, h_img - 50), (50 + px_for_50mm, h_img - 50), (255, 0, 0), 3)
    cv2.putText(annotated_img, "50 mm", (50, h_img - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return annotated_img

def main():
    print("Measurement module loaded.")

if __name__ == "__main__":
    main()
