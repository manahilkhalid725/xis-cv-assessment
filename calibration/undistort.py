import cv2
import numpy as np
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class CameraUndistorter:
    """
    Loads camera calibration parameters from a JSON file and undistorts images.
    """
    def __init__(self, params_path):
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Camera parameters file not found at {params_path}. Run calibration/calibrate.py first.")
            
        with open(params_path, 'r') as f:
            self.params = json.load(f)
            
        self.camera_matrix = np.array(self.params["camera_matrix"], dtype=np.float32)
        self.dist_coeffs = np.array(self.params["distortion_coefficients"], dtype=np.float32)
        self.reprojection_error = self.params["reprojection_error"]
        
    def undistort(self, img):
        """
        Undistorts a given BGR image.
        """
        h, w = img.shape[:2]
        # Get optimal new camera matrix. alpha=0 crops black edges, alpha=1 keeps all pixels
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h))
        # Undistort
        undistorted_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # Crop the image if roi is valid
        # x, y, w_roi, h_roi = roi
        # if w_roi > 0 and h_roi > 0:
        #     undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]
            
        return undistorted_img

def main():
    # Simple verification code
    params_file = os.path.join(PROJECT_ROOT, "calibration", "camera_params.json")
    if not os.path.exists(params_file):
        print(f"Please run calibrate.py first. Could not find {params_file}")
        return
        
    undistorter = CameraUndistorter(params_file)
    print("Camera matrix:")
    print(undistorter.camera_matrix)
    print("Distortion coefficients:")
    print(undistorter.dist_coeffs)
    print(f"Camera calibration parameters loaded successfully. Reprojection error: {undistorter.reprojection_error:.4f} px")

if __name__ == "__main__":
    main()
