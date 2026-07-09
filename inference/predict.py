import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import argparse
import sys

# Ensure import of unet and CameraUndistorter works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNet
from calibration.undistort import CameraUndistorter

class InferencePipeline:
    """
    End-to-end inference pipeline: Raw Image => Undistortion => Custom U-Net Segmentation.
    """
    def __init__(self, params_path, model_weights_path, img_size=(128, 128)):
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load undistorter
        self.undistorter = CameraUndistorter(params_path)
        
        # Load U-Net model
        self.model = UNet(n_channels=3, n_classes=1, bilinear=False)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded segmentation model and camera parameters on {self.device}.")
        
    def predict(self, raw_img_path, output_dir=None):
        # 1. Read raw image
        raw_img = cv2.imread(raw_img_path)
        if raw_img is None:
            raise FileNotFoundError(f"Could not read image at {raw_img_path}")
            
        # Ensure landscape orientation if it was captured as portrait
        h_orig, w_orig = raw_img.shape[:2]
        if h_orig > w_orig:
            raw_img = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)
            
        # 2. Apply camera undistortion (Step 1 dependency)
        img_undistorted = self.undistorter.undistort(raw_img)
        h_undist, w_undist = img_undistorted.shape[:2]
        
        # 3. Preprocess for model
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize to match U-Net input size
        img_resized = img_pil.resize(self.img_size, Image.BILINEAR)
        img_tensor = T.functional.to_tensor(img_resized)
        
        # Normalize
        img_tensor = T.functional.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.to(self.device).unsqueeze(0)
        
        # 4. Run model inference
        with torch.no_grad():
            prob_map = self.model(img_tensor).squeeze(0).squeeze(0).cpu().numpy()
            
        # 5. Post-process mask
        # Resize probability map back to undistorted image dimensions
        prob_map_resized = cv2.resize(prob_map, (w_undist, h_undist), interpolation=cv2.INTER_LINEAR)
        binary_mask = (prob_map_resized > 0.5).astype(np.uint8) * 255
        
        # Morphological operations to clean boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        
        # Keep only largest component
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(processed_mask)
        if num_labels > 1:
            largest_component_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            processed_mask = (labels_im == largest_component_idx).astype(np.uint8) * 255
            
        # 6. Generate overlay
        mask_rgb = np.zeros_like(img_undistorted)
        mask_rgb[:, :, 2] = processed_mask # Red mask overlay
        
        overlay = cv2.addWeighted(img_undistorted, 0.7, mask_rgb, 0.3, 0)
        
        # Save output if path is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            basename = os.path.basename(raw_img_path)
            out_path = os.path.join(output_dir, f"segmented_{basename}")
            mask_path = os.path.join(output_dir, f"mask_{basename}")
            
            cv2.imwrite(out_path, overlay)
            cv2.imwrite(mask_path, processed_mask)
            print(f"Saved inference overlay to {out_path} and mask to {mask_path}")
            
        return img_undistorted, processed_mask, overlay

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to raw input image')
    args = parser.parse_args()
    
    params = r"C:\Users\Manahil Khalid\Desktop\Assessment\calibration\camera_params.json"
    weights = r"C:\Users\Manahil Khalid\Desktop\Assessment\models\best_model.pth"
    out_dir = r"C:\Users\Manahil Khalid\Desktop\Assessment\inference"
    
    if not os.path.exists(weights):
        print("Error: Trained model weights not found. Run models/train.py first.")
        sys.exit(1)
        
    pipeline = InferencePipeline(params, weights)
    pipeline.predict(args.image, output_dir=out_dir)
