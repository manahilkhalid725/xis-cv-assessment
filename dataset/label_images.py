import cv2
import numpy as np
import glob
import os
import torch
import torchvision
from torchvision.transforms import functional as F
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from calibration.undistort import CameraUndistorter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def label_dataset(remote_dir, params_path, output_images_dir, output_masks_dir):
    """
    Uses a pre-trained Mask R-CNN model to automatically generate segmentation masks 
    for the 'remote' class (COCO index 75) from undistorted images.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # Initialize undistorter
    try:
        undistorter = CameraUndistorter(params_path)
        print("Camera undistorter initialized successfully.")
    except Exception as e:
        print(f"Error initializing undistorter: {e}. Please run calibration/calibrate.py first.")
        return

    # Load pre-trained Mask R-CNN model
    print("Loading pre-trained Mask R-CNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")

    search_path = os.path.join(remote_dir, "*")
    images = glob.glob(search_path)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Found {len(images)} remote images to label.")

    # Category index for remote is 75 (1-based index in torchvision is 75, or check classes)
    coco_classes = weights.meta["categories"]
    remote_class_idx = coco_classes.index("remote") # 75 in COCO
    print(f"COCO 'remote' class index: {remote_class_idx}")

    labeled_count = 0
    skipped_count = 0

    for idx, img_path in enumerate(images):
        basename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue
            
        # 1. Undistort image (Mandatory Step 1 dependency!)
        img_undistorted = undistorter.undistort(img)
        
        # 2. Run model inference
        # Convert BGR to RGB and scale to [0, 1]
        img_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(device).unsqueeze(0)
        
        with torch.no_grad():
            predictions = model(img_tensor)[0]
            
        # 3. Extract remote masks
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().squeeze(1).cpu().numpy()
        
        # Filter for remote class and high confidence score (>0.4)
        remote_indices = np.where((labels == remote_class_idx) & (scores > 0.40))[0]
        
        if len(remote_indices) > 0:
            # Sort by confidence score and get the highest confidence mask
            best_idx = remote_indices[np.argmax(scores[remote_indices])]
            best_score = scores[best_idx]
            raw_mask = masks[best_idx]
            
            # Convert raw float mask [0, 1] to binary mask [0, 255]
            binary_mask = (raw_mask > 0.5).astype(np.uint8) * 255
            
            # 4. Post-process mask (morphological opening/closing and largest component filter)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
            
            # Keep only the largest connected component to filter out background noise
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(processed_mask)
            if num_labels > 1:
                # stats format: [x, y, w, h, area]
                # component 0 is background, we look at components 1+
                largest_component_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                processed_mask = (labels_im == largest_component_idx).astype(np.uint8) * 255
            
            # Save undistorted image and corresponding mask
            out_img_name = f"undistorted_{os.path.splitext(basename)[0]}.png"
            out_mask_name = f"mask_{os.path.splitext(basename)[0]}.png"
            
            cv2.imwrite(os.path.join(output_images_dir, out_img_name), img_undistorted)
            cv2.imwrite(os.path.join(output_masks_dir, out_mask_name), processed_mask)
            
            labeled_count += 1
            print(f"[{idx+1}/{len(images)}] Labeled {basename} -> Score: {best_score:.4f}")
        else:
            skipped_count += 1
            print(f"[{idx+1}/{len(images)}] Warning: Remote NOT detected in {basename}. Skipping.")

    print(f"\nLabeling completed. Labeled: {labeled_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label remote images with a pretrained Mask R-CNN.")
    parser.add_argument('--remote_dir', type=str, default=os.path.join(PROJECT_ROOT, "Dataset", "Remote"))
    parser.add_argument('--camera_params', type=str, default=os.path.join(PROJECT_ROOT, "calibration", "camera_params.json"))
    parser.add_argument('--out_images', type=str, default=os.path.join(PROJECT_ROOT, "dataset", "processed", "images"))
    parser.add_argument('--out_masks', type=str, default=os.path.join(PROJECT_ROOT, "dataset", "processed", "masks"))
    args = parser.parse_args()

    label_dataset(args.remote_dir, args.camera_params, args.out_images, args.out_masks)
