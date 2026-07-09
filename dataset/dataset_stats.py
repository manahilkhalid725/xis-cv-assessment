import cv2
import numpy as np
import glob
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def calculate_dataset_stats(split_dir):
    """
    Computes statistics of the splits and mask properties for documentation.
    """
    stats = {}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images = glob.glob(os.path.join(split_dir, split, "images", "*.png"))
        masks = glob.glob(os.path.join(split_dir, split, "masks", "*.png"))
        stats[split] = {
            "count": len(images),
            "masks_count": len(masks)
        }
        
    all_masks = glob.glob(os.path.join(split_dir, "*", "masks", "*.png"))
    areas = []
    aspect_ratios = []
    widths = []
    heights = []
    
    for mask_path in all_masks:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            areas.append(area)
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(h / w if w > 0 else 0)
            
    stats["mask_stats"] = {
        "total_labeled_objects": len(areas),
        "mean_pixel_area": float(np.mean(areas)) if areas else 0.0,
        "std_pixel_area": float(np.std(areas)) if areas else 0.0,
        "mean_width_px": float(np.mean(widths)) if widths else 0.0,
        "mean_height_px": float(np.mean(heights)) if heights else 0.0,
        "mean_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
    }
    
    print("----- Dataset Statistics -----")
    for split in splits:
        print(f"Split '{split}': {stats[split]['count']} images, {stats[split]['masks_count']} masks")
    print(f"Total labeled objects: {stats['mask_stats']['total_labeled_objects']}")
    print(f"Mean object bounding box: {stats['mask_stats']['mean_width_px']:.2f} x {stats['mask_stats']['mean_height_px']:.2f} px")
    print(f"Mean object aspect ratio: {stats['mask_stats']['mean_aspect_ratio']:.2f}")
    print(f"Mean object area: {stats['mask_stats']['mean_pixel_area']:.2f} px^2")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute dataset split statistics.")
    parser.add_argument('--split_dir', type=str, default=os.path.join(PROJECT_ROOT, "dataset", "split"))
    args = parser.parse_args()

    if os.path.exists(args.split_dir):
        calculate_dataset_stats(args.split_dir)
    else:
        print(f"Split directory '{args.split_dir}' does not exist. Please run prepare_dataset.py first.")
