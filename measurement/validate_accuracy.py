import cv2
import numpy as np
import os
import glob
import sys
import json
import argparse

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.predict import InferencePipeline
from measurement.measure import estimate_pixel_to_mm_ratio, measure_object, draw_measurements

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def run_accuracy_validation(split_dir, params_path, model_weights_path, output_dir):
    """
    Runs the end-to-end measurement pipeline on the test dataset images 
    and compares the system measurements against physical ground-truth dimensions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the inference pipeline (loads undistorter and custom U-Net model)
    try:
        pipeline = InferencePipeline(params_path, model_weights_path)
    except Exception as e:
        print(f"Error initializing pipeline: {e}. Check if training is completed.")
        return None

    # Physical ground truth dimensions of the target TV remote control (in mm)
    GT_WIDTH_MM = 48.0
    GT_HEIGHT_MM = 186.0

    test_images = sorted(glob.glob(os.path.join(split_dir, "test", "images", "*.png")))
    test_masks = sorted(glob.glob(os.path.join(split_dir, "test", "masks", "*.png")))
    
    print(f"Found {len(test_images)} test images for metric validation.")
    
    results = []
    width_errors = []
    height_errors = []
    width_pct_errors = []
    height_pct_errors = []

    for idx, img_path in enumerate(test_images):
        basename = os.path.basename(img_path)
        # Load corresponding ground truth mask
        name_part = basename.replace("undistorted_", "")
        gt_mask_path = os.path.join(split_dir, "test", "masks", f"mask_{name_part}")
        
        # 1. Run pipeline inference to get model's predicted mask
        # Note: input image is already undistorted in the split folder,
        # but InferencePipeline handles raw images by undistorting them.
        # Since these are already undistorted, we pass them directly, or let predict do it.
        # To be clean, we read the image and predict.
        img_bgr = cv2.imread(img_path)
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # We run the pipeline's predict method
        # This will return the undistorted image, predicted mask, and overlay
        # Since it's already undistorted, undistortion will be a near-identity mapping (correct)
        img_undist, pred_mask, _ = pipeline.predict(img_path)
        
        # 2. Derive pixel-to-mm ratio from background chessboard
        debug_path = os.path.join(output_dir, f"debug_corners_{basename}")
        px_to_mm, _ = estimate_pixel_to_mm_ratio(img_undist, pred_mask, square_size_mm=25.0, debug_path=debug_path)
        
        # 3. Measure target object using predicted mask and derived ratio
        measurement = measure_object(img_undist, pred_mask, px_to_mm)
        
        if measurement is None:
            print(f"Warning: Measurement failed for {basename}. Skipping.")
            continue
            
        pred_w = measurement["width_mm"]
        pred_h = measurement["height_mm"]
        
        # 4. Compute errors
        w_err = abs(pred_w - GT_WIDTH_MM)
        h_err = abs(pred_h - GT_HEIGHT_MM)
        w_pct = (w_err / GT_WIDTH_MM) * 100
        h_pct = (h_err / GT_HEIGHT_MM) * 100
        
        width_errors.append(w_err)
        height_errors.append(h_err)
        width_pct_errors.append(w_pct)
        height_pct_errors.append(h_pct)
        
        # Draw and save annotations
        annotated_img = draw_measurements(img_undist, measurement, px_to_mm)
        cv2.imwrite(os.path.join(output_dir, f"measured_{basename}"), annotated_img)
        
        results.append({
            "image": basename,
            "px_to_mm": float(px_to_mm),
            "pred_width": float(pred_w),
            "pred_height": float(pred_h),
            "width_error": float(w_err),
            "height_error": float(h_err),
            "width_pct_error": float(w_pct),
            "height_pct_error": float(h_pct)
        })
        
        print(f"[{idx+1}/{len(test_images)}] {basename} | Ratio: {px_to_mm:.2f} px/mm | Pred: {pred_w:.1f}x{pred_h:.1f}mm | Err: {w_err:.1f}x{h_err:.1f}mm")

    if not results:
        print("Error: No successful measurements to compile.")
        return None

    # Calculate aggregate metrics
    mae_w = np.mean(width_errors)
    mae_h = np.mean(height_errors)
    mpe_w = np.mean(width_pct_errors)
    mpe_h = np.mean(height_pct_errors)
    
    print("\n" + "="*50)
    print("ACCURACY VALIDATION SUMMARY")
    print(f"Mean Absolute Error (Width):  {mae_w:.2f} mm")
    print(f"Mean Absolute Error (Height): {mae_h:.2f} mm")
    print(f"Mean Percentage Error (Width):  {mpe_w:.2f} %")
    print(f"Mean Percentage Error (Height): {mpe_h:.2f} %")
    print("="*50 + "\n")
    
    # Save results as JSON
    summary = {
        "ground_truth": {"width_mm": GT_WIDTH_MM, "height_mm": GT_HEIGHT_MM},
        "individual_results": results,
        "aggregate_metrics": {
            "mae_width_mm": float(mae_w),
            "mae_height_mm": float(mae_h),
            "mpe_width_pct": float(mpe_w),
            "mpe_height_pct": float(mpe_h)
        }
    }
    
    with open(os.path.join(output_dir, "validation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    # Generate markdown table for reporting
    markdown_table = "| Image Filename | Px/MM Ratio | Pred Width | Pred Height | GT Width | GT Height | Width Error | Height Error |\n"
    markdown_table += "|---|---|---|---|---|---|---|---|\n"
    for r in results:
        markdown_table += f"| {r['image']} | {r['px_to_mm']:.2f} | {r['pred_width']:.1f} mm | {r['pred_height']:.1f} mm | {GT_WIDTH_MM:.1f} mm | {GT_HEIGHT_MM:.1f} mm | {r['width_error']:.1f} mm ({r['width_pct_error']:.1f}%) | {r['height_error']:.1f} mm ({r['height_pct_error']:.1f}%) |\n"
    
    markdown_table += f"| **AVERAGE (MAE / MPE)** | | | | | | **{mae_w:.2f} mm ({mpe_w:.1f}%)** | **{mae_h:.2f} mm ({mpe_h:.1f}%)** |\n"
    
    with open(os.path.join(output_dir, "validation_table.md"), 'w') as f:
        f.write(markdown_table)
        
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end measurement accuracy validation on the test split.")
    parser.add_argument('--split_dir', type=str, default=os.path.join(PROJECT_ROOT, "dataset", "split"))
    parser.add_argument('--params', type=str, default=os.path.join(PROJECT_ROOT, "calibration", "camera_params.json"))
    parser.add_argument('--weights', type=str, default=os.path.join(PROJECT_ROOT, "models", "best_model.pth"))
    parser.add_argument('--out_dir', type=str, default=os.path.join(PROJECT_ROOT, "measurement"))
    args = parser.parse_args()

    run_accuracy_validation(args.split_dir, args.params, args.weights, args.out_dir)
