import os
import glob
import shutil
import random
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def split_dataset(src_images_dir, src_masks_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits the generated dataset into train, val, and test partitions.
    """
    # Create split directories
    splits = ['train', 'val', 'test']
    for s in splits:
        os.makedirs(os.path.join(output_dir, s, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, s, 'masks'), exist_ok=True)

    images = glob.glob(os.path.join(src_images_dir, "*.png"))
    print(f"Found {len(images)} images to split.")
    
    if not images:
        print("Error: No images found. Run dataset/label_images.py first.")
        return

    # Couple images and masks to ensure correct alignment
    dataset = []
    for img_path in images:
        basename = os.path.basename(img_path)
        # Corresponding mask name is mask_... instead of undistorted_...
        # Image name format: undistorted_XYZ.png
        # Mask name format: mask_XYZ.png
        name_part = basename.replace("undistorted_", "")
        mask_name = f"mask_{name_part}"
        mask_path = os.path.join(src_masks_dir, mask_name)
        
        if os.path.exists(mask_path):
            dataset.append((img_path, mask_path))
        else:
            print(f"Warning: Corresponding mask not found for {basename} at {mask_path}")

    # Shuffle dataset for random distribution
    random.seed(42)
    random.shuffle(dataset)

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    splits_data = {
        'train': dataset[:n_train],
        'val': dataset[n_train:n_train+n_val],
        'test': dataset[n_train+n_val:]
    }

    print(f"Splits distribution: Train={n_train}, Val={n_val}, Test={n_test}")

    # Copy files
    for split_name, data in splits_data.items():
        for img_path, mask_path in data:
            dest_img = os.path.join(output_dir, split_name, 'images', os.path.basename(img_path))
            dest_mask = os.path.join(output_dir, split_name, 'masks', os.path.basename(mask_path))
            
            shutil.copy(img_path, dest_img)
            shutil.copy(mask_path, dest_mask)

    print("Dataset split completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split labeled dataset into train/val/test.")
    parser.add_argument('--src_img', type=str, default=os.path.join(PROJECT_ROOT, "dataset", "processed", "images"))
    parser.add_argument('--src_mask', type=str, default=os.path.join(PROJECT_ROOT, "dataset", "processed", "masks"))
    parser.add_argument('--out_dir', type=str, default=os.path.join(PROJECT_ROOT, "dataset", "split"))
    args = parser.parse_args()

    split_dataset(args.src_img, args.src_mask, args.out_dir)
