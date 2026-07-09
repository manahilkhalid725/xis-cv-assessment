import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import glob
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# Ensure import of unet works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNet

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class RemoteDataset(Dataset):
    """
    Dataset loader for remote images and masks.
    """
    def __init__(self, split_dir, split='train', img_size=(128, 128), augment=False):
        self.img_dir = os.path.join(split_dir, split, "images")
        self.mask_dir = os.path.join(split_dir, split, "masks")
        self.img_size = img_size
        self.augment = augment
        
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        self.mask_paths = []
        
        for img_path in self.img_paths:
            basename = os.path.basename(img_path)
            # Find corresponding mask
            name_part = basename.replace("undistorted_", "")
            mask_name = f"mask_{name_part}"
            mask_path = os.path.join(self.mask_dir, mask_name)
            self.mask_paths.append(mask_path)
            
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Resize
        img = img.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)
        
        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img = T.functional.hflip(img)
                mask = T.functional.hflip(mask)
            if np.random.rand() > 0.5:
                img = T.functional.vflip(img)
                mask = T.functional.vflip(mask)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                img = T.functional.rotate(img, angle)
                mask = T.functional.rotate(mask, angle)
                
        # Transform to tensor
        img_tensor = T.functional.to_tensor(img) # [0, 1] range
        mask_tensor = T.functional.to_tensor(mask) # [0, 1] range, binary
        
        # Apply standard normalization
        img_tensor = T.functional.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img_tensor, mask_tensor

class DiceBCELoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss for robust segmentation.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        bce_loss = self.bce(inputs_flat, targets_flat)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        return bce_loss + dice_loss

def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculates segmentation metrics: IoU, Precision, Recall, F1.
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum().item()
    
    # Intersection over Union (IoU)
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 1.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return iou, precision, recall, f1

def train_model(split_dir, output_model_path, epochs=15, batch_size=4, lr=1e-4, quick_test=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load datasets
    train_dataset = RemoteDataset(split_dir, split='train', augment=True)
    val_dataset = RemoteDataset(split_dir, split='val', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()
    
    best_val_iou = 0.0
    
    # To log metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'val_iou': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_prec = 0.0
        val_rec = 0.0
        val_f1 = 0.0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                
                # Calculate metrics for the batch
                for i in range(imgs.size(0)):
                    iou, p, r, f1 = calculate_metrics(outputs[i], masks[i])
                    val_iou += iou
                    val_prec += p
                    val_rec += r
                    val_f1 += f1
                    
        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        val_prec /= len(val_loader.dataset)
        val_rec /= len(val_loader.dataset)
        val_f1 /= len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best weights
        if val_iou > best_val_iou and not quick_test:
            best_val_iou = val_iou
            torch.save(model.state_dict(), output_model_path)
            print(f"--> Saved new best model weights (Val IoU: {best_val_iou:.4f})")
            
    if quick_test:
        # Save placeholder for check
        torch.save(model.state_dict(), output_model_path)
        print("Quick test run complete. Model saved.")
        return
        
    print(f"Training finished. Best Val IoU: {best_val_iou:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BCE + Dice Loss Curves')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.plot(history['val_f1'], label='Val F1')
    plt.plot(history['val_precision'], label='Val Precision')
    plt.plot(history['val_recall'], label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    
    curves_path = os.path.join(os.path.dirname(output_model_path), "training_curves.png")
    plt.tight_layout()
    plt.savefig(curves_path)
    print(f"Saved training curves to {curves_path}")
    
    # Print final test metrics if desired
    # For now return history
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', type=str, default=os.path.join(PROJECT_ROOT, "dataset", "split"),
                         help="Directory containing train/val/test split folders.")
    parser.add_argument('--output_model', type=str, default=os.path.join(PROJECT_ROOT, "models", "best_model.pth"),
                         help="Path to save the best model weights.")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--quick_test', action='store_true', help='Run 1 epoch quickly for testing')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    train_model(args.split_dir, args.output_model, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, quick_test=args.quick_test)
