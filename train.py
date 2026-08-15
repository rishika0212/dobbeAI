"""
Main Training Pipeline Script for OPG Dental Segmentation.
Execute via CLI:
    python train.py --data_dir ./dataset --epochs 30 --batch_size 8 --dry_run
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from src.dataset import DentalSegmentationDataset
from src.model import build_segmentation_model, BCEDiceLoss
from src.post_processing import apply_post_processing
from src.metrics import compute_segmentation_metrics, print_metrics_summary
from src.utils import plot_qualitative_results, plot_eda_summary

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dental OPG Cavity & Lesion Segmentation Model")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset directory containing images and masks")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--max_samples", type=int, default=2500, help="Maximum number of real images to train on (default 2500 to hit 2000+ requirement fast)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="Target image resolution (512x512)")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "unetplusplus", "deeplabv3plus"], help="Model architecture")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder network")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for checkpoints and visual results")
    parser.add_argument("--dry_run", action="store_true", help="Perform 1-batch dry run to verify pipeline integrity without full training")
    return parser.parse_args()


def get_image_mask_pairs(data_dir):
    """Discovers real images and mask pairs from dataset directory."""
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(
            f"Dataset directory '{img_dir}' not found. "
            "Please download the dataset using 'python download_dataset.py --source dentex' "
            "or point --data_dir to your local dataset root containing 'images' and 'masks' subdirectories."
        )

    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        raise FileNotFoundError(
            f"No image files found in '{img_dir}'. "
            "Please download the dataset using 'python download_dataset.py --source dentex'."
        )

    mask_paths = []
    for img_p in img_paths:
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        # Look for matching mask
        candidate_masks = [
            os.path.join(mask_dir, f"{base_name}.png"),
            os.path.join(mask_dir, f"{base_name}_mask.png"),
            os.path.join(mask_dir, f"{base_name}.jpg"),
        ]
        found_mask = None
        for cand in candidate_masks:
            if os.path.exists(cand):
                found_mask = cand
                break
        mask_paths.append(found_mask)

    return img_paths, mask_paths


def train_pipeline():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Step 1: Dataset Acquisition & Discovery
    img_paths, mask_paths = get_image_mask_pairs(args.data_dir)
    print(f"Found {len(img_paths)} total real images and masks in repository.")

    if len(img_paths) == 0:
        raise ValueError(f"No image files found in {args.data_dir}")

    # Subsample to max_samples (e.g. 2,500) for fast training while satisfying 2,000+ recommendation
    if args.max_samples and len(img_paths) > args.max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(img_paths), size=args.max_samples, replace=False)
        img_paths = [img_paths[i] for i in indices]
        mask_paths = [mask_paths[i] for i in indices]
        print(f"Subsampled to {len(img_paths)} real images & masks to meet the 2,000+ dataset recommendation efficiently.")

    # Generate EDA Plot
    plot_eda_summary(img_paths, mask_paths, save_path=os.path.join(args.output_dir, "eda_summary.png"))

    # Step 2: Stratified Train / Validation / Test Split (80 / 10 / 10)
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        test_imgs, test_masks, test_size=0.5, random_state=42
    )

    print(f"Split sizes -> Train: {len(train_imgs)} | Validation: {len(val_imgs)} | Test: {len(test_imgs)}")

    # Create Datasets & DataLoaders
    train_dataset = DentalSegmentationDataset(train_imgs, train_masks, target_size=(args.img_size, args.img_size), is_train=True)
    val_dataset = DentalSegmentationDataset(val_imgs, val_masks, target_size=(args.img_size, args.img_size), is_train=False)
    test_dataset = DentalSegmentationDataset(test_imgs, test_masks, target_size=(args.img_size, args.img_size), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Step 3: Model & Loss Function Selection
    model = build_segmentation_model(
        architecture=args.arch,
        encoder_name=args.encoder,
        in_channels=3,
        classes=1,
        encoder_weights="imagenet"
    ).to(device)

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Step 4: Training & Validation Loop
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")

    epochs = 1 if args.dry_run else args.epochs
    print(f"\nStarting training loop for {epochs} epoch(s) (Dry Run Mode: {args.dry_run})...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs:02d}] (Train)", leave=False)
        for step, (images, masks, _) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

            if args.dry_run:
                print(f"Dry run step completed successfully. Train Batch Loss: {loss.item():.4f}")
                break

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                if args.dry_run:
                    break

        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint to {checkpoint_path}")

    # Step 5: Test Set Evaluation & Post-processing Analysis
    print("\nEvaluating final model on held-out Test split...")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    all_raw_preds, all_post_preds, all_gts, eval_img_paths = [], [], [], []

    with torch.no_grad():
        for images, masks, paths in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(axis=1) # (B, H, W)
            gts = masks.numpy().squeeze(axis=1)

            for i in range(len(paths)):
                raw_prob = probs[i]
                gt_mask = gts[i]
                
                # Apply post-processing
                post_mask = apply_post_processing(raw_prob, threshold=0.5, min_area=100, do_morphology=True)

                all_raw_preds.append(raw_prob)
                all_post_preds.append(post_mask)
                all_gts.append(gt_mask)
                eval_img_paths.append(paths[i])

            if args.dry_run:
                break

    # Calculate final test set metrics
    raw_metrics = compute_segmentation_metrics(np.array(all_raw_preds) > 0.5, np.array(all_gts))
    post_metrics = compute_segmentation_metrics(np.array(all_post_preds), np.array(all_gts))

    print_metrics_summary(post_metrics, split_name="Test Split (Post-Processed)")

    # Step 6: Qualitative Visualizations
    vis_path = os.path.join(args.output_dir, "qualitative_results.png")
    plot_qualitative_results(eval_img_paths, all_gts, all_raw_preds, all_post_preds, save_path=vis_path, num_samples=4)

    print(f"\nPipeline run completed successfully. Weights saved to '{checkpoint_path}'.")

if __name__ == "__main__":
    train_pipeline()
