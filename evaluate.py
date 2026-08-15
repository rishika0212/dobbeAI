"""
Standalone Evaluation Script for OPG Dental Segmentation Model.
Loads a trained PyTorch checkpoint and computes metrics + qualitative visual comparisons.

Execute via CLI:
    python evaluate.py --model_path ./outputs/best_model.pth --data_dir ./dataset
"""

import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DentalSegmentationDataset
from src.model import build_segmentation_model
from src.post_processing import apply_post_processing
from src.metrics import print_metrics_summary
from src.utils import plot_qualitative_results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Dental OPG Segmentation Model")
    parser.add_argument("--model_path", type=str, default="./outputs/best_model.pth", help="Path to saved .pth model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_results", help="Directory to save visual overlays")
    parser.add_argument("--arch", type=str, default="unet", help="Model architecture")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Backbone encoder")
    parser.add_argument("--img_size", type=int, default=512, help="Target image resolution")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation (default 16 for high GPU utilization)")
    parser.add_argument("--max_samples", type=int, default=None, help="Optionally limit evaluation to N samples (default: evaluate all)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--min_area", type=int, default=100, help="Connected component minimum pixel area filter")
    return parser.parse_args()

def compute_metrics_from_counts(counts, eps=1e-7):
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1_score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "dice_score": float(f1_score),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model checkpoint: '{args.model_path}' on device: {device}")

    # Discover images & masks
    img_dir = os.path.join(args.data_dir, "images")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(
            f"Dataset directory '{img_dir}' not found. "
            "Please download the dataset using 'python download_dataset.py --source dentex' "
            "or point --data_dir to your local dataset root containing 'images' and 'masks' subdirectories."
        )
    
    from train import get_image_mask_pairs
    img_paths, mask_paths = get_image_mask_pairs(args.data_dir)

    if args.max_samples and len(img_paths) > args.max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(img_paths), size=args.max_samples, replace=False)
        img_paths = [img_paths[i] for i in indices]
        mask_paths = [mask_paths[i] for i in indices]
        print(f"Subsampled evaluation dataset to {len(img_paths)} samples.")
    else:
        print(f"Evaluating on all {len(img_paths)} dataset images (batch_size={args.batch_size}).")

    test_dataset = DentalSegmentationDataset(img_paths, mask_paths, target_size=(args.img_size, args.img_size), is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Instantiate Model
    model = build_segmentation_model(
        architecture=args.arch,
        encoder_name=args.encoder,
        in_channels=3,
        classes=1,
        encoder_weights=None
    ).to(device)

    if os.path.exists(args.model_path):
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded weights from '{args.model_path}'")
    else:
        print(f"Warning: Model checkpoint '{args.model_path}' not found. Evaluating with uninitialized weights.")

    model.eval()

    raw_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    post_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    vis_paths, vis_gt, vis_raw, vis_post = [], [], [], []
    num_vis_samples = 6

    pbar = tqdm(test_loader, desc="Evaluating Batches", unit="batch")
    with torch.no_grad():
        for images, masks, paths in pbar:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(axis=1) # (B, H, W)
            gts = masks.numpy().squeeze(axis=1) # (B, H, W)

            if probs.ndim == 2:
                probs = np.expand_dims(probs, axis=0)
                gts = np.expand_dims(gts, axis=0)

            for i in range(len(paths)):
                p = probs[i]
                g = gts[i]
                post = apply_post_processing(p, threshold=args.threshold, min_area=args.min_area, do_morphology=True)

                raw_bin = (p > args.threshold)
                post_bin = (post > 0)
                gt_bin = (g > 0.5)

                raw_counts["tp"] += int(np.sum((raw_bin == 1) & (gt_bin == 1)))
                raw_counts["fp"] += int(np.sum((raw_bin == 1) & (gt_bin == 0)))
                raw_counts["fn"] += int(np.sum((raw_bin == 0) & (gt_bin == 1)))
                raw_counts["tn"] += int(np.sum((raw_bin == 0) & (gt_bin == 0)))

                post_counts["tp"] += int(np.sum((post_bin == 1) & (gt_bin == 1)))
                post_counts["fp"] += int(np.sum((post_bin == 1) & (gt_bin == 0)))
                post_counts["fn"] += int(np.sum((post_bin == 0) & (gt_bin == 1)))
                post_counts["tn"] += int(np.sum((post_bin == 0) & (gt_bin == 0)))

                if len(vis_paths) < num_vis_samples:
                    vis_paths.append(paths[i])
                    vis_gt.append(g)
                    vis_raw.append(p)
                    vis_post.append(post)

    # Compute Metrics
    raw_metrics = compute_metrics_from_counts(raw_counts)
    post_metrics = compute_metrics_from_counts(post_counts)

    print("\n--- RAW MODEL METRICS (Before Post-Processing) ---")
    print_metrics_summary(raw_metrics, split_name="Raw Predictions")

    print("\n--- REFINED METRICS (After Post-Processing) ---")
    print_metrics_summary(post_metrics, split_name="Post-Processed")

    # Generate Visual Overlays
    vis_path = os.path.join(args.output_dir, "evaluation_visual_overlay.png")
    plot_qualitative_results(vis_paths, vis_gt, vis_raw, vis_post, save_path=vis_path, num_samples=num_vis_samples)
    print(f"\nEvaluation finished successfully. Diagnostic figures saved to '{vis_path}'.")

if __name__ == "__main__":
    main()
