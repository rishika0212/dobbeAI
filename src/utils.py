"""
Utility functions for Exploratory Data Analysis (EDA), visualization of segmentation overlays,
and dataset helper routines.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_color_overlay(image, gt_mask, pred_mask, alpha=0.4):
    """
    Generates a color-coded diagnostic overlay:
    - Green = Ground Truth
    - Red = Model Prediction
    - Yellow = True Positive (Overlap)
    
    Args:
        image (np.ndarray): HxWx3 uint8 RGB image
        gt_mask (np.ndarray): HxW binary ground truth mask
        pred_mask (np.ndarray): HxW binary prediction mask
        alpha (float): Transparency blending weight
    """
    overlay = image.copy()
    
    gt_bin = (gt_mask > 0.5)
    pred_bin = (pred_mask > 0.5)
    
    tp = gt_bin & pred_bin
    fp = pred_bin & ~gt_bin
    fn = gt_bin & ~pred_bin

    # False Negatives (Missed Cavities) -> Cyan / Green tint
    overlay[fn] = (1.0 - alpha) * overlay[fn] + alpha * np.array([0, 255, 0])
    # False Positives (False Alarms) -> Red tint
    overlay[fp] = (1.0 - alpha) * overlay[fp] + alpha * np.array([255, 0, 0])
    # True Positives (Accurate Cavity Detection) -> Yellow tint
    overlay[tp] = (1.0 - alpha) * overlay[tp] + alpha * np.array([255, 255, 0])

    return overlay.astype(np.uint8)


def plot_qualitative_results(image_paths, gt_masks, raw_preds, post_preds, save_path=None, num_samples=4):
    """
    Generates side-by-side diagnostic figures comparing:
    Original X-ray | Ground Truth Mask | Model Prediction | Post-Processed Mask | Diagnostic Overlay
    """
    n = min(len(image_paths), num_samples)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["1. Original OPG", "2. Ground Truth", "3. Raw Prediction Map", "4. Post-Processed", "5. Diagnostic Overlay"]

    for i in range(n):
        # Load original image
        img = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        gt = gt_masks[i]
        raw = raw_preds[i]
        post = post_preds[i]
        
        # Resize GT & Preds to match original image size if needed
        if gt.shape != img.shape[:2]:
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if raw.shape != img.shape[:2]:
            raw = cv2.resize(raw, (img.shape[1], img.shape[0]))
        if post.shape != img.shape[:2]:
            post = cv2.resize(post.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = create_color_overlay(img, gt, post)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 2].imshow(raw, cmap="magma")
        axes[i, 3].imshow(post, cmap="gray")
        axes[i, 4].imshow(overlay)

        for j in range(5):
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=13, fontweight='bold')
            axes[i, j].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved qualitative inspection figure to: {save_path}")
    plt.close()


def plot_eda_summary(image_paths, mask_paths=None, save_path=None):
    """
    Generates Exploratory Data Analysis (EDA) summary plots:
    - Image resolutions distribution
    - Mask coverage / lesion area percentage histogram
    - Sample image grid
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    widths, heights, aspect_ratios = [], [], []
    mask_coverages = []

    for img_p in image_paths[:500]: # Sample up to 500 for speed
        img = cv2.imread(img_p)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

    if mask_paths:
        for m_p in mask_paths[:500]:
            if m_p and os.path.exists(m_p):
                mask = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    coverage = (mask > 127).sum() / (mask.shape[0] * mask.shape[1]) * 100
                    mask_coverages.append(coverage)

    # Plot 1: Image Dimensions Scatter
    axes[0].scatter(widths, heights, alpha=0.5, color="teal")
    axes[0].set_title("1. OPG Image Resolution Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Aspect Ratio Histogram
    axes[1].hist(aspect_ratios, bins=20, color="indigo", edgecolor="black", alpha=0.7)
    axes[1].set_title("2. Panoramic Aspect Ratio Distribution (W/H)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Aspect Ratio")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Plot 3: Mask Coverage / Imbalance
    if mask_coverages:
        axes[2].hist(mask_coverages, bins=25, color="crimson", edgecolor="black", alpha=0.7)
        axes[2].set_title("3. Cavity/Lesion Area Coverage per Image (%)", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Mask Pixel Coverage (%)")
        axes[2].set_ylabel("Number of Images")
        axes[2].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[2].text(0.5, 0.5, "Mask coverage data unavailable", ha="center", va="center")
        axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved EDA summary plot to: {save_path}")
    plt.close()
