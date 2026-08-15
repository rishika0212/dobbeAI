"""
Evaluation metrics module for image segmentation.
Calculates Precision, Recall, F1-Score (Dice Score), and Intersection-over-Union (IoU).
"""

import numpy as np
import torch

def compute_segmentation_metrics(pred_masks, gt_masks, eps=1e-7):
    """
    Computes global segmentation metrics across arrays or PyTorch tensors of masks.
    
    Args:
        pred_masks: Binary predictions matrix (N, H, W) or (H, W) [0 or 1]
        gt_masks: Ground truth masks matrix (N, H, W) or (H, W) [0 or 1]
        eps: Small constant for numerical stability

    Returns:
        dict: Containing Precision, Recall, F1_Score (Dice), IoU, TP, FP, FN, TN.
    """
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.detach().cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.detach().cpu().numpy()

    pred_binary = (pred_masks > 0.5).astype(np.uint8).flatten()
    gt_binary = (gt_masks > 0.5).astype(np.uint8).flatten()

    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1_score = (2 * tp + eps) / (2 * tp + fp + fn + eps) # Equivalent to Dice Score
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "dice_score": float(f1_score),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

def print_metrics_summary(metrics, split_name="Test"):
    """Prints formatted metrics summary to terminal."""
    print("=" * 45)
    print(f"       SEGMENTATION METRICS ({split_name.upper()} SET)")
    print("=" * 45)
    print(f" Precision          : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f" Recall (Sensitivity): {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f" F1-Score / Dice    : {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f" IoU (Jaccard Index): {metrics['iou']:.4f} ({metrics['iou']*100:.2f}%)")
    print("-" * 45)
    print(f" Pixel Counts       : TP={metrics['tp']:,} | FP={metrics['fp']:,} | FN={metrics['fn']:,} | TN={metrics['tn']:,}")
    print("=" * 45)
