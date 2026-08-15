"""
Post-processing pipeline for refining raw model prediction maps, reducing false positives,
and applying morphological/spatial constraints on dental X-ray segmentations.
"""

import cv2
import numpy as np

def confidence_thresholding(prob_map, threshold=0.5):
    """Converts continuous probability map [0, 1] into a binary mask."""
    return (prob_map >= threshold).astype(np.uint8)

def morphological_refinement(binary_mask, kernel_size=3):
    """
    Applies morphological closing (fills interior lesion holes) 
    and opening (removes thin background noise artifacts).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Closing to bridge gaps in cavity predictions
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    # Opening to detach minor pixel speckles
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened

def filter_connected_components(binary_mask, min_area=100):
    """
    Removes small disconnected component predictions whose area in pixels is below min_area threshold.
    Crucial in OPG X-rays to suppress spurious false positive noise in bone marrow or air gap regions.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 1
            
    return cleaned_mask

def apply_post_processing(prob_map, threshold=0.5, min_area=100, do_morphology=True):
    """
    Full post-processing pipeline for single-channel probability output.
    
    Args:
        prob_map (np.ndarray): Probability matrix of shape (H, W) in [0, 1].
        threshold (float): Decision threshold for foreground assignment.
        min_area (int): Minimum pixel area for valid cavity/lesion connected component.
        do_morphology (bool): Whether to perform morphological closing & opening.

    Returns:
        np.ndarray: Refined binary mask (0 or 1) of shape (H, W).
    """
    # Step 1: Probability thresholding
    binary = confidence_thresholding(prob_map, threshold=threshold)
    
    # Step 2: Morphological refinement
    if do_morphology:
        binary = morphological_refinement(binary, kernel_size=3)
        
    # Step 3: Area-based connected component filtering
    if min_area > 0:
        binary = filter_connected_components(binary, min_area=min_area)
        
    return binary
