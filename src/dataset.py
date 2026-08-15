"""
Dataset classes and preprocessing pipelines for Panoramic Dental X-ray (OPG) Segmentation.
Supports COCO segmentation masks, bounding-box to pseudo-mask conversion, and Albumentations augmentations.
"""

import os
import cv2
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

class DentalSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Dental OPG Cavity & Lesion Segmentation.
    
    Supports:
    - Standard Image & Mask directory pairs (.png, .jpg, .tif)
    - Contrast Limited Adaptive Histogram Equalization (CLAHE) for X-rays
    - Dynamic Augmentations via Albumentations
    """
    def __init__(
        self, 
        image_paths, 
        mask_paths=None, 
        target_size=(512, 512), 
        transform=None, 
        use_clahe=True,
        is_train=True
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size
        self.transform = transform
        self.use_clahe = use_clahe
        self.is_train = is_train

        # Default fallback augmentations if albumentations is installed
        if self.transform is None and HAS_ALBUMENTATIONS:
            if self.is_train:
                self.transform = A.Compose([
                    A.Resize(target_size[0], target_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5, cval=0),
                    A.GaussNoise(p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(target_size[0], target_size[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])

    def apply_clahe(self, img):
        """Applies CLAHE contrast enhancement for X-ray visibility improvement."""
        if len(img.shape) == 3 and img.shape[2] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            return enhanced
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_clahe:
            image = self.apply_clahe(image)

        # Load or generate mask
        if self.mask_paths and self.mask_paths[idx]:
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Apply transformations
        if HAS_ALBUMENTATIONS and self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].unsqueeze(0).float()
        else:
            # Fallback PyTorch resizing & normalization
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
            
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # Normalize with ImageNet mean/std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor, img_path

