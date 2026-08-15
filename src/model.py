"""
Model architectures and loss functions for Dental X-ray Segmentation.
Supports segmentation_models_pytorch (U-Net, U-Net++, DeepLabV3+) with fallback PyTorch U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


class DiceLoss(nn.Module):
    """Dice Loss for binary image segmentation."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss for handling foreground/background pixel imbalance in dental X-rays.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FallbackUNet(nn.Module):
    """
    Standalone PyTorch U-Net architecture fallback when segmentation_models_pytorch is not installed.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(FallbackUNet, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final_conv(d1)


def build_segmentation_model(
    architecture="unet",
    encoder_name="resnet34",
    in_channels=3,
    classes=1,
    encoder_weights="imagenet"
):
    """
    Factory function for building segmentation models.
    
    Args:
        architecture (str): 'unet', 'unetplusplus', or 'deeplabv3plus'
        encoder_name (str): Backbone encoder ('resnet34', 'resnet50', 'efficientnet-b2')
        in_channels (int): Input image channels (3 for RGB)
        classes (int): Number of output segmentation mask channels (1 for binary)
        encoder_weights (str): 'imagenet' or None
    """
    if HAS_SMP:
        print(f"Building SMP Model Architecture: {architecture.upper()} with Encoder: {encoder_name}")
        arch_lower = architecture.lower()
        weights = encoder_weights if encoder_weights != "none" else None
        
        if arch_lower == "unet":
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=classes
            )
        elif arch_lower == "unetplusplus":
            model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=classes
            )
        elif arch_lower == "deeplabv3plus":
            model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=weights,
                in_channels=in_channels,
                classes=classes
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        return model
    else:
        print("segmentation_models_pytorch not found. Falling back to native PyTorch FallbackUNet.")
        return FallbackUNet(in_channels=in_channels, out_channels=classes)
