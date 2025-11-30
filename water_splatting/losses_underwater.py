import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Helper Function: Differentiable RGB to LAB (Native PyTorch)
# Purpose: Converts sRGB images to CIELAB space for perceptually uniform
# color distance calculations. No Kornia dependency required.
# ==============================================================================
def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    Converts an RGB image to LAB color space (Differentiable).
    Input: RGB [0, 1]
    Output: LAB (L: 0-100, a: -128~127, b: -128~127)
    """
    # 1. RGB to XYZ
    # Linearization (Inverse Gamma Correction) for sRGB
    mask = rgb > 0.04045
    rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # RGB to XYZ matrix (Reference D65)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    xyz = torch.stack([x, y, z], -1)

    # 2. XYZ to LAB
    # Reference white point (D65)
    xyz_ref_white = torch.tensor([0.95047, 1.00000, 1.08883], device=rgb.device, dtype=rgb.dtype)
    xyz = xyz / xyz_ref_white
    
    # Nonlinear projection function
    # threshold 0.008856 is approx (6/29)^3
    mask = xyz > 0.008856
    xyz = torch.where(mask, torch.pow(xyz, 1.0/3.0), 7.787 * xyz + 16.0 / 116.0)
    
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    
    L = 116.0 * y - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    
    return torch.stack([L, a, b], -1)


# ==============================================================================
# Loss 1: Monocular Depth Loss (Geometry)
# Purpose: Core geometry reinforcement. Forces 3DGS depth to align with 
# the structure predicted by monocular depth models (e.g., Depth Anything).
# Uses Pearson Correlation to be scale-invariant.
# ==============================================================================
class MonocularDepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_depth, mono_depth, mask=None):
        """
        Args:
            pred_depth: Depth rendered by 3DGS (H, W, 1)
            mono_depth: Ground truth depth from monocular model (H, W, 1)
            mask: Optional mask
        Returns:
            1 - Pearson Correlation (Range 0~2, lower is better)
        """
        # Flatten data
        pred = pred_depth.reshape(-1)
        target = mono_depth.reshape(-1)

        if mask is not None:
            mask = mask.reshape(-1)
            pred = pred[mask]
            target = target[mask]
        
        # Epsilon for numerical stability
        eps = 1e-6

        # Standardization: subtract mean, divide by std
        # This makes the loss Scale-Invariant
        pred_norm = (pred - pred.mean()) / (pred.std() + eps)
        target_norm = (target - target.mean()) / (target.std() + eps)

        # Calculate Pearson Correlation Coefficient
        correlation = (pred_norm * target_norm).mean()

        return 1.0 - correlation


# ==============================================================================
# Loss 2: Opacity Loss V3 (De-hazing / De-artifact)
# Purpose: Removes floating artifacts. If a pixel's color is very similar
# to the water background color, force its opacity (alpha) to zero.
# Features: Uses LAB color space + Soft Weighting (exp).
# ==============================================================================
class OpacityLoss(nn.Module):
    def __init__(self, threshold_sim=2.5, soft_scale=1.0):
        """
        Args:
            threshold_sim: Distance threshold in LAB space.
                           Recommended: 2.5 ~ 5.0 for LAB.
            soft_scale: Controls the steepness of the exponential decay.
        """
        super().__init__()
        self.threshold_sim = threshold_sim
        self.soft_scale = soft_scale

    def forward(self, pred_image, background_rgb, accumulation):
        """
        Args:
            pred_image: (H, W, 3) RGB [0, 1]
            background_rgb: (H, W, 3) RGB [0, 1]
            accumulation: (H, W, 1) Alpha [0, 1]
        """
        # 1. Convert to LAB space (Clamped to prevent NaN)
        lab_pred = rgb_to_lab(torch.clamp(pred_image, 1e-6, 1.0))
        lab_bg = rgb_to_lab(torch.clamp(background_rgb, 1e-6, 1.0))

        # 2. Calculate Euclidean distance in LAB space
        # dim=-1 computes norm across L, a, b channels
        diff = lab_pred - lab_bg
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True) # (H, W, 1)

        # 3. Calculate Soft Mask
        # If dist < threshold, clamped_diff = 0 -> exp(0) = 1 (Full Penalty)
        # If dist > threshold, clamped_diff > 0 -> exp(-x) decays to 0 (No Penalty)
        clamped_diff = F.relu(dist - self.threshold_sim)
        mask = torch.exp(-clamped_diff * self.soft_scale)
        
        # 4. Calculate Loss
        # Only penalize accumulation (alpha), detach mask to stop gradient flow to color
        loss = (accumulation * mask.detach()).mean()
        
        return loss


# ==============================================================================
# Loss 3: Gray World Loss (Color Restoration)
# Purpose: Color correction. Enforces the "Gray World Assumption" on the 
# restored object color, preventing global color shifts (e.g., overly blue/green).
# ==============================================================================
class GrayWorldLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rgb_object):
        """
        Args:
            rgb_object: Restored object color J (H, W, 3)
        """
        # Calculate mean of R, G, B channels across the image
        mean_channels = torch.mean(rgb_object, dim=[0, 1]) # (3,)
        
        # Penalize deviation from 0.5 (gray)
        loss = torch.mean((mean_channels - 0.5) ** 2)
        return loss


# ==============================================================================
# Loss 4: Saturation Loss (Color Restoration)
# Purpose: Color correction. Penalizes pixels that are oversaturated 
# (RGB values exceeding a certain threshold).
# ==============================================================================
class SaturationLoss(nn.Module):
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold = threshold

    def forward(self, rgb_object):
        """
        Args:
            rgb_object: Restored object color J (H, W, 3)
        """
        # Penalize values exceeding the threshold
        # ReLU(x - T) is equivalent to max(x - T, 0)
        excess = F.relu(rgb_object - self.threshold)
        loss = excess.mean()
        return loss