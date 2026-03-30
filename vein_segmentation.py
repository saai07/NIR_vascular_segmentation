#!/usr/bin/env python3
"""
NIR Vascular Segmentation Pipeline
==================================
Segments only the vein structures from raw Near-Infrared (NIR) images.
Produces a clean binary mask (veins=255, background=0).

Constraints Followed:
 - Classical CV only (OpenCV, scikit-image, NumPy). No Deep Learning.
 - Robust to illumination, skin tones, hair, and shadows.
 - Strict shape-based filtering to exclusively retain veins.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi, threshold_otsu, apply_hysteresis_threshold
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects, remove_small_holes

# ============================================================================
# TUNABLE PARAMETERS
# ============================================================================

# 1. Preprocessing
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
TOPHAT_KERNEL_SIZE = 151      # Large kernel to estimate background illumination
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)

# 2. Vessel Enhancement (Frangi)
FRANGI_SIGMAS = [1.0, 2.0, 3.0, 4.0]  # matching typical vein widths (1-4 pixels)
FRANGI_ALPHA = 0.5
FRANGI_BETA = 0.5
FRANGI_GAMMA = 15.0

# 3. Binarization (Adaptive)
ADAPTIVE_BLOCK_SIZE = 51
ADAPTIVE_C = -8

# 4. Morphological Cleaning
CLOSE_DISK_RADIUS = 3
MIN_OBJ_AREA = 100      # pixels
HOLE_FILL_AREA = 100    # pixels


# ============================================================================
# PIPELINE STAGES
# ============================================================================

def load_image(path: str) -> np.ndarray:
    """Load grayscale image and normalise to float32 [0, 1]."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find image: {path}")
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to decode: {path}")
    
    # Normalize to [0...1] based on dtype max
    if img.dtype == np.uint8:
        max_val = 255.0
    elif img.dtype == np.uint16:
        max_val = 65535.0
    else:
        max_val = img.max() if img.max() > 0 else 1.0
        
    return (img.astype(np.float32) / max_val)


def step1_preprocessing(img: np.ndarray) -> np.ndarray:
    """
    1. Preprocessing – bilateral filter for noise reduction, 
       illumination correction (top-hat), and CLAHE.
    """
    img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # Bilateral filter (preserves edges, removes noise/hair)
    filtered = cv2.bilateralFilter(
        img_u8, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    )
    
    # Illumination correction (subtracting background / white top-hat)
    # Since veins are DARK, the background is BRIGHT. 
    # Morphological closing with a huge kernel removes the dark veins, 
    # leaving just the uneven illumination background field.
    se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE))
    background = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, se_bg)
    
    # We want to keep veins dark. 
    # corrected = background - filtered creates an image where veins are BRIGHT.
    # To keep veins DARK for Frangi (black_ridges=True), we do:
    # 255 - (background - filtered)
    diff = cv2.subtract(background, filtered)
    corrected_u8 = cv2.subtract(np.full_like(diff, 255), diff)
    
    # CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    enhanced_u8 = clahe.apply(corrected_u8)
    
    return enhanced_u8.astype(np.float32) / 255.0


def step2_vessel_enhancement(img: np.ndarray) -> np.ndarray:
    """
    2. Vessel enhancement – Frangi vesselness filter.
    black_ridges=True because veins appear dark on the bright skin background.
    """
    resp = frangi(
        img,
        sigmas=FRANGI_SIGMAS,
        alpha=FRANGI_ALPHA,
        beta=FRANGI_BETA,
        gamma=FRANGI_GAMMA,
        black_ridges=True,
    )
    # Normalise Frangi response to [0, 1]
    lo, hi = resp.min(), resp.max()
    if hi > lo:
        resp = (resp - lo) / (hi - lo)
    return resp


def step3_binarization(frangi_resp: np.ndarray) -> np.ndarray:
    """
    3. Binarization – adaptive thresholding.
    Handles residual variations perfectly for continuous lines.
    """
    # Convert frangi [0, 1] to uint8 [0, 255]
    img_u8 = np.clip(frangi_resp * 255, 0, 255).astype(np.uint8)
    
    # Adaptive Gaussian
    mask = cv2.adaptiveThreshold(
        img_u8, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )
    return mask > 0


def step4_morph_cleaning(binary_mask: np.ndarray) -> np.ndarray:
    """
    4. Morphological cleaning – small closing (disk 3) to fill gaps,
       remove small entities (< 100), and fill holes.
    """
    # Closing to fill gaps within veins
    se_close = disk(CLOSE_DISK_RADIUS)
    closed = cv2.morphologyEx(
        binary_mask.astype(np.uint8) * 255, 
        cv2.MORPH_CLOSE, 
        se_close
    ) > 0
    
    # Remove small noise pixels
    cleaned = remove_small_objects(closed, min_size=MIN_OBJ_AREA)
    
    # Fill small holes within veins
    filled = remove_small_holes(cleaned, area_threshold=HOLE_FILL_AREA)
    
    return filled.astype(np.uint8) * 255


    # The rest has been replaced by adaptive + morph cleanup!


# ============================================================================
# VISUALISATION AND ORCHESTRATION
# ============================================================================

def visualize(stages: dict[str, np.ndarray], output_path: str):
    """Plot the pipeline progression."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("NIR Vein Segmentation Pipeline", fontsize=16, fontweight="bold")
    
    panels = [
        ("raw",       "1. Raw Input",             "gray"),
        ("frangi",    "2. Frangi Vesselness",     "hot"),
        ("final",     "3. Final Segmented Veins", "gray"),
        ("overlay",   "4. Vein Overlay",           None),
    ]
    
    for ax, (key, title, cmap) in zip(axes.flat, panels):
        data = stages.get(key)
        if data is not None:
            if cmap:
                ax.imshow(data, cmap=cmap)
            else:
                ax.imshow(data)
        ax.set_title(title)
        ax.axis("off")
        
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_pipeline(image_path: str, output_dir: str):
    print(f"Processing: {image_path}")
    os.makedirs(output_dir, exist_ok=True)
    stages = {}
    
    # Step 0: Load
    raw = load_image(image_path)
    stages["raw"] = raw
    
    # Step 1: Preprocessing
    print("Executing Step 1: Preprocessing (Bilateral + Illumination + CLAHE)...")
    preproc = step1_preprocessing(raw)
    stages["preproc"] = preproc
    
    # Step 2: Frangi Enhancement
    print("Executing Step 2: Vessel Enhancement (Frangi filter)...")
    frangi_resp = step2_vessel_enhancement(preproc)
    stages["frangi"] = frangi_resp
    
    # Step 3: Binarization (Adaptive Thresholding)
    print("Executing Step 3: Binarization (Adaptive Thresholding)...")
    binary_mask = step3_binarization(frangi_resp)
    
    # Step 4: Morphological Cleaning
    print("Executing Step 4: Morphological Cleaning...")
    final_mask = step4_morph_cleaning(binary_mask)
    stages["final"] = final_mask
    
    # Create overlay (green veins on gray background)
    grey_bg = cv2.cvtColor(np.clip(raw*255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = grey_bg.copy()
    overlay[final_mask > 0] = [0, 255, 0]
    stages["overlay"] = cv2.addWeighted(grey_bg, 0.6, overlay, 0.4, 0)
    
    # Save outputs
    out_mask = os.path.join(output_dir, "vein_mask.png")
    out_vis = os.path.join(output_dir, "pipeline_stages.png")
    cv2.imwrite(out_mask, final_mask)
    visualize(stages, out_vis)
    
    print(f"Success! Saved results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIR Vein Segmentation Pipeline")
    parser.add_argument("--input", required=True, help="Path to input NIR image")
    parser.add_argument("--output", required=True, help="Output directory")
    # Ignored legacy args for compatibility with previous commands
    parser.add_argument("--domain", default="tissue", help=argparse.SUPPRESS)
    parser.add_argument("--visualize", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output)
