# pre_processing_acdc.py

import cv2
import numpy as np
from slicer_parameters_acdc import slice_size, interpolation_method

"""
Minimal preprocessing for ACDC cardiac MRI images
"""

def normalize_acdc(slice):
    """
    Normalize ACDC MRI images to [0, 1]
    ACDC images can have varying intensity ranges
    """
    slice = slice.astype(np.float32)
    
    # Normalize to [0, 1] based on actual min/max
    slice_min = slice.min()
    slice_max = slice.max()
    
    if slice_max > slice_min:
        slice = (slice - slice_min) / (slice_max - slice_min)
    else:
        slice = np.zeros_like(slice)
    
    return np.clip(slice, 0.0, 1.0)


def enhance_mri_contrast(slice, clip_limit=2.0):
    """
    Optional: Apply CLAHE for MRI contrast enhancement
    """
    # Convert to uint8 for CLAHE
    slice_uint8 = (slice * 255).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(slice_uint8)
    
    # Convert back to float [0, 1]
    return enhanced.astype(np.float32) / 255.0


# Preprocessing pipeline - MINIMAL for super-resolution
pre_processing_order = [
    (normalize_acdc, ['mri']),  # Just normalize to [0, 1]
    # (enhance_mri_contrast, ['mri']),  # Optional: Uncomment if needed
]

# Validate
for method, targets in pre_processing_order:
    for ele in targets:
        if ele not in ['mri']:
            raise ValueError(f'Incorrect modality "{ele}". Use "mri" only.')