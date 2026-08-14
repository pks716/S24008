import cv2
import numpy as np
from slicer_parameters import slice_size, interpolation_method

"""
Define Preprocessing steps here. Write numpy slice functions. Expected function type:
def f_name(slice):
    # BOC
    return processed_slice
"""

def place_holder_fn(ele):
    # Apply this logic on ct and mri
    return ele

def resize(slice):
    """Resize slice to target size - Note: This is now applied after mask-based cropping"""
    return cv2.resize(slice, slice_size, interpolation=interpolation_method)

def normalize_mask(mask_slice):
    """Ensure mask values are in [0, 1] range"""
    if mask_slice.max() > 1:
        return (mask_slice > 0).astype(np.float32)
    return mask_slice.astype(np.float32)

def enhance_contrast(slice, alpha=1.2, beta=10):
    """Apply contrast enhancement (optional preprocessing step)"""
    return np.clip(alpha * slice + beta, slice.min(), slice.max())

def gaussian_blur(slice, kernel_size=3):
    """Apply Gaussian blur for noise reduction (optional preprocessing step)"""
    return cv2.GaussianBlur(slice.astype(np.float32), (kernel_size, kernel_size), 0)

# Since mask-based cropping and padding to slice_size is now handled in the main script,
# we can remove the resize step from preprocessing or keep it for additional processing
pre_processing_order = [
    (place_holder_fn, ['ct', 'mri']),
    (normalize_mask, ['mask'])
    # Add any additional preprocessing steps here if needed
    # (enhance_contrast, ['ct', 'mri']),
    # (gaussian_blur, ['ct', 'mri'])
]

# Validate preprocessing order
for method, targets in pre_processing_order:
    for ele in targets:
        if ele not in ['ct', 'mri', 'mask']:
            raise ValueError('Incorrect modalities specified in pre_processing_order, pre_processing.py')