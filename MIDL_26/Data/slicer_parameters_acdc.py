# slicer_parameters_acdc.py

import cv2
import numpy as np

# ============================================================================
# ACDC-SPECIFIC PARAMETERS
# ============================================================================

# Dataset paths
data_directory = '/home/pks/Desktop/Peeyush/cardiac_work/ACDC/training/'  # ← CHANGE THIS to your ACDC master folder
destination_directory = '/home/pks/Desktop/Peeyush/cardiac_work/diffuison_work/Data/slices_acdc'  # ← CHANGE THIS

# Processing options
full_dataset = True   # Process all patients
all_slices = True     # Extract ALL slices from each 3D volume

TARGET_SIZE = (256, 256)  # Final image size

# ACDC-specific: Process all frames found in each patient folder
# Frames are named: patient###_frame##.nii.gz
# The script will automatically detect all frame numbers (00, 01, 02, etc.)

slice_size = TARGET_SIZE
interpolation_method = cv2.INTER_LINEAR

print(f"ACDC Slicer Configuration:")
print(f"  Data directory: {data_directory}")
print(f"  Output directory: {destination_directory}")
print(f"  Extract all slices: {all_slices}")
print(f"  Target size: {TARGET_SIZE}")