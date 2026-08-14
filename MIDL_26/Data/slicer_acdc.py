# slicer_acdc.py - BLIND SR VERSION (MRI + MASKS)

"""
ACDC Blind Super-Resolution Slicer
Extracts all slices from 3D MRI volumes and corresponding masks

Input structure:
master_folder/
    patient001/
        patient001_frame01.nii.gz     # 3D volume (H×W×Slices)
        patient001_frame01_gt.nii.gz  # 3D mask
        patient001_frame12.nii.gz
        patient001_frame12_gt.nii.gz
    patient002/
        ...

Outputs:
patient001/
    patient001_frame01_slice00.npy     # High-res MRI (256×256)
    patient001_frame01_slice00_gt.npy  # High-res Mask (256×256)
    patient001_frame01_slice01.npy
    patient001_frame01_slice01_gt.npy
    ...
"""

import numpy as np
import os
import nibabel as nib
import cv2
from multiprocessing import Pool
import glob

from slicer_parameters_acdc import (
    data_directory, 
    destination_directory,
    full_dataset,
    all_slices,
    TARGET_SIZE
)
from pre_processing_acdc import pre_processing_order

num_cores = 8


def load_nifti_volume(file_path):
    """Load .nii.gz 3D volume using nibabel"""
    try:
        nii = nib.load(file_path)
        array = nii.get_fdata()
        array = np.squeeze(array)  # Remove singleton dimensions
        
        # Ensure 3D (H, W, Slices)
        if array.ndim == 2:
            array = array[:, :, np.newaxis]  # Add slice dimension
        
        return array
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def resize_to_target(image, target_size=(256, 256), is_mask=False):
    """
    Resize 2D slice to target size
    For masks, use nearest neighbor interpolation to preserve labels
    """
    if image.shape[:2] == target_size:
        return image
    
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(image, target_size, interpolation=interpolation)


def get_acdc_frame_files(patient_dir, patient_name):
    """
    Get list of ACDC frame files to process
    Returns: list of file paths matching patient###_frame##.nii.gz
    EXCLUDES ground truth masks (files with '_gt' in name)
    """
    # Pattern: patient###_frame##.nii.gz
    pattern = os.path.join(patient_dir, f"{patient_name}_frame*.nii.gz")
    all_files = sorted(glob.glob(pattern))
    
    # Filter out ground truth masks (files containing '_gt')
    frame_files = [f for f in all_files if '_gt' not in os.path.basename(f)]
    
    return frame_files


def process_acdc_patient(patient_path):
    """
    Process a single ACDC patient - extract all slices from all frames and masks
    """
    patient_name = os.path.basename(patient_path)
    
    if not os.path.isdir(patient_path):
        return
    
    print(f"Processing {patient_name}...")
    
    frame_files = get_acdc_frame_files(patient_path, patient_name)
    
    if len(frame_files) == 0:
        print(f"  No frame files found for {patient_name}, skipping.")
        return
    
    print(f"  Found {len(frame_files)} frames")
    
    # Create output directory
    os.makedirs(f'{destination_directory}/{patient_name}', exist_ok=True)
    
    total_slices_saved = 0
    
    # Process each frame (3D volume)
    for frame_path in frame_files:
        # Extract frame number from filename
        # e.g., patient001_frame01.nii.gz -> frame01
        frame_basename = os.path.basename(frame_path)
        frame_name = frame_basename.replace('.nii.gz', '')
        frame_num = frame_name.split('_frame')[-1]
        
        # Construct mask path
        mask_path = frame_path.replace('.nii.gz', '_gt.nii.gz')
        
        # Load 3D volume
        volume = load_nifti_volume(frame_path)
        
        if volume is None:
            print(f"  Failed to load {frame_basename}")
            continue
        
        # Load corresponding mask
        mask_volume = None
        if os.path.exists(mask_path):
            mask_volume = load_nifti_volume(mask_path)
            if mask_volume is None:
                print(f"  Warning: Failed to load mask for {frame_basename}")
        else:
            print(f"  Warning: No mask found for {frame_basename}")
        
        # Get number of slices
        if volume.ndim == 2:
            num_slices = 1
            volume = volume[:, :, np.newaxis]
            if mask_volume is not None:
                mask_volume = mask_volume[:, :, np.newaxis]
        else:
            num_slices = volume.shape[2]
        
        # Process each slice
        for slice_idx in range(num_slices):
            # Extract 2D slice
            mri_slice = volume[:, :, slice_idx]
            
            # Skip empty slices
            if mri_slice.max() == 0:
                continue
            
            # Resize to target size
            mri_resized = resize_to_target(mri_slice, TARGET_SIZE, is_mask=False)
            
            # Apply preprocessing
            mri_processed = np.copy(mri_resized)
            
            for method, targets in pre_processing_order:
                if 'mri' in targets:
                    mri_processed = method(mri_processed)
            
            # Save HIGH-RESOLUTION MRI slice
            filename = f"{patient_name}_frame{frame_num}_slice{slice_idx:02d}.npy"
            
            np.save(
                f'{destination_directory}/{patient_name}/{filename}',
                mri_processed
            )
            
            # Process and save mask if available
            if mask_volume is not None:
                mask_slice = mask_volume[:, :, slice_idx]
                
                # Resize mask (using nearest neighbor) - NO PREPROCESSING!
                mask_resized = resize_to_target(mask_slice, TARGET_SIZE, is_mask=True)
                
                # DO NOT APPLY PREPROCESSING TO MASKS!
                # Masks must remain as integer class labels (0, 1, 2, 3)
                # Save mask directly after resizing
                
                # Save HIGH-RESOLUTION mask slice
                mask_filename = f"{patient_name}_frame{frame_num}_slice{slice_idx:02d}_gt.npy"
                
                np.save(
                    f'{destination_directory}/{patient_name}/{mask_filename}',
                    mask_resized  # Save without any preprocessing
                )
            
            total_slices_saved += 1
        
        print(f"  ✓ Processed frame{frame_num}: {num_slices} slices")
    
    print(f"  Total slices saved: {total_slices_saved}")


def fetch_patient_paths(parent_folder):
    """Get list of patient directories"""
    if not os.path.exists(parent_folder):
        raise ValueError(f"Data directory not found: {parent_folder}")
    
    patients = sorted(os.listdir(parent_folder))
    patient_paths = []
    
    for patient in patients:
        # Skip hidden files and non-directories
        if patient.startswith('.'):
            continue
        
        patient_path = os.path.join(parent_folder, patient)
        if os.path.isdir(patient_path):
            patient_paths.append(patient_path)
    
    return patient_paths


if __name__ == "__main__":
    print("="*60)
    print("ACDC Blind Super-Resolution Slicer")
    print("MRI images + Masks - Extracts all slices from 3D volumes")
    print("="*60)
    
    patient_paths = fetch_patient_paths(data_directory)
    
    print(f"Found {len(patient_paths)} patients in {data_directory}")
    
    if not full_dataset:
        patient_paths = patient_paths[:5]
        print(f"Testing mode: Processing only {len(patient_paths)} patients")
    
    print(f"\nProcessing with {num_cores} cores...")
    with Pool(num_cores) as pool:
        pool.map(process_acdc_patient, patient_paths)
    
    print("\n" + "="*60)
    print("✓ ACDC processing complete!")
    print(f"  Output directory: {destination_directory}")
    print("="*60)