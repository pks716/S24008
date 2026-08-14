# Load Nify
# Create slices based on pre_processing

"""
DO NOT EDIT. CONTROL FROM SLICER PARAMETERS.

All the output scans will be saved in the destination folder as
    patient_name
        ct
            -1.npy
            -2.npy
            ...
        mr
            -1.npy
            -2.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import argparse
from multiprocessing import Pool
import cv2

from slicer_parameters import view, view_index_hh, destination_directory, data_directory, full_dataset, all_slices, per_sample_normalize, exclude_start_slices, exclude_end_slices, slice_size, padding_size
from pre_processing import pre_processing_order

view_key, view_index = False, 4
for key, item in view.items():
    if item:
        view_key = key
        view_index = view_index_hh[view_key]

if not view_key:
    raise ValueError("Select a view from slicer_parameters.py")

path = 'data_original/train'
path2 = 'data_fin/val'

path = data_directory
destination = destination_directory

num_cores = 24


def fetch_scan_paths(parent_foler):
    patients = os.listdir(parent_foler)
    patients = sorted(patients)
    patients_paths = []
    for patient in patients:
        if patient == '.DS_Store':
            continue
        if patient == 'overview':
            continue
        patients_paths.append(f"{parent_foler}/{patient}")
    
    return patients_paths

def read_nifti_to_numpy(file_path):
    try:
        nifti_image = nib.load(file_path)
        numpy_array = nifti_image.get_fdata()
        return numpy_array

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def crop_and_pad_with_mask(image_slice, mask_slice, padding_size=(608, 608), target_size=(256, 256)):
    """
    Crop the image slice using the mask to extract anatomy region,
    then pad to padding_size while preserving aspect ratio, finally resize to target_size.
    
    Args:
        image_slice: 2D numpy array of the image slice
        mask_slice: 2D numpy array of the mask slice (binary or non-zero values indicate anatomy)
        padding_size: tuple of (height, width) for intermediate padding size
        target_size: tuple of (height, width) for final output size
    
    Returns:
        Processed image slice of target_size
    """
    # Convert mask to binary if not already
    binary_mask = (mask_slice > 0).astype(np.uint8)
    
    # Find bounding box of the mask
    coords = np.where(binary_mask)
    if len(coords[0]) == 0:
        # If no mask found, return resized original image
        print("Warning: Empty mask found, returning resized original")
        return cv2.resize(image_slice, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Get bounding box coordinates
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Add small padding around the bounding box (optional, can be adjusted)
    padding = 10
    y_min = max(0, y_min - padding)
    y_max = min(image_slice.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image_slice.shape[1], x_max + padding)
    
    # Crop the image to the bounding box
    cropped_image = image_slice[y_min:y_max, x_min:x_max]
    
    # Calculate aspect ratio of cropped region
    crop_h, crop_w = cropped_image.shape
    padding_h, padding_w = padding_size
    
    # Calculate scaling to fit the cropped image into padding size while preserving aspect ratio
    scale_h = padding_h / crop_h
    scale_w = padding_w / crop_w
    scale = min(scale_h, scale_w)
    
    # Resize cropped image to fit within padding size
    new_h = int(crop_h * scale)
    new_w = int(crop_w * scale)
    resized_cropped = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image of padding_size
    padded_image = np.zeros(padding_size, dtype=image_slice.dtype)
    
    # Calculate position to center the resized cropped image
    start_y = (padding_h - new_h) // 2
    start_x = (padding_w - new_w) // 2
    
    # Place the resized cropped image in the center of padded image
    padded_image[start_y:start_y + new_h, start_x:start_x + new_w] = resized_cropped
    
    # Finally resize to target size
    final_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)
    
    return final_image


def populate(scan_path,
            destination_directory):
    # print(scan_path.split('/'))
    patient_name = scan_path.split('/')[-1]

    print("Splitting ", patient_name, "  scan.")
    
    if not os.path.exists(f'{destination_directory}/{patient_name}/mr/processed'):
        os.makedirs(f'{destination_directory}/{patient_name}/mr/processed')
    if not os.path.exists(f'{destination_directory}/{patient_name}/ct/processed'):
        os.makedirs(f'{destination_directory}/{patient_name}/ct/processed')
    if not os.path.exists(f'{destination_directory}/{patient_name}/mask/processed'):
        os.makedirs(f'{destination_directory}/{patient_name}/mask/processed')
        
    if not os.path.exists(f'{destination_directory}/{patient_name}/mr/original'):
        os.makedirs(f'{destination_directory}/{patient_name}/mr/original')
    if not os.path.exists(f'{destination_directory}/{patient_name}/ct/original'):
        os.makedirs(f'{destination_directory}/{patient_name}/ct/original')
    if not os.path.exists(f'{destination_directory}/{patient_name}/mask/original'):
        os.makedirs(f'{destination_directory}/{patient_name}/mask/original')
        
    def read_and_convert(scan_path):
        print("Loading", scan_path)
        mr_vol = read_nifti_to_numpy(f'{scan_path}/mr.nii.gz')
        ct_vol = read_nifti_to_numpy(f'{scan_path }/ct.nii.gz')
        mask_vol = read_nifti_to_numpy(f'{scan_path }/mask.nii.gz')

        # print(mr_vol, ct_vol)
        if mr_vol.shape != ct_vol.shape or mr_vol.shape != mask_vol.shape:
            print("Volume size mismatch at index--", scan_path)

        "train/{patient_name}/mri.---"
        if view_index > 2:
            raise ValueError("Set view in slicer_parameters as axial, coronal or sagittal")

        mr_max, mr_min = np.max(mr_vol), np.min(mr_vol)
        ct_vol = np.clip(ct_vol, -1024, 3000)
        ct_max, ct_min = np.max(ct_vol), np.min(ct_vol)
        
        # Calculate slice range with exclusions
        total_slices = mr_vol.shape[view_index]
        start_idx = exclude_start_slices
        end_idx = total_slices - exclude_end_slices
        
        # Ensure we don't have invalid ranges
        if start_idx >= end_idx:
            print(f"Warning: Volume {patient_name} has too few slices ({total_slices}) for exclusion parameters")
            start_idx = 0
            end_idx = total_slices
        
        for l in range(start_idx, end_idx):
            if not all_slices:
                # Original logic for limiting slices when all_slices is False
                if l < mr_vol.shape[view_index]//2:
                    continue
                if l > (mr_vol.shape[view_index]//2 + 5):
                    break
                
            if view_key == 'axial':                
                mr_slice = mr_vol[:,:,l]
                ct_slice = ct_vol[:,:,l]
                mask = mask_vol[:,:,l]
            elif view_key == 'coronal':
                mr_slice = mr_vol[:,l,:]
                ct_slice = ct_vol[:,l,:]
                mask = mask_vol[:,l,:]
            elif view_key == 'saggital':
                mr_slice = mr_vol[l,:,:]
                ct_slice = ct_vol[l,:,:]
                mask = mask_vol[l,:,:]
            else:
                raise ValueError("Set view in slicer_parameters as axial, coronal or saggital")
            
            if per_sample_normalize:
                mr_slice = (mr_slice-mr_min)/(mr_max-mr_min)
                ct_slice = (ct_slice-ct_min)/(ct_max-ct_min)
            
            # Apply mask-based cropping and padding BEFORE other preprocessing
            mr_slice_cropped = crop_and_pad_with_mask(mr_slice, mask,(padding_size, padding_size), slice_size)
            ct_slice_cropped = crop_and_pad_with_mask(ct_slice, mask,(padding_size, padding_size), slice_size)
            
            # Also process the mask itself for consistency
            mask_cropped = crop_and_pad_with_mask(mask.astype(np.float32), mask, (padding_size, padding_size), slice_size)
            
            ct_slice_processed, mr_slice_processed, mask_processed = np.copy(ct_slice_cropped), np.copy(mr_slice_cropped), np.copy(mask_cropped)

            # This is for extra pre-processing steps on training data. The original data should remain untouched.
            for method, target in pre_processing_order:
                if 'ct' in target:
                    ct_slice_processed = method(ct_slice_processed)
                if 'mri' in target:
                    mr_slice_processed = method(mr_slice_processed)
                if 'mask' in target:
                    mask_processed = method(mask_processed)
            
            np.savez_compressed(f'{destination_directory}/{patient_name}/mr/processed/{patient_name}_processed_{view_key}_{l}.npz', mr_slice_processed)
            np.savez_compressed(f'{destination_directory}/{patient_name}/ct/processed/{patient_name}_processed_{view_key}_{l}.npz', ct_slice_processed)
            np.savez_compressed(f'{destination_directory}/{patient_name}/mask/processed/{patient_name}_processed_{view_key}_{l}.npz', mask_processed)
                
            # Save the cropped versions as "original" since they're the base after mask cropping
            np.savez_compressed(f'{destination_directory}/{patient_name}/mr/original/{patient_name}_original_{view_key}_{l}.npz', mr_slice_cropped)
            np.savez_compressed(f'{destination_directory}/{patient_name}/ct/original/{patient_name}_original_{view_key}_{l}.npz', ct_slice_cropped)
            np.savez_compressed(f'{destination_directory}/{patient_name}/mask/original/{patient_name}_original_{view_key}_{l}.npz', mask_cropped)
            print("Saved to", f"{destination_directory}/{patient_name}")

    read_and_convert(scan_path)

mp_args = []

for paths in fetch_scan_paths(path):
    mp_args.append((paths,destination))

if __name__ == "__main__":
    if  not full_dataset:
        mp_args = mp_args[:10]
    with Pool(num_cores) as pool_fn:
        pool_fn.starmap(populate, mp_args)