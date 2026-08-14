# train_loader_acdc_sr.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ACDCBlindSRDataset(Dataset):
    """
    ACDC Blind Super-Resolution Training Dataset
    
    - Loads high-res cardiac MRI images
    - Applies random MRI-specific degradation on-the-fly
    - Returns: (lowres_degraded, highres_original)
    """
    
    def __init__(self, paths, scale_factor=2, augment=True):
        """
        Args:
            paths: List of patient folder paths
            scale_factor: Downsampling factor (2 = 2× super-resolution)
            augment: Apply data augmentation
        """
        self.scale_factor = scale_factor
        self.augment = augment
        
        # Import MRI-specific blind degradation model
        from blind_degradation_acdc import BlindDegradationModelMRI
        self.degradation_model = BlindDegradationModelMRI(scale_factor=scale_factor)
        
        # Collect all .npy files from patient folders
        self.image_paths = []
        
        for patient_path in paths:
            if '.DS_Store' in patient_path:
                continue
            
            # ACDC structure: patient_folder/*.npy
            if os.path.isdir(patient_path):
                npy_files = [f for f in os.listdir(patient_path) if f.endswith('.npy')]
                for npy_file in npy_files:
                    full_path = os.path.join(patient_path, npy_file)
                    self.image_paths.append(full_path)
        
        print(f"ACDC Blind SR Dataset: {len(self.image_paths)} MRI slices loaded")
        print(f"  Scale factor: {scale_factor}×")
        print(f"  Augmentation: {augment}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load HIGH-RES MRI image
        highres_image = np.load(self.image_paths[idx])
        
        # Convert to float32 and ensure [0, 1] range
        highres_image = highres_image.astype(np.float32)
        if highres_image.max() > 1.0:
            highres_image = highres_image / 255.0
        
        # Apply RANDOM MRI-specific degradation on-the-fly (BLIND SR!)
        lowres_degraded, degradation_params = self.degradation_model.apply_random_degradation(
            highres_image
        )
        
        # Data augmentation (optional - commented out for now)
        # MRI is typically not flipped/rotated as it changes anatomy orientation
        # if self.augment:
        #     # Random horizontal flip
        #     if np.random.rand() > 0.5:
        #         highres_image = np.fliplr(highres_image)
        #         lowres_degraded = np.fliplr(lowres_degraded)
            
        #     # Random vertical flip  
        #     if np.random.rand() > 0.5:
        #         highres_image = np.flipud(highres_image)
        #         lowres_degraded = np.flipud(lowres_degraded)
        
        # Convert to tensors
        lowres_tensor = torch.from_numpy(lowres_degraded).float()   # Input (degraded)
        highres_tensor = torch.from_numpy(highres_image).float()    # Target (clean)
        
        # Add channel dimension if needed (for single-channel grayscale)
        if lowres_tensor.dim() == 2:
            lowres_tensor = lowres_tensor.unsqueeze(0)
        if highres_tensor.dim() == 2:
            highres_tensor = highres_tensor.unsqueeze(0)
        
        return lowres_tensor, highres_tensor


# Alias for compatibility 
class train_dataset(ACDCBlindSRDataset):
    """Alias to match your original train_dataset naming"""
    pass