# test_loader_acdc_sr.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

class ACDCTestDataset(Dataset):
    """
    ACDC Test Dataset with Multiple MRI-specific Degradation Types
    Supports testing model robustness across diverse MRI artifacts
    """
    
    def __init__(self, patient_path, training_type='super_resolution', 
                 scale_factor=2, degradation_type='bicubic', 
                 degradation_params=None):
        """
        Args:
            patient_path: Path to single patient folder
            training_type: Task type
            scale_factor: Downsampling factor (2× default)
            degradation_type: Type of MRI degradation to apply
                - 'bicubic': Standard bicubic downsampling
                - 'bilinear': Bilinear downsampling
                - 'area': Area-based downsampling (antialiased)
                - 'gaussian_blur': Gaussian blur + bicubic (partial volume)
                - 'motion_blur': Motion blur + bicubic (cardiac motion)
                - 'anisotropic': Anisotropic blur (slice thickness)
                - 'rician_noise': Rician noise + bicubic (MRI noise)
                - 'combined': Motion blur + Rician noise + bicubic
                - 'none': No degradation (clean images)
            degradation_params: Dict with degradation-specific parameters
                - blur_sigma: float (for gaussian_blur, default: 2.0)
                - noise_level: float (for rician_noise, default: 0.05)
                - kernel_size: int (for motion_blur, default: 9)
        """
        self.patient_path = patient_path
        self.scale_factor = scale_factor
        self.degradation_type = degradation_type
        self.degradation_params = degradation_params or {}
        
        # Collect all .npy files
        self.image_paths = []
        
        if os.path.isdir(patient_path):
            npy_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.npy')])
            for npy_file in npy_files:
                full_path = os.path.join(patient_path, npy_file)
                self.image_paths.append(full_path)
        
        patient_name = os.path.basename(patient_path)
        print(f"Test Dataset - {patient_name}: {len(self.image_paths)} MRI slices, "
              f"Degradation: {degradation_type}")
    
    def apply_gaussian_blur(self, image, sigma=2.0):
        """Apply Gaussian blur (partial volume effects)"""
        return gaussian_filter(image, sigma=sigma)
    
    def apply_anisotropic_blur(self, image):
        """Apply anisotropic blur (slice thickness artifacts)"""
        sigma_x = self.degradation_params.get('sigma_x', 0.8)
        sigma_z = self.degradation_params.get('sigma_z', 3.0)
        
        if np.random.rand() > 0.5:
            return gaussian_filter(image, sigma=[sigma_x, sigma_z])
        else:
            return gaussian_filter(image, sigma=[sigma_z, sigma_x])
    
    def apply_motion_blur(self, image, kernel_size=9):
        """Apply motion blur (cardiac/respiratory motion)"""
        from scipy.signal import convolve2d
        from scipy.ndimage import rotate as scipy_rotate
        
        angle = np.random.uniform(0, 180)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        kernel = scipy_rotate(kernel, angle, reshape=False)
        return convolve2d(image, kernel, mode='same', boundary='symm')
    
    def add_rician_noise(self, image, noise_level=0.05):
        """Add Rician noise (PRIMARY MRI noise characteristic)"""
        noise_real = np.random.normal(0, noise_level, image.shape)
        noise_imag = np.random.normal(0, noise_level, image.shape)
        
        # Magnitude reconstruction (Rician distribution)
        noisy_real = image + noise_real
        noisy_imag = noise_imag
        magnitude = np.sqrt(noisy_real**2 + noisy_imag**2)
        
        return np.clip(magnitude, 0, 1)
    
    def add_gaussian_noise(self, image, noise_level=0.03):
        """Add Gaussian noise (less realistic for MRI)"""
        noise = np.random.normal(0, noise_level, image.shape)
        return np.clip(image + noise, 0, 1)
    
    def downsample_upsample(self, image, method='bicubic'):
        """Downsample then upsample back to original size"""
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        
        # Downsample
        if method == 'bicubic':
            lowres = F.interpolate(
                image_tensor,
                scale_factor=1.0/self.scale_factor,
                mode='bicubic',
                align_corners=False
            )
        elif method == 'bilinear':
            lowres = F.interpolate(
                image_tensor,
                scale_factor=1.0/self.scale_factor,
                mode='bilinear',
                align_corners=False
            )
        elif method == 'area':
            lowres = F.interpolate(
                image_tensor,
                scale_factor=1.0/self.scale_factor,
                mode='area'
            )
        elif method == 'nearest':
            lowres = F.interpolate(
                image_tensor,
                scale_factor=1.0/self.scale_factor,
                mode='nearest'
            )
        else:
            lowres = image_tensor
        
        # Upsample back to original size
        lowres_upsampled = F.interpolate(
            lowres,
            size=image_tensor.shape[2:],
            mode='bicubic',
            align_corners=False
        )
        
        return lowres_upsampled.squeeze().numpy()
    
    def apply_degradation(self, highres_image):
        """Apply specified MRI-specific degradation type"""
        if self.degradation_type == 'none':
            # No degradation - return original
            return highres_image
        
        elif self.degradation_type == 'bicubic':
            return self.downsample_upsample(highres_image, method='bicubic')
        
        elif self.degradation_type == 'bilinear':
            return self.downsample_upsample(highres_image, method='bilinear')
        
        elif self.degradation_type == 'area':
            return self.downsample_upsample(highres_image, method='area')
        
        elif self.degradation_type == 'gaussian_blur':
            # Partial volume effects
            sigma = self.degradation_params.get('blur_sigma', 2.0)
            blurred = self.apply_gaussian_blur(highres_image, sigma=sigma)
            return self.downsample_upsample(blurred, method='bicubic')
        
        elif self.degradation_type == 'motion_blur':
            # Cardiac/respiratory motion
            kernel_size = self.degradation_params.get('kernel_size', 9)
            blurred = self.apply_motion_blur(highres_image, kernel_size=kernel_size)
            return self.downsample_upsample(blurred, method='bicubic')
        
        elif self.degradation_type == 'anisotropic':
            # Slice thickness artifacts
            blurred = self.apply_anisotropic_blur(highres_image)
            return self.downsample_upsample(blurred, method='bicubic')
        
        elif self.degradation_type == 'rician_noise':
            # MRI Rician noise
            noise_level = self.degradation_params.get('noise_level', 0.05)
            noisy = self.add_rician_noise(highres_image, noise_level=noise_level)
            return self.downsample_upsample(noisy, method='bicubic')
        
        elif self.degradation_type == 'combined':
            # Realistic MRI: motion blur + Rician noise
            kernel_size = self.degradation_params.get('kernel_size', 7)
            noise_level = self.degradation_params.get('noise_level', 0.05)
            
            # Apply motion blur
            blurred = self.apply_motion_blur(highres_image, kernel_size=kernel_size)
            # Add Rician noise
            noisy = self.add_rician_noise(blurred, noise_level=noise_level)
            # Downsample
            return self.downsample_upsample(noisy, method='bicubic')
        
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load HIGH-RES MRI image (target)
        highres_image = np.load(self.image_paths[idx])
        
        # Normalize
        highres_image = highres_image.astype(np.float32)
        if highres_image.max() > 1.0:
            highres_image = highres_image / 255.0
        
        # Apply degradation (input)
        lowres_image = self.apply_degradation(highres_image)
        
        # Convert to tensors
        lowres_tensor = torch.from_numpy(lowres_image).float()
        highres_tensor = torch.from_numpy(highres_image).float()
        
        # Add channel dimension if needed
        if lowres_tensor.dim() == 2:
            lowres_tensor = lowres_tensor.unsqueeze(0)
        if highres_tensor.dim() == 2:
            highres_tensor = highres_tensor.unsqueeze(0)
        
        return lowres_tensor, highres_tensor


# Alias
class test_dataset(ACDCTestDataset):
    pass