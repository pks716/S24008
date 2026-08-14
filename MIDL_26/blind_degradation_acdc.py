# blind_degradation_acdc.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.ndimage import rotate as scipy_rotate

class BlindDegradationModelMRI:
    """
    MRI-specific degradation model for blind super-resolution
    Simulates realistic cardiac MRI artifacts with STRONG degradations
    """
    
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor
        
        # MRI-specific degradation types with WEIGHTS
        self.degradation_types = [
            'bilinear',                          # 15% - Basic interpolation
            'gaussian_blur_downsample',          # 25% - Out-of-plane blur, partial volume
            'motion_blur_downsample',            # 20% - Cardiac/respiratory motion
            'anisotropic_downsample',            # 20% - Slice thickness artifacts
            'rician_noise_downsample',           # 15% - MRI Rician noise
            'combined_motion_rician',            # 5%  - Realistic combo
        ]
        
        # Corresponding weights (must sum to 1.0)
        self.degradation_weights = [0.15, 0.25, 0.20, 0.20, 0.15, 0.05]
        
        assert abs(sum(self.degradation_weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    def apply_blur(self, image, blur_type='gaussian', blur_sigma=None):
        """Apply various blur types - STRONGER for MRI"""
        if blur_sigma is None:
            # INCREASED blur strength for strong representations
            blur_sigma = np.random.uniform(1.0, 3.5)  # Much stronger than ultrasound
        
        if blur_type == 'gaussian':
            # Standard Gaussian blur (partial volume effects)
            return gaussian_filter(image, sigma=blur_sigma)
        
        elif blur_type == 'anisotropic':
            # Anisotropic blur - simulates slice thickness artifacts
            # Z-axis (slice) has much lower resolution than in-plane
            sigma_x = np.random.uniform(0.5, 1.5)      # In-plane (fine)
            sigma_y = np.random.uniform(0.5, 1.5)      # In-plane (fine)
            sigma_z = np.random.uniform(2.0, 4.0)      # Through-plane (coarse)
            
            # Apply stronger blur in one direction to simulate anisotropy
            if np.random.rand() > 0.5:
                return gaussian_filter(image, sigma=[sigma_x, sigma_z])
            else:
                return gaussian_filter(image, sigma=[sigma_z, sigma_y])
        
        elif blur_type == 'motion':
            # Motion blur - cardiac/respiratory motion during acquisition
            kernel_size = int(np.random.choice([5, 7, 9, 11, 13]))  # LARGER kernels
            angle = np.random.uniform(0, 180)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1.0 / kernel_size
            kernel = scipy_rotate(kernel, angle, reshape=False)
            return convolve2d(image, kernel, mode='same', boundary='symm')
        
        return image
    
    def add_noise(self, image, noise_type='rician', noise_level=None):
        """Add MRI-specific noise types - STRONGER"""
        if noise_level is None:
            # INCREASED noise levels for strong degradations
            noise_level = np.random.uniform(0.03, 0.10)  # Much stronger
        
        if noise_type == 'gaussian':
            # Simple Gaussian noise (less realistic for MRI)
            noise = np.random.normal(0, noise_level, image.shape)
            return np.clip(image + noise, 0, 1)
        
        elif noise_type == 'rician':
            # Rician noise - PRIMARY MRI noise characteristic
            # MRI magnitude images have Rician distribution
            # Simulate by adding Gaussian noise to real and imaginary components
            noise_real = np.random.normal(0, noise_level, image.shape)
            noise_imag = np.random.normal(0, noise_level, image.shape)
            
            # Magnitude reconstruction (Rician noise)
            noisy_real = image + noise_real
            noisy_imag = noise_imag
            magnitude = np.sqrt(noisy_real**2 + noisy_imag**2)
            
            return np.clip(magnitude, 0, 1)
        
        return image
    
    def apply_k_space_undersampling(self, image):
        """
        Simulate k-space undersampling artifacts (optional advanced degradation)
        This creates aliasing artifacts typical in accelerated MRI
        """
        # Convert to frequency domain
        k_space = np.fft.fft2(image)
        k_space_shifted = np.fft.fftshift(k_space)
        
        # Random undersampling pattern (keep center, undersample periphery)
        h, w = k_space_shifted.shape
        mask = np.ones((h, w))
        
        # Keep center 30% fully sampled
        center_fraction = 0.3
        center_h = int(h * center_fraction)
        center_w = int(w * center_fraction)
        
        # Undersample periphery (keep only 20-40% of k-space lines)
        acceleration = np.random.uniform(0.2, 0.4)
        
        for i in range(h):
            if i < (h - center_h) // 2 or i > (h + center_h) // 2:
                if np.random.rand() > acceleration:
                    mask[i, :] = 0
        
        # Apply mask
        k_space_undersampled = k_space_shifted * mask
        
        # Transform back
        k_space_unshifted = np.fft.ifftshift(k_space_undersampled)
        image_undersampled = np.fft.ifft2(k_space_unshifted)
        image_undersampled = np.abs(image_undersampled)
        
        return np.clip(image_undersampled, 0, 1)
    
    def downsample(self, image, method='bicubic'):
        """Downsample with various methods"""
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        else:
            image_tensor = image
        
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
        
        # Upsample back to original size (this is the "degraded" input)
        lowres_upsampled = F.interpolate(
            lowres,
            size=image_tensor.shape[2:],
            mode='bicubic',
            align_corners=False
        )
        
        # Convert back to numpy
        if isinstance(image, np.ndarray):
            return lowres_upsampled.squeeze().numpy()
        return lowres_upsampled
    
    def apply_random_degradation(self, highres_image):
        """
        Apply random MRI-specific degradation pipeline with STRONG degradations
        
        Returns:
            lowres_image: Degraded low-resolution image (upsampled to original size)
            degradation_params: Dict with degradation parameters
        """
        # Choose random degradation type with WEIGHTS
        deg_type = np.random.choice(
            self.degradation_types, 
            p=self.degradation_weights
        )
        
        degradation_params = {'type': deg_type}
        
        # Apply degradation pipeline
        degraded = highres_image.copy()
        
        if deg_type == 'bilinear':
            # Simple bilinear downsampling (15%)
            degraded = self.downsample(degraded, method='bilinear')
        
        elif deg_type == 'gaussian_blur_downsample':
            # Strong Gaussian blur + downsample (25%)
            # Simulates partial volume effects, out-of-plane blur
            blur_sigma = np.random.uniform(1.5, 3.5)  # STRONG blur
            degradation_params['blur_sigma'] = blur_sigma
            degraded = self.apply_blur(degraded, 'gaussian', blur_sigma)
            degraded = self.downsample(degraded, method='bicubic')
        
        elif deg_type == 'motion_blur_downsample':
            # Motion blur + downsample (20%)
            # Simulates cardiac/respiratory motion
            kernel_size = int(np.random.choice([5, 7, 9, 11, 13]))  # LARGE kernels
            angle = np.random.uniform(0, 180)
            degradation_params['motion_kernel_size'] = kernel_size
            degradation_params['motion_angle'] = angle
            degraded = self.apply_blur(degraded, 'motion')
            degraded = self.downsample(degraded, method='bicubic')
        
        elif deg_type == 'anisotropic_downsample':
            # Strong anisotropic blur + downsample (20%)
            # Simulates slice thickness artifacts (common in cardiac MRI)
            sigma_x = np.random.uniform(0.5, 1.5)
            sigma_z = np.random.uniform(2.0, 4.0)  # MUCH stronger in one direction
            degradation_params['sigma_x'] = sigma_x
            degradation_params['sigma_z'] = sigma_z
            degraded = self.apply_blur(degraded, 'anisotropic')
            degraded = self.downsample(degraded, method='bicubic')
        
        elif deg_type == 'rician_noise_downsample':
            # Rician noise + downsample (15%)
            # PRIMARY MRI noise characteristic
            noise_level = np.random.uniform(0.04, 0.10)  # STRONG noise
            degradation_params['noise_level'] = noise_level
            degradation_params['noise_type'] = 'rician'
            degraded = self.add_noise(degraded, 'rician', noise_level)
            degraded = self.downsample(degraded, method='bicubic')
        
        elif deg_type == 'combined_motion_rician':
            # Realistic combo: motion blur + Rician noise + downsample (5%)
            # This simulates real cardiac MRI artifacts
            blur_sigma = np.random.uniform(1.0, 2.5)
            noise_level = np.random.uniform(0.04, 0.08)
            degradation_params['blur_sigma'] = blur_sigma
            degradation_params['noise_level'] = noise_level
            degradation_params['noise_type'] = 'rician'
            
            # Apply motion blur first
            degraded = self.apply_blur(degraded, 'motion')
            # Then add Rician noise
            degraded = self.add_noise(degraded, 'rician', noise_level)
            # Finally downsample
            degraded = self.downsample(degraded, method='bicubic')
        
        return degraded, degradation_params