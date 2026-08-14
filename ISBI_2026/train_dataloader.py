"""
Contrastive Learning Dataloader for Multi-Modal Medical Images
================================================================

Optimized batch structure for contrastive learning with two objectives:
1. Anatomy-invariant learning: same patient, different modalities → POSITIVE
2. Contrast-invariant learning: different patients, same modality → POSITIVE

Batch Structure Strategy:
-------------------------
Each batch contains:
- Multiple patients (P patients)
- Multiple modalities per patient (M modalities)
- Multiple slices per modality (S slices)

This creates a rich set of positive pairs:
- Anatomy positives: P × M × (M-1) × S pairs (same patient, different modalities)
- Contrast positives: M × P × (P-1) × S pairs (same modality, different patients)

Example with batch_size=16, num_patients=4, num_modalities=4:
- 4 patients × 4 modalities = 16 samples per batch
- Each patient contributes 4×3=12 anatomy positive pairs
- Each modality contributes 4×3=12 contrast positive pairs
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Tuple, Dict


class ContrastiveMedicalDataset(Dataset):
    """
    Dataset that structures data for multi-space contrastive learning.
    
    - Ensures same slice indices across modalities for a patient
    - Returns structured batches with patient_id and modality_id metadata
    - Applies brain masks to focus on relevant anatomy
    """
    
    def __init__(
        self,
        root: str,
        modalities: List[str] = ["t1", "t1ce", "t2", "flair"],
        num_slices: int = 1,
        target_size: Tuple[int, int] = (256, 256),
        apply_mask: bool = True,
        mask_threshold: float = 0.5,
        split: str = "train"
    ):
        """
        Args:
            root: Root directory containing patient folders
            modalities: List of modality names (excluding 'mask')
            num_slices: Number of slices to sample per modality per patient
            target_size: (H, W) target image size
            apply_mask: Whether to apply brain mask to images
            mask_threshold: Threshold for binarizing masks
            split: 'train' or 'val'
        """
        self.root = root
        self.modalities = [m for m in modalities if m != 'mask']
        self.num_modalities = len(self.modalities)
        self.num_slices = num_slices
        self.target_size = target_size
        self.apply_mask = apply_mask
        self.mask_threshold = mask_threshold
        self.split = split
        
        # Find all valid patient directories
        self.patient_dirs = self._find_valid_patients()
        
        # Create mappings for labels
        self.modality_to_idx = {m: i for i, m in enumerate(self.modalities)}
        
        print(f"\n{'='*60}")
        print(f"ContrastiveMedicalDataset - {split.upper()} split")
        print(f"{'='*60}")
        print(f"Patients: {len(self.patient_dirs)}")
        print(f"Modalities: {self.modalities}")
        print(f"Slices per modality: {num_slices}")
        print(f"Image size: {target_size}")
        print(f"Apply mask: {apply_mask}")
        print(f"{'='*60}\n")
    
    def _find_valid_patients(self) -> List[str]:
        """Find all patient directories with complete data."""
        patient_dirs = sorted([
            os.path.join(self.root, p)
            for p in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, p))
        ])
        
        valid_patients = []
        for patient_dir in patient_dirs:
            # Check if patient has all required modalities
            has_all_modalities = all(
                os.path.exists(os.path.join(patient_dir, m))
                for m in self.modalities + ['mask']
            )
            
            if has_all_modalities:
                # Verify each modality has slices
                all_have_slices = True
                for m in self.modalities + ['mask']:
                    mod_dir = os.path.join(patient_dir, m)
                    processed_dir = os.path.join(mod_dir, 'processed')
                    search_dir = processed_dir if os.path.exists(processed_dir) else mod_dir
                    
                    slice_files = [f for f in os.listdir(search_dir) 
                                   if f.endswith(('.npz', '.npy'))]
                    if len(slice_files) == 0:
                        all_have_slices = False
                        break
                
                if all_have_slices:
                    valid_patients.append(patient_dir)
                else:
                    print(f"Skipping {os.path.basename(patient_dir)}: missing slices")
            else:
                print(f"Skipping {os.path.basename(patient_dir)}: incomplete modalities")
        
        if len(valid_patients) == 0:
            raise RuntimeError(f"No valid patients found in {self.root}")
        
        return valid_patients
    
    def __len__(self) -> int:
        """Each patient is one sample (we'll load all modalities)."""
        return len(self.patient_dirs)
    
    def _load_slice(self, file_path: str) -> np.ndarray:
        """Load a single slice from .npz or .npy file."""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Extract slice data
            if 'data' in data:
                slice_data = data['data']
            elif 'arr_0' in data:
                slice_data = data['arr_0']
            else:
                slice_data = data[data.files[0]]
            
            # Squeeze to 2D
            if slice_data.ndim > 2:
                slice_data = slice_data.squeeze()
            
            return slice_data.astype(np.float32)
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise
    
    def _resize_slice(self, slice_data: np.ndarray) -> np.ndarray:
        """Resize slice to target size."""
        if slice_data.shape == self.target_size:
            return slice_data
        
        slice_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            slice_tensor, 
            size=self.target_size, 
            mode='bilinear', 
            align_corners=False
        )
        return resized.squeeze().numpy()
    
    def _load_modality(
        self, 
        patient_dir: str, 
        modality: str, 
        slice_indices: List[int]
    ) -> torch.Tensor:
        """
        Load specific slices for a modality.
        
        Args:
            patient_dir: Path to patient directory
            modality: Modality name
            slice_indices: List of slice indices to load
            
        Returns:
            slices: [S, H, W] tensor of slices
        """
        modality_dir = os.path.join(patient_dir, modality)
        processed_dir = os.path.join(modality_dir, 'processed')
        search_dir = processed_dir if os.path.exists(processed_dir) else modality_dir
        
        # Get all slice files
        slice_files = sorted([
            os.path.join(search_dir, f)
            for f in os.listdir(search_dir)
            if f.endswith(('.npz', '.npy'))
        ])
        
        if len(slice_files) == 0:
            raise RuntimeError(f"No slices found in {modality_dir}")
        
        # Load selected slices
        slices = []
        for idx in slice_indices:
            if idx >= len(slice_files):
                idx = idx % len(slice_files)  # Wrap around if needed
            
            slice_data = self._load_slice(slice_files[idx])
            
            # Resize if needed
            if slice_data.shape != self.target_size:
                slice_data = self._resize_slice(slice_data)
            
            slices.append(slice_data)
        
        return torch.tensor(np.stack(slices), dtype=torch.float32)
    
    def _apply_brain_mask(
        self, 
        images: torch.Tensor, 
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply binary brain mask to images.
        
        Args:
            images: [S, H, W] images
            masks: [S, H, W] masks
            
        Returns:
            masked_images: [S, H, W] masked images
        """
        # Ensure mask has same shape as images
        if masks.shape != images.shape:
            masks = F.interpolate(
                masks.unsqueeze(1), 
                size=images.shape[-2:], 
                mode='nearest'
            ).squeeze(1)
        
        # Binarize mask
        binary_mask = (masks > self.mask_threshold).float()
        
        return images * binary_mask
    
    def _normalize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Normalize volume to [0, 1] range.
        
        Args:
            volume: [S, H, W] or [M, S, H, W] tensor
            
        Returns:
            normalized: Same shape, normalized to [0, 1]
        """
        vol_min = volume.min()
        vol_max = volume.max()
        
        if (vol_max - vol_min) > 1e-8:
            return (volume - vol_min) / (vol_max - vol_min)
        else:
            return torch.zeros_like(volume)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load all modalities for a single patient.
        
        Returns:
            batch: Dictionary containing:
                - images: [M*S, 1, H, W] all images for this patient
                - patient_id: [M*S] patient ID repeated
                - modality_id: [M*S] modality indices
                
        Where M = num_modalities, S = num_slices
        """
        patient_dir = self.patient_dirs[idx]
        patient_id = idx
        
        try:
            # Sample slice indices (same across all modalities for this patient)
            # This ensures anatomical correspondence
            modality_dir = os.path.join(patient_dir, self.modalities[0])
            processed_dir = os.path.join(modality_dir, 'processed')
            search_dir = processed_dir if os.path.exists(processed_dir) else modality_dir
            
            all_slice_files = sorted([
                f for f in os.listdir(search_dir)
                if f.endswith(('.npz', '.npy'))
            ])
            num_available = len(all_slice_files)
            
            # Sample random slice indices
            slice_indices = random.sample(
                range(num_available), 
                min(self.num_slices, num_available)
            )
            
            # Load mask if needed
            mask_slices = None
            if self.apply_mask:
                mask_slices = self._load_modality(patient_dir, 'mask', slice_indices)
            
            # Load all modalities
            all_images = []
            all_modality_ids = []
            
            for modality_idx, modality in enumerate(self.modalities):
                # Load slices for this modality
                slices = self._load_modality(patient_dir, modality, slice_indices)
                
                # Apply mask if enabled
                if self.apply_mask and mask_slices is not None:
                    slices = self._apply_brain_mask(slices, mask_slices)
                
                # Normalize
                slices = self._normalize_volume(slices)
                
                # Add channel dimension: [S, H, W] -> [S, 1, H, W]
                slices = slices.unsqueeze(1)
                
                all_images.append(slices)
                all_modality_ids.extend([modality_idx] * len(slices))
            
            # Concatenate all modalities: [M*S, 1, H, W]
            images = torch.cat(all_images, dim=0)
            
            # Create labels
            patient_ids = torch.full((len(images),), patient_id, dtype=torch.long)
            modality_ids = torch.tensor(all_modality_ids, dtype=torch.long)
            
            return {
                'images': images,               # [M*S, 1, H, W]
                'patient_id': patient_ids,      # [M*S]
                'modality_id': modality_ids,    # [M*S]
            }
        
        except Exception as e:
            print(f"Error loading patient {patient_dir}: {e}")
            # Return dummy batch
            num_samples = self.num_modalities * self.num_slices
            return {
                'images': torch.zeros((num_samples, 1, *self.target_size)),
                'patient_id': torch.full((num_samples,), idx, dtype=torch.long),
                'modality_id': torch.zeros(num_samples, dtype=torch.long),
            }


def contrastive_collate_fn(batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that preserves structure for contrastive learning.
    
    Input: List of P patient dictionaries (from __getitem__)
    Output: Single batch dictionary with all data concatenated
    
    The key is to maintain patient_id and modality_id labels so we can
    identify positive pairs during contrastive loss computation.
    """
    # Each batch_list item has shape [M*S, ...]
    # We concatenate to get [P*M*S, ...]
    
    all_images = []
    all_patient_ids = []
    all_modality_ids = []
    
    for batch_idx, sample in enumerate(batch_list):
        all_images.append(sample['images'])
        all_patient_ids.append(sample['patient_id'])
        all_modality_ids.append(sample['modality_id'])
    
    return {
        'images': torch.cat(all_images, dim=0),           # [P*M*S, 1, H, W]
        'patient_id': torch.cat(all_patient_ids, dim=0),  # [P*M*S]
        'modality_id': torch.cat(all_modality_ids, dim=0), # [P*M*S]
    }


# =============================================================================
# DATALOADER FACTORY FUNCTIONS
# =============================================================================

def create_contrastive_dataloaders(
    train_root: str,
    val_root: str,
    modalities: List[str] = ["t1", "t1ce", "t2", "flair"],
    num_patients_per_batch: int = 4,
    num_slices_per_modality: int = 1,
    target_size: Tuple[int, int] = (256, 256),
    num_workers: int = 8,
    apply_mask: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders optimized for contrastive learning.
    
    Args:
        train_root: Path to training data
        val_root: Path to validation data
        modalities: List of modality names
        num_patients_per_batch: Number of patients per batch (batch_size = this × num_modalities)
        num_slices_per_modality: Number of slices to sample per modality
        target_size: (H, W) image size
        num_workers: Number of dataloader workers
        apply_mask: Whether to apply brain masks
        
    Returns:
        train_loader, val_loader
        
    Example:
        With num_patients_per_batch=4, modalities=4, num_slices=1:
        - Each batch has 4 patients × 4 modalities × 1 slice = 16 images
        - This gives:
          * 4 patients × 4×3 = 48 anatomy positive pairs (same patient, diff modality)
          * 4 modalities × 4×3 = 48 contrast positive pairs (same modality, diff patient)
    """
    train_dataset = ContrastiveMedicalDataset(
        root=train_root,
        modalities=modalities,
        num_slices=num_slices_per_modality,
        target_size=target_size,
        apply_mask=apply_mask,
        split="train"
    )
    
    val_dataset = ContrastiveMedicalDataset(
        root=val_root,
        modalities=modalities,
        num_slices=num_slices_per_modality,
        target_size=target_size,
        apply_mask=apply_mask,
        split="val"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=num_patients_per_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for contrastive learning
        collate_fn=contrastive_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=num_patients_per_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=contrastive_collate_fn
    )
    
    # Print batch statistics
    print(f"\n{'='*60}")
    print(f"DATALOADER CONFIGURATION")
    print(f"{'='*60}")
    print(f"Patients per batch: {num_patients_per_batch}")
    print(f"Modalities: {len(modalities)}")
    print(f"Slices per modality: {num_slices_per_modality}")
    print(f"Effective batch size: {num_patients_per_batch * len(modalities) * num_slices_per_modality} images")
    print(f"\nPOSITIVE PAIRS PER BATCH:")
    anatomy_positives = num_patients_per_batch * len(modalities) * (len(modalities) - 1) * num_slices_per_modality
    contrast_positives = len(modalities) * num_patients_per_batch * (num_patients_per_batch - 1) * num_slices_per_modality
    print(f"   Anatomy (same patient, diff modality): ~{anatomy_positives}")
    print(f"   Contrast (same modality, diff patient): ~{contrast_positives}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the dataloader structure"""
    
    train_root = ""
    val_root = ""
    
    train_loader, val_loader = create_contrastive_dataloaders(
        train_root=train_root,
        val_root=val_root,
        modalities=["t1", "t1ce", "t2", "flair"],
        num_patients_per_batch=4,
        num_slices_per_modality=1,
        target_size=(256, 256),
        num_workers=4,
        apply_mask=True
    )
    
    print("\nTESTING DATALOADER...")
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"   images: {batch['images'].shape}")
    print(f"   patient_id: {batch['patient_id'].shape}")
    print(f"   modality_id: {batch['modality_id'].shape}")
    
    print(f"\nLabel distribution:")
    print(f"   Unique patients: {torch.unique(batch['patient_id']).tolist()}")
    print(f"   Unique modalities: {torch.unique(batch['modality_id']).tolist()}")
    
    print(f"\nContrastive pair analysis:")
    N = len(batch['images'])
    patient_ids = batch['patient_id']
    modality_ids = batch['modality_id']
    
    # Count anatomy positives (same patient, different modality)
    same_patient = (patient_ids.unsqueeze(0) == patient_ids.unsqueeze(1))
    diff_modality = (modality_ids.unsqueeze(0) != modality_ids.unsqueeze(1))
    anatomy_positives = (same_patient & diff_modality).sum().item() // 2  # Divide by 2 for symmetric pairs
    
    # Count contrast positives (same modality, different patient)
    same_modality = (modality_ids.unsqueeze(0) == modality_ids.unsqueeze(1))
    diff_patient = (patient_ids.unsqueeze(0) != patient_ids.unsqueeze(1))
    contrast_positives = (same_modality & diff_patient).sum().item() // 2
    
    print(f"   Anatomy positives: {anatomy_positives}")
    print(f"   Contrast positives: {contrast_positives}")
    print(f"   Total negatives: {N * (N - 1) // 2 - anatomy_positives - contrast_positives}")
    
    print(f"\nDataloader test passed!")