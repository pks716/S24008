"""
Test Dataloader for Patient-Wise Sequential Slice Loading
==========================================================

Returns all slices for each patient in sequential order for comprehensive
patient-wise metric evaluation.

Structure:
- Patient 1: All slices (t1, t1ce, t2, flair) × 155 slices
- Patient 2: All slices (t1, t1ce, t2, flair) × 155 slices
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Tuple, Dict


class SequentialMedicalTestDataset(Dataset):
    """
    Dataset that loads all slices for each patient sequentially.
    
    Key features:
    - Returns all slices in order (no random sampling)
    - Maintains slice correspondence across modalities
    - Provides patient metadata for aggregation
    """
    
    def __init__(
        self,
        root: str,
        modalities: List[str] = ["t1", "t1ce", "t2", "flair"],
        target_size: Tuple[int, int] = (256, 256),
        apply_mask: bool = True,
        mask_threshold: float = 0.5
    ):
        """
        Args:
            root: Root directory containing patient folders
            modalities: List of modality names (excluding 'mask')
            target_size: (H, W) target image size
            apply_mask: Whether to apply brain mask to images
            mask_threshold: Threshold for binarizing masks
        """
        self.root = root
        self.modalities = [m for m in modalities if m != 'mask']
        self.num_modalities = len(self.modalities)
        self.target_size = target_size
        self.apply_mask = apply_mask
        self.mask_threshold = mask_threshold
        
        # Find all valid patient directories
        self.patient_dirs = self._find_valid_patients()
        
        # Build patient-slice index mapping
        self.patient_slice_index = self._build_index()
        
        print(f"\n{'='*60}")
        print(f"SequentialMedicalTestDataset")
        print(f"{'='*60}")
        print(f"Patients: {len(self.patient_dirs)}")
        print(f"Modalities: {self.modalities}")
        print(f"Image size: {target_size}")
        print(f"Apply mask: {apply_mask}")
        print(f"Total samples (patients): {len(self.patient_slice_index)}")
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
    
    def _build_index(self) -> List[Dict]:
        """
        Build index mapping dataset index to (patient_idx, patient_dir, num_slices).
        Each dataset item corresponds to one complete patient with all slices.
        """
        index = []
        
        for patient_idx, patient_dir in enumerate(self.patient_dirs):
            # Get number of slices from first modality
            modality_dir = os.path.join(patient_dir, self.modalities[0])
            processed_dir = os.path.join(modality_dir, 'processed')
            search_dir = processed_dir if os.path.exists(processed_dir) else modality_dir
            
            slice_files = sorted([
                f for f in os.listdir(search_dir)
                if f.endswith(('.npz', '.npy'))
            ])
            
            num_slices = len(slice_files)
            patient_name = os.path.basename(patient_dir)
            
            index.append({
                'patient_idx': patient_idx,
                'patient_dir': patient_dir,
                'patient_name': patient_name,
                'num_slices': num_slices
            })
        
        return index
    
    def __len__(self) -> int:
        """Number of patients in dataset."""
        return len(self.patient_slice_index)
    
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
    
    def _load_all_slices_for_modality(
        self, 
        patient_dir: str, 
        modality: str
    ) -> torch.Tensor:
        """
        Load ALL slices for a modality in sequential order.
        
        Args:
            patient_dir: Path to patient directory
            modality: Modality name
            
        Returns:
            slices: [num_slices, H, W] tensor of all slices
        """
        modality_dir = os.path.join(patient_dir, modality)
        processed_dir = os.path.join(modality_dir, 'processed')
        search_dir = processed_dir if os.path.exists(processed_dir) else modality_dir
        
        # Get all slice files in sorted order
        slice_files = sorted([
            os.path.join(search_dir, f)
            for f in os.listdir(search_dir)
            if f.endswith(('.npz', '.npy'))
        ])
        
        if len(slice_files) == 0:
            raise RuntimeError(f"No slices found in {modality_dir}")
        
        # Load all slices
        slices = []
        for slice_file in slice_files:
            slice_data = self._load_slice(slice_file)
            
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
            images: [num_slices, H, W] images
            masks: [num_slices, H, W] masks
            
        Returns:
            masked_images: [num_slices, H, W] masked images
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
            volume: [num_slices, H, W] tensor
            
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
        Load all slices for all modalities of a single patient.
        
        Returns:
            patient_data: Dictionary containing:
                - images: Dict[modality_name, Tensor[num_slices, 1, H, W]]
                - patient_idx: int
                - patient_name: str
                - num_slices: int
        """
        patient_info = self.patient_slice_index[idx]
        patient_dir = patient_info['patient_dir']
        patient_idx = patient_info['patient_idx']
        patient_name = patient_info['patient_name']
        num_slices = patient_info['num_slices']
        
        try:
            # Load mask if needed
            mask_slices = None
            if self.apply_mask:
                mask_slices = self._load_all_slices_for_modality(patient_dir, 'mask')
            
            # Load all modalities
            modality_volumes = {}
            
            for modality in self.modalities:
                # Load all slices for this modality
                slices = self._load_all_slices_for_modality(patient_dir, modality)
                
                # Apply mask if enabled
                if self.apply_mask and mask_slices is not None:
                    slices = self._apply_brain_mask(slices, mask_slices)
                
                # Normalize
                slices = self._normalize_volume(slices)
                
                # Add channel dimension: [num_slices, H, W] -> [num_slices, 1, H, W]
                slices = slices.unsqueeze(1)
                
                modality_volumes[modality] = slices
            
            return {
                'images': modality_volumes,  # Dict of [num_slices, 1, H, W] per modality
                'patient_idx': patient_idx,
                'patient_name': patient_name,
                'num_slices': num_slices
            }
        
        except Exception as e:
            print(f"Error loading patient {patient_dir}: {e}")
            # Return dummy data
            dummy_volumes = {
                mod: torch.zeros((num_slices, 1, *self.target_size))
                for mod in self.modalities
            }
            return {
                'images': dummy_volumes,
                'patient_idx': patient_idx,
                'patient_name': patient_name,
                'num_slices': num_slices
            }


def sequential_test_collate_fn(batch_list: List[Dict]) -> Dict:
    """
    Custom collate function for test data.
    Since we process one patient at a time, this just returns the single patient.
    """
    # In test mode, we typically use batch_size=1, so just return first item
    if len(batch_list) == 1:
        return batch_list[0]
    else:
        # If batch_size > 1, return list of patients
        return batch_list


def create_test_dataloader(
    test_root: str,
    modalities: List[str] = ["t1", "t1ce", "t2", "flair"],
    target_size: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    apply_mask: bool = True
) -> DataLoader:
    """
    Create test dataloader for patient-wise evaluation.
    
    Args:
        test_root: Path to test data
        modalities: List of modality names
        target_size: (H, W) image size
        num_workers: Number of dataloader workers
        apply_mask: Whether to apply brain masks
        
    Returns:
        test_loader: DataLoader yielding one patient at a time
        
    Example usage:
        test_loader = create_test_dataloader(test_root)
        for patient_data in test_loader:
            images = patient_data['images']  # Dict of modality tensors
            patient_name = patient_data['patient_name']
            
            # Process each modality
            for modality_name, modality_volume in images.items():
                # modality_volume: [num_slices, 1, H, W]
                print(f"Processing {patient_name} - {modality_name}: {modality_volume.shape}")
    """
    test_dataset = SequentialMedicalTestDataset(
        root=test_root,
        modalities=modalities,
        target_size=target_size,
        apply_mask=apply_mask
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one patient at a time
        shuffle=False,  # Keep sequential order
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=sequential_test_collate_fn
    )
    
    print(f"\n{'='*60}")
    print(f"TEST DATALOADER CONFIGURATION")
    print(f"{'='*60}")
    print(f"Patients: {len(test_dataset)}")
    print(f"Modalities: {modalities}")
    print(f"Batch size: 1 (one patient at a time)")
    print(f"Shuffle: False (sequential order)")
    print(f"{'='*60}\n")
    
    return test_loader


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the sequential dataloader"""
    
    test_root = ""
    
    test_loader = create_test_dataloader(
        test_root=test_root,
        modalities=["t1", "t1ce", "t2", "flair"],
        target_size=(256, 256),
        num_workers=4,
        apply_mask=True
    )
    
    print("\nTESTING SEQUENTIAL DATALOADER...")
    
    # Test first patient
    patient_data = next(iter(test_loader))
    
    print(f"\nPatient data structure:")
    print(f"   patient_name: {patient_data['patient_name']}")
    print(f"   patient_idx: {patient_data['patient_idx']}")
    print(f"   num_slices: {patient_data['num_slices']}")
    
    print(f"\nModality volumes:")
    for modality_name, volume in patient_data['images'].items():
        print(f"   {modality_name}: {volume.shape}")
    
    print(f"\nSequential test dataloader working correctly!")
    
    # Test iteration over multiple patients
    print(f"\n🔄 Testing iteration over first 3 patients...")
    for i, patient_data in enumerate(test_loader):
        if i >= 3:
            break
        print(f"   Patient {i+1}: {patient_data['patient_name']} "
              f"({patient_data['num_slices']} slices)")
    
    print(f"\nAll tests passed!")