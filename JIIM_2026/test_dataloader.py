import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

'''
CT will always be the first argument returnned by the dataloader

'''

# 1. Make so it works with only one scan at a time. Reason 1, easier final volume creation. Reason 2 easier volume metric calculation

def leaf_folder(path):
    print(type(path))
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isfile(full_path):  # Ensure it's a file
            return entry.endswith(('.npy', '.npz'))
    return False  # No files found

class test_dataset(Dataset):
    def __init__(self, paths, type, masks = True, transform=None):
        ct_paths= [os.path.join(paths,path,'ct/processed') for path in os.listdir(paths)]
        mr_paths = [os.path.join(paths,path,'mr/processed') for path in os.listdir(paths)]
        mask_paths = [os.path.join(paths,path,'mask/processed') for path in os.listdir(paths)]
        self.type = type
        self.mask_boolean = masks
        
        assert isinstance(paths, str), "Expecting one patient at a time for validation. Rest will be evaluated once metrics for this one are taken."
        
        ct_path_processed = [os.path.join(paths,'ct/processed')]
        mr_path_processed = [os.path.join(paths,'mr/processed')]
        mask_path_processed = [os.path.join(paths,'mask/processed')]
        
        ct_path_original = [os.path.join(paths,'ct/original')]
        mr_path_original = [os.path.join(paths,'mr/original')]
        mask_path_original = [os.path.join(paths,'mask/original')]

        ct_path_processed = [pat for pat in ct_path_processed if '.DS_Store' not in pat]
        mr_path_processed = [pat for pat in mr_path_processed if '.DS_Store' not in pat]
        mask_path_processed = [pat for pat in mask_path_processed if '.DS_Store' not in pat]
        ct_path_original = [pat for pat in ct_path_original if '.DS_Store' not in pat]
        mr_path_original = [pat for pat in mr_path_original if '.DS_Store' not in pat]
        mask_path_original = [pat for pat in mask_path_original if '.DS_Store' not in pat]
        

        self.ct_slices_processed, self.mr_slices_processed, self.mask_slices_processed = [], [], []
        self.ct_slices_original, self.mr_slices_original, self.mask_slices_original = [], [], []


        
        for scan_path in ct_path_processed:
            ct_temp = os.listdir(scan_path)
            for slice in ct_temp:
                self.ct_slices_processed.append(os.path.join(scan_path, slice))
        self.ct_slices_processed = sorted(self.ct_slices_processed )
                
        for scan_path in mr_path_processed:
            temp = os.listdir(scan_path)
            for slice in temp:
                self.mr_slices_processed.append(os.path.join(scan_path, slice))
        self.mr_slices_processed = sorted(self.mr_slices_processed)
                
        for scan_path in mask_path_processed:
            temp = os.listdir(scan_path)
            for slice in temp:
                self.mask_slices_processed.append(os.path.join(scan_path, slice))
        self.mask_slices_processed = sorted(self.mask_slices_processed ) 

        for scan_path in ct_path_original:
            ct_temp = os.listdir(scan_path)
            for slice in ct_temp:
                self.ct_slices_original.append(os.path.join(scan_path, slice))
        self.ct_slices_original = sorted(self.ct_slices_original)
                
        for scan_path in mr_path_original:
            ct_temp = os.listdir(scan_path)
            for slice in ct_temp:
                self.mr_slices_original.append(os.path.join(scan_path, slice))
        self.mr_slices_original = sorted(self.mr_slices_original)
        
        for scan_path in mask_path_original:
            ct_temp = os.listdir(scan_path)
            for slice in ct_temp:
                self.mask_slices_original.append(os.path.join(scan_path, slice))
        self.mask_slices_original = sorted(self.mask_slices_original)        
                

        
        assert len(self.ct_slices_processed) == len(self.mr_slices_processed) == len(self.ct_slices_original) == len(self.mr_slices_original) #== len(self.mask_slices_original)== len(self.mask_slices_processed)  , "MR and CT folder lists must be the same length"

    def __len__(self):
        # if evaluating_run_boolean:
        # return 16
        return len(self.ct_slices_processed)  # Ensuring pairing

    def __getitem__(self, idx):
        # Load MR and CT numpy arrays
        
        # Required Optimization step [Done]
        if self.type == 'MR2CT':
            mr_processed = np.load(self.mr_slices_processed[idx])['arr_0']
            ct_orignial = np.load(self.ct_slices_original[idx])['arr_0']
            
            ct_orignial = torch.tensor(ct_orignial, dtype=torch.float32)
            mr_processed = torch.tensor(mr_processed, dtype=torch.float32)
            
            
            
            if self.mask_boolean:
                mask = np.load(self.mask_slices_original[idx])['arr_0']
                mask = torch.tensor(mask, dtype=torch.float32)
                
                return ct_orignial, mr_processed, mask
            else:
                return ct_orignial, mr_processed
            
            
        elif self.type == 'CT2MR':
            ct_processed = np.load(self.ct_slices_processed[idx])['arr_0']
            mr_original = np.load(self.mr_slices_original[idx])['arr_0']
            
            ct_processed = torch.tensor(ct_processed, dtype=torch.float32)
            mr_original = torch.tensor(mr_original, dtype=torch.float32)
            
            
            if self.mask_boolean:
                mask = np.load(self.mask_slices_original[idx])['arr_0']
                mask = torch.tensor(mask, dtype=torch.float32)
                
                return ct_processed, mr_original, mask
            else:
                return ct_processed, mr_original
        else:
            raise ValueError("Only acceptable methods are MR2CT or CT2MR")

            
