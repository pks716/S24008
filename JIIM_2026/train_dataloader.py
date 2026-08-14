import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class train_dataset(Dataset):
    def __init__(self, paths, masks = True, transform=None):
        self.mask_boolean = masks
        
        ct_paths= [os.path.join(path,'ct/processed') for path in paths if '.DS_Store' not in path]
        mr_paths = [os.path.join(path,'mr/processed') for path in paths if '.DS_Store' not in path]
        mask_paths = [os.path.join(path,'mask/processed') for path in paths if '.DS_Store' not in path]
        # These now contain all directories
        # print('---?>', ct_paths)
        # input()
        
        self.ct_slices, self.mr_slices, self.mask_slices = [], [], []
        
        for scan_path in ct_paths:
            ct_temp = os.listdir(scan_path)
            for slice in ct_temp:
                self.ct_slices.append(os.path.join(scan_path, slice))
        for scan_path in mr_paths:
            ct_temp = os.listdir(scan_path)
            for slice in ct_temp:
                self.mr_slices.append(os.path.join(scan_path, slice))
        for scan_path in mask_paths:
            ct_temp = os.listdir(scan_path)
            for slice in ct_temp:
                self.mask_slices.append(os.path.join(scan_path, slice))
                
                
        # for slice in mr_paths:
        #     self.mr_slices.append(slice)
        # for slice in mask_paths:
        #     self.mask_slices.append(slice)
        # print(len(self.ct_slices))
        # print(len(self.mr_slices))
        
        assert len(self.ct_slices) == len(self.mr_slices), "MR and CT folder lists must be the same length"

    def __len__(self):
        # return 16
        return len(self.ct_slices)  # Ensuring pairing

    def __getitem__(self, idx):
        # Load MR and CT numpy arrays
        mr = np.load(self.mr_slices[idx])['arr_0']
        ct = np.load(self.ct_slices[idx])['arr_0']
        
        mr = torch.tensor(mr, dtype=torch.float32)
        ct = torch.tensor(ct, dtype=torch.float32)
        
        
        "DO AS YOU PLEASE HERE"
        
        
        if self.mask_boolean:    
            mask = np.load(self.mask_slices[idx])['arr_0']
            mask = torch.tensor(mask, dtype=torch.float32)
            return ct, mr , mask
        
        return ct ,mr