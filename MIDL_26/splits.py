# splits_acdc.py

import os
from training_hyperparameters import data_directory

# Get all patient folders
all_patients = os.listdir(data_directory)
all_patients = sorted([p for p in all_patients if '.DS_Store' not in p])

# ACDC split: First 100 patients = train, Last 50 patients = validation
# Total ACDC patients = 150 (100 train, 50 validation)
train_count = 100
validation_count = 50

# Ensure we have enough patients
if len(all_patients) < train_count + validation_count:
    print(f"Warning: Only {len(all_patients)} patients found, adjusting split...")
    train_count = int(len(all_patients) * 0.67)  # Fallback to ~67% train
    validation_count = len(all_patients) - train_count

SPLITS = {1: {
    'train': [os.path.join(data_directory, all_patients[i]) for i in range(train_count)],
    'validation': [os.path.join(data_directory, all_patients[i]) for i in range(train_count, train_count + validation_count)],
}}

if __name__ == '__main__':
    print("="*60)
    print("ACDC Dataset Splits")
    print("="*60)
    print(f"Total patients: {len(all_patients)}")
    print(f"\nTrain patients: {len(SPLITS[1]['train'])} ({len(SPLITS[1]['train'])/len(all_patients)*100:.1f}%)")
    print(f"  Range: {os.path.basename(SPLITS[1]['train'][0])} to {os.path.basename(SPLITS[1]['train'][-1])}")
    
    print(f"\nValidation patients: {len(SPLITS[1]['validation'])} ({len(SPLITS[1]['validation'])/len(all_patients)*100:.1f}%)")
    print(f"  Range: {os.path.basename(SPLITS[1]['validation'][0])} to {os.path.basename(SPLITS[1]['validation'][-1])}")
    print("="*60)