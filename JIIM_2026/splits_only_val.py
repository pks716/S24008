import os
from training_hyperparameters import data_directory

all_patients = os.listdir(data_directory)
all_patients = sorted([p for p in all_patients if '.DS_Store' not in p])

# Calculate balanced fold sizes
n_patients = len(all_patients)
base_size = n_patients // 5  # Base size for each fold
remainder = n_patients % 5   # Extra patients to distribute

# Create fold sizes: first 'remainder' folds get +1 patient
fold_sizes = [base_size + (1 if i < remainder else 0) for i in range(5)]

# Calculate start indices for each fold
fold_starts = [sum(fold_sizes[:i]) for i in range(5)]

# Create splits
def create_split(val_fold_idx):
    val_start = fold_starts[val_fold_idx]
    val_end = val_start + fold_sizes[val_fold_idx]
    
    validation_indices = list(range(val_start, val_end))
    train_indices = list(range(0, val_start)) + list(range(val_end, n_patients))
    
    return {
        'validation': validation_indices,
        'train': train_indices
    }

# Generate all splits
s1_indexes = create_split(0)
s2_indexes = create_split(1) 
s3_indexes = create_split(2)
s4_indexes = create_split(3)
s5_indexes = create_split(4)

SPLITS = {
    1: {
        'train': [os.path.join(data_directory, all_patients[i]) for i in s1_indexes['train']],
        'validation': [os.path.join(data_directory, all_patients[i]) for i in s1_indexes['validation']],
    },
    2: {
        'train': [os.path.join(data_directory, all_patients[i]) for i in s2_indexes['train']],
        'validation': [os.path.join(data_directory, all_patients[i]) for i in s2_indexes['validation']],
    },
    3: {
        'train': [os.path.join(data_directory, all_patients[i]) for i in s3_indexes['train']],
        'validation': [os.path.join(data_directory, all_patients[i]) for i in s3_indexes['validation']],
    },
    4: {
        'train': [os.path.join(data_directory, all_patients[i]) for i in s4_indexes['train']],
        'validation': [os.path.join(data_directory, all_patients[i]) for i in s4_indexes['validation']],
    },
    5: {
        'train': [os.path.join(data_directory, all_patients[i]) for i in s5_indexes['train']],
        'validation': [os.path.join(data_directory, all_patients[i]) for i in s5_indexes['validation']],
    },
}

if __name__ == '__main__':
    from train_dataloader import train_dataset
    from test_dataloader import test_dataset
    from training_hyperparameters import HP
    
    for key, splitc in SPLITS.items(): 
        print(f"Split {key}: Train={len(train_dataset(splitc['train']))}, Val={sum(len(test_dataset(p, HP['training_type'])) for p in splitc['validation'])}")