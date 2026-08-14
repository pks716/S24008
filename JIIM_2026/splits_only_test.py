import os
from training_hyperparameters import data_directory
all_patients = os.listdir(data_directory)
all_patients = sorted([p for p in all_patients if '.DS_Store' not in p])
# Use all data for testing only
SPLITS = {
    1: {
        'train': [],
        'validation': [os.path.join(data_directory, patient) for patient in all_patients],
    }
}
if __name__ == '__main__':
    print(f"Total patients: {len(all_patients)}")
    print(f"All {len(all_patients)} patients will be used for testing only")
    print(f"Train: 0 patients")
    print(f"Validation: {len(all_patients)} patients")