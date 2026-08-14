# visualize_npy.py

import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_npy(file_path, cmap='gray', figsize=(8, 8)):
    """
    Visualize a single .npy file
    
    Args:
        file_path: Path to .npy file
        cmap: Colormap ('gray' for grayscale, 'viridis', 'hot', etc.)
        figsize: Figure size (width, height)
    """
    # Load the array
    data = np.load(file_path)
    
    # Get file info
    filename = os.path.basename(file_path)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Display image
    if data.ndim == 2:
        # 2D image
        plt.imshow(data, cmap=cmap)
        plt.colorbar(label='Intensity')
        plt.title(f'{filename}\nShape: {data.shape}\nRange: [{data.min():.3f}, {data.max():.3f}]')
    elif data.ndim == 3:
        # 3D image - show middle slice
        slice_idx = data.shape[0] // 2
        plt.imshow(data[slice_idx], cmap=cmap)
        plt.colorbar(label='Intensity')
        plt.title(f'{filename} (slice {slice_idx}/{data.shape[0]})\nShape: {data.shape}\nRange: [{data.min():.3f}, {data.max():.3f}]')
    else:
        print(f"Unsupported dimensions: {data.ndim}D")
        return
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"File: {filename}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Range: [{data.min():.6f}, {data.max():.6f}]")
    print(f"Mean: {data.mean():.6f}")
    print(f"Std: {data.std():.6f}")
    print(f"{'='*50}\n")


def visualize_multiple_npy(file_paths, cmap='gray', ncols=4):
    """
    Visualize multiple .npy files in a grid
    
    Args:
        file_paths: List of paths to .npy files
        cmap: Colormap
        ncols: Number of columns in grid
    """
    n_images = len(file_paths)
    nrows = (n_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, file_path in enumerate(file_paths):
        data = np.load(file_path)
        filename = os.path.basename(file_path)
        
        # Handle 2D or 3D
        if data.ndim == 3:
            data = data[data.shape[0] // 2]  # Middle slice
        
        axes[idx].imshow(data, cmap=cmap)
        axes[idx].set_title(f'{filename}\n{data.shape}', fontsize=8)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_patient_folder(patient_folder, cmap='gray'):
    """
    Visualize all .npy files in a patient folder
    
    Args:
        patient_folder: Path to patient folder containing .npy files
        cmap: Colormap
    """
    import glob
    
    # Get all .npy files
    npy_files = sorted(glob.glob(os.path.join(patient_folder, '*.npy')))
    
    if len(npy_files) == 0:
        print(f"No .npy files found in {patient_folder}")
        return
    
    print(f"Found {len(npy_files)} .npy files in {patient_folder}")
    
    visualize_multiple_npy(npy_files, cmap=cmap)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Visualize single file
    # visualize_npy('camus_sr_slices/patient0001/patient0001_2CH_ED.npy')
    
    # Example 2: Visualize multiple files
    # files = [
    #     'camus_sr_slices/patient0001/patient0001_2CH_ED.npy',
    #     'camus_sr_slices/patient0001/patient0001_2CH_ES.npy',
    # ]
    # visualize_multiple_npy(files)
    
    # Example 3: Visualize entire patient folder
    visualize_patient_folder('/home/pks/Desktop/Peeyush/cardiac_work/diffuison_work/acdc/Data/slices_acdc/test/patient101')
    
    # Example 4: Different colormaps
    # visualize_npy('camus_sr_slices/patient0001/patient0001_2CH_ED.npy', cmap='hot')