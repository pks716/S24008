"""
Phase 2 Inference: Patient-Wise Cross-Modal Synthesis Evaluation
================================================================

1. Loads trained Phase 2 model
2. Processes entire patients slice-by-slice
3. Generates all cross-modal translations (e.g., T1→T2, T2→T1, etc.)
4. Computes patient-wise metrics (PSNR, SSIM)
5. Saves results and visualizations

Metrics are calculated:
- Per patient (aggregate all slices)
- Per source-target modality pair
- Per slice (optional)
"""

import torch
import numpy as np
import os
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.cuda.amp import autocast
import pandas as pd

# Import models and utilities
from diffusion_film import Model as DiffusionModel
from ema import EMAHelper
from train_phase1 import ContrastiveModel
from test_dataloader import create_test_dataloader


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
PHASE1_CHECKPOINT = ""
PHASE2_CHECKPOINT = ""

# Data paths
TEST_DATA_ROOT = ""

# Output paths
RESULTS_DIR = ""
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/visualizations", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/metrics", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/volumes", exist_ok=True)

# Inference settings
MODALITIES = ["t1", "t1ce", "t2", "flair"]
TARGET_SIZE = (256, 256)
NUM_WORKERS = 4

# Fast-DDPM sampling parameters
NUM_DIFFUSION_TIMESTEPS = 1000
FAST_DDPM_TIMESTEPS = 10
SCHEDULER_TYPE = 'non-uniform'
BETA_SCHEDULE = 'linear'
BETA_START = 0.0001
BETA_END = 0.02

# Visualization settings
SAVE_VOLUMES = True  # Save generated 3D volumes
SAVE_SLICE_VISUALIZATIONS = True  # Save slice comparisons
VIS_SLICE_STEP = 10  # Save every Nth slice for visualization

# Mixed precision
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16


# ==============================================================================
# MODEL CONFIGURATION AND LOADING
# ==============================================================================

class ConditionedFastDDPMConfig:
    """Fast-DDPM configuration matching training setup"""
    def __init__(self):
        self.model = type('ModelConfig', (), {})()
        self.model.type = "sg"
        self.model.in_channels = 1 + 1  # noisy_image + edge_map
        self.model.out_ch = 1        
        self.model.ch = 128
        self.model.ch_mult = [1, 1, 2, 2, 4, 4]
        self.model.num_res_blocks = 2
        self.model.attn_resolutions = [16]
        self.model.dropout = 0.0
        self.model.var_type = "fixedsmall"
        self.model.ema_rate = 0.999
        self.model.ema = True
        self.model.resamp_with_conv = True
        
        self.data = type('DataConfig', (), {})()
        self.data.image_size = 256   
        self.data.channels = 1
        self.data.rescaled = True
        
        self.diffusion = type('DiffusionConfig', (), {})()
        self.diffusion.beta_schedule = BETA_SCHEDULE
        self.diffusion.beta_start = BETA_START
        self.diffusion.beta_end = BETA_END
        self.diffusion.num_diffusion_timesteps = NUM_DIFFUSION_TIMESTEPS


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Beta schedule for diffusion process"""
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def data_transform_ddpm(x):
    """Transform [0,1] → [-1,1]"""
    return 2 * x - 1.0


def inverse_data_transform_ddpm(x):
    """Transform [-1,1] → [0,1]"""
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def sobel_edge_map(img):
    """Compute per-image normalized Sobel edge magnitude"""
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=img.dtype, device=img.device
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=img.dtype, device=img.device
    ).unsqueeze(0).unsqueeze(0)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    # Per-image normalization
    B = grad_mag.size(0)
    grad_min = grad_mag.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
    grad_max = grad_mag.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    edges_norm = (grad_mag - grad_min) / (grad_max - grad_min + 1e-6)

    # Optional smoothing
    edges_norm = F.avg_pool2d(edges_norm, kernel_size=3, stride=1, padding=1)

    return edges_norm


def load_phase1_model(checkpoint_path, device):
    """Load frozen Phase 1 contrastive model"""
    print(f"Loading Phase 1 model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = ContrastiveModel( #This is dependenet upon stage 1 model, a higher capacity model will lead to better embeddings
        encoder_dim=768,
        proj_dim=config['proj_dim'],
        hidden_dim=512,
        num_modalities=len(config['modalities'])
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Phase 1 model loaded and frozen")
    return model, config


def load_phase2_model(checkpoint_path, device):
    """Load Phase 2 diffusion model"""
    print(f"Loading Phase 2 model from {checkpoint_path}")
    
    config = ConditionedFastDDPMConfig()
    model = DiffusionModel(config).to(device)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load EMA weights if available
    if 'ema_state_dict' in checkpoint:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(checkpoint['ema_state_dict'])
        ema_helper.ema(model)
        print("Loaded EMA weights")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")
    
    model.eval()
    
    print(f"Phase 2 model loaded")
    return model, config


# ==============================================================================
# SAMPLING FUNCTION
# ==============================================================================

def sample_conditioned_fast_ddpm(
    model,
    x_source_image,
    anatomy_emb,
    contrast_emb,
    config,
    betas,
    timesteps=10,
    scheduler_type='non-uniform',
    device='cuda'
):
    """Sample from conditioned diffusion model"""
    model.eval()
    with torch.no_grad():
        B = anatomy_emb.shape[0]
        H, W = config.data.image_size, config.data.image_size

        # Edge map from source
        edge_map = sobel_edge_map(x_source_image).detach()
        
        # Start from pure noise
        x = torch.randn(B, 1, H, W, device=device)
        
        # Sampling schedule
        if scheduler_type == 'uniform':
            skip = config.diffusion.num_diffusion_timesteps // timesteps
            seq = list(range(-1, config.diffusion.num_diffusion_timesteps, skip))
            seq[0] = 0
        elif scheduler_type == 'non-uniform':
            if timesteps == 5:
                seq = [0, 199, 499, 799, 999]
            elif timesteps == 10:
                seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]
            else:
                num_1 = int(timesteps * 0.4)
                num_2 = int(timesteps * 0.6)
                stage_1 = np.linspace(0, 699, num_1+1)[:-1]
                stage_2 = np.linspace(699, 999, num_2)
                stage_1 = np.ceil(stage_1).astype(int)
                stage_2 = np.ceil(stage_2).astype(int)
                seq = np.concatenate((stage_1, stage_2)).tolist()
        
        # Denoising loop
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.ones(B, device=device).long() * i
            next_t = torch.ones(B, device=device).long() * j
            
            # Compute alphas
            a_cum = (1 - betas).cumprod(dim=0)
            t = t.clamp(0, len(a_cum)-1).long()
            next_t = next_t.clamp(-1, len(a_cum)-1).long()

            at = a_cum.index_select(0, t).view(-1, 1, 1, 1)
            at_next = a_cum.index_select(0, next_t.clamp(min=0)).view(-1, 1, 1, 1)
            
            xt = xs[-1].to(device)
            
            cond = torch.cat([anatomy_emb, contrast_emb], dim=1)
            model_input = torch.cat([xt, edge_map], dim=1)
            et = model(model_input, t.float(), cond=cond)
            
            # DDIM update
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = 0.0
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et
            
            xs.append(xt_next.to('cpu'))
        
        return xs[-1].to(device)


# ==============================================================================
# INFERENCE FUNCTION
# ==============================================================================

def run_inference():
    """Main inference loop - processes all patients"""
    
    print("="*80)
    print("PHASE 2 INFERENCE: PATIENT-WISE EVALUATION")
    print("="*80)
    
    # 1. Load models
    phase1_model, phase1_config = load_phase1_model(PHASE1_CHECKPOINT, DEVICE)
    phase2_model, phase2_config = load_phase2_model(PHASE2_CHECKPOINT, DEVICE)
    
    # 2. Setup beta schedule
    betas = get_beta_schedule(
        beta_schedule=phase2_config.diffusion.beta_schedule,
        beta_start=phase2_config.diffusion.beta_start,
        beta_end=phase2_config.diffusion.beta_end,
        num_diffusion_timesteps=phase2_config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(DEVICE)
    
    # 3. Create test dataloader
    test_loader = create_test_dataloader(
        test_root=TEST_DATA_ROOT,
        modalities=MODALITIES,
        target_size=TARGET_SIZE,
        num_workers=NUM_WORKERS,
        apply_mask=True
    )
    
    # 4. Initialize metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    
    # 5. Results storage
    all_results = []
    patient_results = defaultdict(lambda: defaultdict(list))
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(test_loader)} PATIENTS")
    print(f"{'='*80}\n")
    
    # 6. Process each patient
    for patient_idx, patient_data in enumerate(tqdm(test_loader, desc="Processing patients")):
        patient_name = patient_data['patient_name']
        num_slices = patient_data['num_slices']
        images = patient_data['images']  # Dict of [num_slices, 1, H, W]
        
        print(f"\n{'='*60}")
        print(f"Patient: {patient_name} ({num_slices} slices)")
        print(f"{'='*60}")
        
        # Process all source-target modality pairs
        for source_mod_idx, source_mod_name in enumerate(MODALITIES):
            source_volume = images[source_mod_name].to(DEVICE)  # [num_slices, 1, H, W]
            
            # Transform to [-1, 1]
            source_volume_transformed = data_transform_ddpm(source_volume)
            
            # Extract anatomy embeddings for all slices at once
            with torch.no_grad(), autocast(dtype=AMP_DTYPE, enabled=USE_AMP):
                features = phase1_model.encoder(source_volume_transformed)
                anatomy_embs = phase1_model.anatomy_head(features)  # [num_slices, 128]
            
            # Generate each target modality
            for target_mod_idx, target_mod_name in enumerate(MODALITIES):
                # Skip same modality (optional - you can include for identity mapping test)
                if source_mod_name == target_mod_name:
                    continue
                
                print(f"  Generating: {source_mod_name.upper()} → {target_mod_name.upper()}")
                
                # Get target ground truth
                target_volume_gt = images[target_mod_name].to(DEVICE)  # [num_slices, 1, H, W]
                
                # Get contrast code for target modality
                contrast_code = phase1_model.contrast_bank(
                    torch.tensor([target_mod_idx], device=DEVICE)
                )  # [1, 128]
                
                # Expand to batch size
                contrast_codes = contrast_code.expand(num_slices, -1)  # [num_slices, 128]
                
                # Generate all slices (batch processing for efficiency)
                batch_size = 8  # Process 8 slices at a time
                generated_slices = []
                
                for start_idx in range(0, num_slices, batch_size):
                    end_idx = min(start_idx + batch_size, num_slices)
                    batch_source = source_volume_transformed[start_idx:end_idx]
                    batch_anatomy = anatomy_embs[start_idx:end_idx]
                    batch_contrast = contrast_codes[start_idx:end_idx]
                    
                    with torch.no_grad(), autocast(dtype=AMP_DTYPE, enabled=USE_AMP):
                        batch_generated = sample_conditioned_fast_ddpm(
                            phase2_model,
                            batch_source,
                            batch_anatomy,
                            batch_contrast,
                            phase2_config,
                            betas,
                            timesteps=FAST_DDPM_TIMESTEPS,
                            scheduler_type=SCHEDULER_TYPE,
                            device=DEVICE
                        )
                    
                    # Transform back to [0, 1]
                    batch_generated = inverse_data_transform_ddpm(batch_generated)
                    batch_generated = torch.clamp(batch_generated, 0.0, 1.0)
                    generated_slices.append(batch_generated)
                
                # Concatenate all generated slices
                generated_volume = torch.cat(generated_slices, dim=0)  # [num_slices, 1, H, W]
                
                # Calculate metrics for each slice
                slice_psnrs = []
                slice_ssims = []
                
                for slice_idx in range(num_slices):
                    gt_slice = target_volume_gt[slice_idx:slice_idx+1]
                    gen_slice = generated_volume[slice_idx:slice_idx+1]
                    
                    # Calculate metrics
                    with torch.no_grad():
                        psnr_val = psnr_metric(gen_slice, gt_slice).item()
                        ssim_val = ssim_metric(gen_slice, gt_slice).item()
                    
                    slice_psnrs.append(psnr_val)
                    slice_ssims.append(ssim_val)
                
                # Patient-level metrics (average across slices)
                patient_psnr = np.mean(slice_psnrs)
                patient_ssim = np.mean(slice_ssims)
                
                print(f"    PSNR: {patient_psnr:.2f} dB, SSIM: {patient_ssim:.4f}")
                
                # Store results
                result_entry = {
                    'patient_name': patient_name,
                    'patient_idx': patient_idx,
                    'source_modality': source_mod_name,
                    'target_modality': target_mod_name,
                    'num_slices': num_slices,
                    'psnr_mean': patient_psnr,
                    'psnr_std': np.std(slice_psnrs),
                    'ssim_mean': patient_ssim,
                    'ssim_std': np.std(slice_ssims),
                    'slice_psnrs': slice_psnrs,
                    'slice_ssims': slice_ssims
                }
                all_results.append(result_entry)
                
                # Store for aggregation
                pair_key = f"{source_mod_name}_to_{target_mod_name}"
                patient_results[pair_key]['psnr'].append(patient_psnr)
                patient_results[pair_key]['ssim'].append(patient_ssim)
                
                # Save visualizations
                if SAVE_SLICE_VISUALIZATIONS:
                    vis_dir = f"{RESULTS_DIR}/visualizations/{patient_name}"
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # Save middle slice and a few others
                    vis_slices = [num_slices // 4, num_slices // 2, 3 * num_slices // 4]
                    
                    for vis_idx in vis_slices:
                        if vis_idx >= num_slices:
                            continue
                        
                        # Stack source, GT, generated
                        vis_images = torch.cat([
                            source_volume[vis_idx:vis_idx+1],
                            target_volume_gt[vis_idx:vis_idx+1],
                            generated_volume[vis_idx:vis_idx+1]
                        ], dim=0)  # [3, 1, H, W]
                        
                        save_path = f"{vis_dir}/slice{vis_idx:03d}_{pair_key}.png"
                        save_image(vis_images, save_path, nrow=3, normalize=False)
                
                # Save volumes
                if SAVE_VOLUMES:
                    vol_dir = f"{RESULTS_DIR}/volumes/{patient_name}"
                    os.makedirs(vol_dir, exist_ok=True)
                    
                    # Save as numpy arrays
                    np.savez_compressed(
                        f"{vol_dir}/{pair_key}_generated.npz",
                        volume=generated_volume.cpu().numpy(),
                        psnr=slice_psnrs,
                        ssim=slice_ssims
                    )
    
    # ==============================================================================
    # AGGREGATE RESULTS
    # ==============================================================================
    
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}\n")
    
    # Overall statistics
    overall_stats = {}
    for pair_key, metrics in patient_results.items():
        overall_stats[pair_key] = {
            'psnr_mean': np.mean(metrics['psnr']),
            'psnr_std': np.std(metrics['psnr']),
            'ssim_mean': np.mean(metrics['ssim']),
            'ssim_std': np.std(metrics['ssim']),
            'num_patients': len(metrics['psnr'])
        }
    
    # Print summary
    print("Overall Results (across all patients):")
    print("-" * 80)
    for pair_key, stats in overall_stats.items():
        source, target = pair_key.split('_to_')
        print(f"{source.upper():5s} → {target.upper():5s}: "
              f"PSNR = {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB, "
              f"SSIM = {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")
    
    # Save detailed results
    print(f"\nSaving results to {RESULTS_DIR}/metrics/")
    
    # Save JSON
    with open(f"{RESULTS_DIR}/metrics/all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open(f"{RESULTS_DIR}/metrics/overall_stats.json", 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    # Save CSV
    df = pd.DataFrame(all_results)
    df.to_csv(f"{RESULTS_DIR}/metrics/all_results.csv", index=False)
    
    # Create summary plots
    create_summary_plots(all_results, overall_stats)
    
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved to: {RESULTS_DIR}")


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_summary_plots(all_results, overall_stats):
    """Create summary visualization plots"""
    
    # 1. Box plot: PSNR and SSIM per modality pair
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data
    pairs = sorted(overall_stats.keys())
    psnr_data = []
    ssim_data = []
    labels = []
    
    for pair_key in pairs:
        pair_results = [r for r in all_results if 
                       f"{r['source_modality']}_to_{r['target_modality']}" == pair_key]
        psnr_data.append([r['psnr_mean'] for r in pair_results])
        ssim_data.append([r['ssim_mean'] for r in pair_results])
        source, target = pair_key.split('_to_')
        labels.append(f"{source.upper()}→{target.upper()}")
    
    # PSNR box plot
    axes[0].boxplot(psnr_data, labels=labels)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR Distribution per Modality Pair', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # SSIM box plot
    axes[1].boxplot(ssim_data, labels=labels)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM Distribution per Modality Pair', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/metrics/summary_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap: Average metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create matrices
    n_mods = len(MODALITIES)
    psnr_matrix = np.zeros((n_mods, n_mods))
    ssim_matrix = np.zeros((n_mods, n_mods))
    
    for pair_key, stats in overall_stats.items():
        source, target = pair_key.split('_to_')
        i = MODALITIES.index(source)
        j = MODALITIES.index(target)
        psnr_matrix[i, j] = stats['psnr_mean']
        ssim_matrix[i, j] = stats['ssim_mean']
    
    # PSNR heatmap
    im1 = axes[0].imshow(psnr_matrix, cmap='viridis', aspect='auto')
    axes[0].set_xticks(range(n_mods))
    axes[0].set_yticks(range(n_mods))
    axes[0].set_xticklabels([m.upper() for m in MODALITIES])
    axes[0].set_yticklabels([m.upper() for m in MODALITIES])
    axes[0].set_xlabel('Target Modality', fontsize=12)
    axes[0].set_ylabel('Source Modality', fontsize=12)
    axes[0].set_title('Average PSNR (dB)', fontsize=14, fontweight='bold')
    
    # Add values to cells
    for i in range(n_mods):
        for j in range(n_mods):
            if i != j:
                text = axes[0].text(j, i, f'{psnr_matrix[i, j]:.1f}',
                                   ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im1, ax=axes[0])
    
    # SSIM heatmap
    im2 = axes[1].imshow(ssim_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(range(n_mods))
    axes[1].set_yticks(range(n_mods))
    axes[1].set_xticklabels([m.upper() for m in MODALITIES])
    axes[1].set_yticklabels([m.upper() for m in MODALITIES])
    axes[1].set_xlabel('Target Modality', fontsize=12)
    axes[1].set_ylabel('Source Modality', fontsize=12)
    axes[1].set_title('Average SSIM', fontsize=14, fontweight='bold')
    
    # Add values to cells
    for i in range(n_mods):
        for j in range(n_mods):
            if i != j:
                text = axes[1].text(j, i, f'{ssim_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/metrics/summary_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plots saved to {RESULTS_DIR}/metrics/")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    run_inference()