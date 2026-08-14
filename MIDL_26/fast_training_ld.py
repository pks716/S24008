# import data loader

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import pandas as pd
from collections import deque
import pickle
import torch.nn.functional as F
import pandas as pd
from training_hyperparameters import *
from splits import SPLITS
from train_loader_acdc_sr import train_dataset
from test_loader_acdc_sr import test_dataset
from tqdm import tqdm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision.utils import make_grid, save_image
import json
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt


# Fast-DDPM imports
from models.diffusion import Model as DiffusionModel
from models.ema import EMAHelper
from functions.denoising import sg_generalized_steps

if wand_db_boolean:
    import wandb

DEVICE = HP['DEVICE']

# After DEVICE definition, add VAE initialization
print("Loading Stable Diffusion VAE...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae = vae.to(DEVICE)
vae.eval()  # Freeze VAE in eval mode
for param in vae.parameters():
    param.requires_grad = False
print("✓ VAE loaded and frozen")

ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

base_path = f"/home/pks/Desktop/Peeyush/cardiac_work/diffuison_work/acdc/sessions/{EXPERIMENT_NAME}"

# Fast-DDPM utility functions
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
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

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

# Modify sg_noise_estimation_loss to work with latents
def sg_noise_estimation_loss(model, x_img_latent, x_gt_latent, t, e, b, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x_gt_latent * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x_img_latent, x], dim=1), t.float())
    
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

# Fast-DDPM Configuration
class FastDDPMConfig:
    def __init__(self):
        # Model config
        self.model = type('ModelConfig', (), {})()
        self.model.type = "sg"  # single guidance for medical imaging
        self.model.in_channels = 8  # CT + noisy MR
        self.model.out_ch = 4  # clean MR
        self.model.ch = HP['model_params']['num_channels']
        self.model.ch_mult = [1, 1, 2, 2, 4, 4]
        self.model.num_res_blocks = 2
        self.model.attn_resolutions = [16]
        self.model.dropout = 0.0
        self.model.var_type = "fixedsmall"
        self.model.ema_rate = 0.999
        self.model.ema = True
        self.model.resamp_with_conv = True
        
        # Data config
        self.data = type('DataConfig', (), {})()
        self.data.image_size = 32
        self.data.channels = 4
        self.data.rescaled = True
        
        # Diffusion config
        self.diffusion = type('DiffusionConfig', (), {})()
        self.diffusion.beta_schedule = HP['diffusion_beta_schedule']
        self.diffusion.beta_start = HP['diffusion_beta_start']
        self.diffusion.beta_end = HP['diffusion_beta_end']
        self.diffusion.num_diffusion_timesteps = HP['diffusion_num_timesteps']


# Add VAE encoding/decoding helper functions
def encode_to_latent(vae, images):
    """Encode images to latent space using VAE encoder"""
    with torch.no_grad():
        # Ensure VAE is on same device as images (critical for validation)
        target_device = images.device
        if next(vae.parameters()).device != target_device:
            vae.to(target_device)
        
        # Expand grayscale to 3 channels (VAE expects RGB)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # SD VAE expects [-1, 1] range
        images_normalized = images * 2.0 - 1.0
        latent_dist = vae.encode(images_normalized).latent_dist
        latents = latent_dist.sample()
        # SD VAE uses scaling factor
        latents = latents * 0.18215
    return latents

def decode_from_latent(vae, latents):
    """Decode latents back to pixel space using VAE decoder"""
    with torch.no_grad():
        # Ensure VAE is on same device as latents (critical for validation)
        target_device = latents.device
        if next(vae.parameters()).device != target_device:
            vae.to(target_device)
        
        # Unscale latents
        latents = latents / 0.18215
        images = vae.decode(latents).sample
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        # Convert RGB back to grayscale (average across channels)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        
    return images

# Data transform functions for Fast-DDPM
def data_transform_ddpm(x):
    # Convert from [0,1] to [-1,1] for diffusion model
    return 2 * x - 1.0

def inverse_data_transform_ddpm(x):
    # Convert from [-1,1] to [0,1]
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

# MODIFIED: Save checkpoint with iteration instead of epoch
def save_checkpoint(model, optimizer, iteration, path, ema_helper=None):
    if '/' in path:
        name = path.split('/')[-1]
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
    
    states = {
        'iteration': iteration,  # CHANGED: iteration instead of epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if ema_helper is not None:
        states['ema_state_dict'] = ema_helper.state_dict()
    
    torch.save(states, path)

# MODIFIED: Load checkpoint with iteration
def load_checkpoint(model, optimizer, checkpoint, device=None, ema_helper=None):
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint.get('model_state_dict', {}))
    else:
        raise ValueError("Checkpoint does not contain key for model_state_dict")
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict',{}))
    else:
        raise ValueError("Checkpoint does not contain key for optimizer_state_dict")
    
    if ema_helper is not None and 'ema_state_dict' in checkpoint:
        ema_helper.load_state_dict(checkpoint['ema_state_dict'])
    
    # CHANGED: Return iteration instead of epoch
    return checkpoint.get('iteration', 0)

def model_restore_state(check_point_path):
    model_ckpt_name = check_point_path.split('/')[-1]
    # CHANGED: Parse iteration instead of epoch
    if 'iter_' in model_ckpt_name:
        iteration = model_ckpt_name.split('iter_')[1].split('__')[0]
        splitd = model_ckpt_name.split('__')[1]
        return int(iteration), int(splitd.split("_")[1])
    else:
        # Fallback for old epoch-based checkpoints
        epoch = model_ckpt_name.split('__')[0]
        splitd = model_ckpt_name.split('__')[1]
        return int(epoch.split("_")[1]), int(splitd.split("_")[1])

def load_checkpoint_eval(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Modify sample_fast_ddpm to work with latents
def sample_fast_ddpm(model, ct_latent, config, betas, timesteps=10, scheduler_type='uniform', device='cuda'):
    model.eval()
    with torch.no_grad():
        n = ct_latent.shape[0]
        
        # Initialize random noise in LATENT space
        mr_fake_latent = torch.randn(
            n, 4, 32, 32,  # CHANGED: latent dimensions
            device=device
        )
        
        # Define sampling schedule (unchanged)
        if scheduler_type == 'uniform':
            skip = config.diffusion.num_diffusion_timesteps // timesteps
            seq = list(range(-1, config.diffusion.num_diffusion_timesteps, skip))
            seq[0] = 0
        elif scheduler_type == 'non-uniform':
            seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]
            if timesteps != 10:
                num_1 = int(timesteps * 0.4)
                num_2 = int(timesteps * 0.6)
                stage_1 = np.linspace(0, 699, num_1+1)[:-1]
                stage_2 = np.linspace(699, 999, num_2)
                stage_1 = np.ceil(stage_1).astype(int)
                stage_2 = np.ceil(stage_2).astype(int)
                seq = np.concatenate((stage_1, stage_2)).tolist()
        
        # Perform denoising in LATENT space
        xs = sg_generalized_steps(mr_fake_latent, ct_latent, seq, model, betas, eta=0.0)
        mr_fake_latent = xs[0][-1]
        
        return mr_fake_latent

# Main training loop
split_resolved, iter_resolved = True, True  # CHANGED: iter_resolved instead of epoch_resolved
if continue_path:
    split_resolved, iter_resolved = False, False

for key, splitc in SPLITS.items():
    # Initialize Fast-DDPM
    ddpm_config = FastDDPMConfig()
    
    # Create diffusion model
    model = DiffusionModel(ddpm_config).to(DEVICE)
    model = torch.nn.DataParallel(model)
    
    # Initialize beta schedule
    betas = get_beta_schedule(
        beta_schedule=ddpm_config.diffusion.beta_schedule,
        beta_start=ddpm_config.diffusion.beta_start,
        beta_end=ddpm_config.diffusion.beta_end,
        num_diffusion_timesteps=ddpm_config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(DEVICE)
    
    # Initialize optimizer (REMOVED scheduler - not needed for iteration-based training)
    optimizer = optim.Adam(model.parameters(), lr=HP['learning_rate'])
    
    # Initialize EMA
    ema_helper = EMAHelper(mu=ddpm_config.model.ema_rate)
    ema_helper.register(model)
    
    best_models_psnr = deque(maxlen=3)
    best_models_ssim = deque(maxlen=3)

    # CHANGED: Track iterations instead of epochs
    global_iteration = 0
    max_iterations = HP['n_iters']  # 5,000,000
    snapshot_freq = HP['snapshot_freq']  # 100,000
    validation_freq = HP.get('validation_freq', 5000000000)  # 5 billion (essentially disabled)
    
    # Resume logic
    if not split_resolved:
        iter_num, spll = model_restore_state(continue_path)
        if key < spll:
            continue
        checkpoint = torch.load(continue_path, map_location=DEVICE)
        global_iteration = load_checkpoint(model, optimizer, checkpoint, DEVICE, ema_helper)  # CHANGED
        
        top_location = ""
        for ele in continue_path.split('/'):
            if ele == "ssim" or ele == "psnr":
                break
            top_location = f"{top_location}/{ele}"
        
        with open(f"{top_location}/Top3PSNR.pkl", "rb") as f:
            best_models_psnr = pickle.load(f)
        with open(f"{top_location}/Top3SSIM.pkl", "rb") as f:
            best_models_ssim = pickle.load(f)
        
        split_resolved = True
    
    if wand_db_boolean:
        run = wandb.init(project=PROJECT, 
                        name=f"{EXPERIMENT_NAME}_FastDDPM_Split-{key}",
                        config=HP,
                        resume="allow"
                        )

    print(f"Processing SPLIT {key} with Fast-DDPM")
    print(f"Starting from iteration {global_iteration}/{max_iterations}")
    split_base_path = f"{base_path}/SPLIT_{key}"
    os.makedirs(split_base_path, exist_ok=True)
    
    trr = train_dataset(splitc['train'])
    train_dataloader = DataLoader(trr, batch_size=HP['batch_size'], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    print(f"Split {key} - Train: {len(splitc['train'])}, Validation: {len(splitc['validation'])}")

    # CHANGED: Infinite iterator for iteration-based training
    train_iter = iter(train_dataloader)
    
    # CHANGED: Progress bar tracks iterations
    pbar = tqdm(initial=global_iteration, total=max_iterations, 
                desc=f"Split {key} Fast-DDPM Training")
    
    # CHANGED: Main training loop is iteration-based
    while global_iteration < max_iterations:
        model.train()
        
        # CHANGED: Get next batch (reset iterator when needed)
        try:
            low_res, high_res = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            low_res, high_res = next(train_iter)
        
        low_res, high_res = low_res.to(DEVICE), high_res.to(DEVICE)
        
        # low_res = low_res.unsqueeze(1)
        # high_res = high_res.unsqueeze(1)
        
        # ENCODE to latent space
        low_res_latent = encode_to_latent(vae, low_res)
        high_res_latent = encode_to_latent(vae, high_res)
        
        # Fast-DDPM training with custom timestep sampling
        n = low_res.shape[0]
        e = torch.randn_like(high_res_latent)
        
        # Fast-DDPM timestep sampling
        timesteps = HP['fast_ddpm_timesteps']
        scheduler_type = HP['scheduler_type']
        
        if scheduler_type == 'uniform':
            skip = ddpm_config.diffusion.num_diffusion_timesteps // timesteps
            t_intervals = torch.arange(-1, ddpm_config.diffusion.num_diffusion_timesteps, skip)
            t_intervals[0] = 0
        elif scheduler_type == 'non-uniform':
            if timesteps == 5:
                t_intervals = torch.tensor([0, 199, 499, 799, 999])
            elif timesteps == 10:
                t_intervals = torch.tensor([0, 199, 399, 599, 699, 799, 849, 899, 949, 999])
            else:
                t_intervals = torch.linspace(0, 999, timesteps).int()
        
        # Antithetic sampling
        idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
        idx_2 = len(t_intervals) - idx_1 - 1
        idx = torch.cat([idx_1, idx_2], dim=0)[:n]
        t = t_intervals[idx].to(DEVICE)
        
        # Compute loss
        loss = sg_noise_estimation_loss(model, low_res_latent, high_res_latent, t, e, betas)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Update EMA
        ema_helper.update(model)
        
        global_iteration += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", iter=global_iteration)
        
        # CHANGED: WandB logging every 100 iterations
        if wand_db_boolean and global_iteration % 100 == 0:
            wandb.log({
                "iteration": global_iteration,
                "train_loss": loss.item(),
                'tag': "Training Loss"
            })
        
        # CHANGED: Checkpoint every snapshot_freq iterations
        if global_iteration % snapshot_freq == 0:
            checkpoint_path = f"{split_base_path}/checkpoints/iter_{global_iteration:07d}__split_{key}__.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            save_checkpoint(model, optimizer, global_iteration, checkpoint_path, ema_helper)
            pbar.write(f"✓ Checkpoint saved at iteration {global_iteration}")
        

        # CHANGED: Validation every validation_freq iterations
        if global_iteration % validation_freq == 0:
            pbar.write(f"Running validation at iteration {global_iteration}...")
            
            # Validation
            model.eval()
            vae.to(DEVICE)
            vae.eval() 
            
            # Define degradation types to test
            DEGRADATION_TYPES = ['bicubic', 'bilinear', 'area', 'gaussian_blur', 'motion_blur', 'rician_noise', 'anisotropic', 'combined']
            
            # Store metrics per degradation type
            degradation_metrics = {deg: {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []} for deg in DEGRADATION_TYPES}
            
            validation_metrics_iter = {
                'iteration': global_iteration,
            }
            
            with torch.no_grad():
                model_path_val = f"{split_base_path}/model_weights"
                os.makedirs(f"{model_path_val}/ssim", exist_ok=True)
                os.makedirs(f"{model_path_val}/psnr", exist_ok=True)
                os.makedirs(f"{split_base_path}/validation", exist_ok=True)
                
                splitc['validation'] = [ele for ele in splitc['validation'] if '.DS_Store' not in ele]
                
                # Store training weights
                ema_helper.store(model.named_parameters())
                # Use EMA model for validation
                ema_helper.ema(model)
                
                # Loop through each validation patient
                for val_patient_idx, patient_path in enumerate(splitc['validation']):
                    patient_name = os.path.basename(patient_path)
                    pbar.write(f"  Validating patient {val_patient_idx+1}/{len(splitc['validation'])}: {patient_name}")
                    
                    # Create figure for this patient showing all degradations
                    n_degradations = len(DEGRADATION_TYPES)
                    fig, axes = plt.subplots(n_degradations, 4, figsize=(16, 4*n_degradations))
                    if n_degradations == 1:
                        axes = axes.reshape(1, -1)
                    
                    # Test each degradation type
                    for deg_idx, deg_type in enumerate(DEGRADATION_TYPES):
                        # Create dataset with specific degradation
                        validation_DS = test_dataset(
                            patient_path, 
                            training_type=HP['training_type'],
                            scale_factor=2,
                            degradation_type=deg_type
                        )
                        
                        if len(validation_DS) == 0:
                            continue
                        
                        validation_loader = DataLoader(
                            validation_DS, 
                            batch_size=HP['batch_size'], 
                            shuffle=False, 
                            num_workers=4
                        )
                        
                        # Metrics for this degradation
                        deg_psnr_total = 0.0
                        deg_ssim_total = 0.0
                        deg_mse_total = 0.0
                        deg_mae_total = 0.0
                        deg_slice_count = 0
                        
                        # Store first sample for visualization
                        first_sample = None
                        
                        for idx, (low_res, high_res) in enumerate(validation_loader):
                            source_real = low_res.to(DEVICE)
                            target_real = high_res.to(DEVICE)
                            
                            # ENCODE to latent space
                            source_latent = encode_to_latent(vae, source_real)
                            
                            # Generate in LATENT space using Fast-DDPM
                            target_fake_latent = sample_fast_ddpm(
                                model, source_latent, ddpm_config, betas, 
                                timesteps=HP['fast_ddpm_timesteps'], 
                                scheduler_type=HP['scheduler_type'], 
                                device=DEVICE
                            )
                            
                            # DECODE back to pixel space
                            target_fake = decode_from_latent(vae, target_fake_latent)
                            
                            # Resize if needed
                            target_size = (target_real.shape[2], target_real.shape[3])
                            target_fake = F.interpolate(
                                target_fake, size=target_size, 
                                mode=HP['inference_interpolation_mode'], 
                                align_corners=HP['inference_interpolation_allign_cornors']
                            ).to(DEVICE)
                            
                            target_fake = torch.clamp(target_fake, 0, 1)
                            
                            # Calculate metrics
                            batch_size = source_real.shape[0]
                            deg_psnr_total += psnr(target_real, target_fake).mean().item() * batch_size
                            deg_ssim_total += ssim(target_real, target_fake).mean().item() * batch_size
                            deg_mse_total += F.mse_loss(target_real, target_fake).item() * batch_size
                            deg_mae_total += F.l1_loss(target_real, target_fake).item() * batch_size
                            deg_slice_count += batch_size
                            
                            # Save first sample for visualization
                            if idx == 0 and first_sample is None:
                                first_sample = {
                                    'lowres': source_real[0].cpu().squeeze().numpy(),
                                    'highres': target_real[0].cpu().squeeze().numpy(),
                                    'enhanced': target_fake[0].cpu().squeeze().numpy()
                                }
                        
                        # Store average metrics for this degradation
                        if deg_slice_count > 0:
                            degradation_metrics[deg_type]['PSNR'].append(deg_psnr_total / deg_slice_count)
                            degradation_metrics[deg_type]['SSIM'].append(deg_ssim_total / deg_slice_count)
                            degradation_metrics[deg_type]['MSE'].append(deg_mse_total / deg_slice_count)
                            degradation_metrics[deg_type]['MAE'].append(deg_mae_total / deg_slice_count)
                        
                        # Visualize first sample
                        if first_sample is not None:
                            import matplotlib.pyplot as plt
                            
                            # Column 0: Ground truth
                            axes[deg_idx, 0].imshow(first_sample['highres'], cmap='gray', vmin=0, vmax=1)
                            axes[deg_idx, 0].set_title('Ground Truth', fontsize=10)
                            axes[deg_idx, 0].axis('off')
                            
                            # Column 1: Degraded input
                            axes[deg_idx, 1].imshow(first_sample['lowres'], cmap='gray', vmin=0, vmax=1)
                            axes[deg_idx, 1].set_title(f'{deg_type.replace("_", " ").title()}\n(Degraded)', fontsize=10)
                            axes[deg_idx, 1].axis('off')
                            
                            # Column 2: Enhanced output
                            axes[deg_idx, 2].imshow(first_sample['enhanced'], cmap='gray', vmin=0, vmax=1)
                            axes[deg_idx, 2].set_title('Enhanced\n(Fast-LDM)', fontsize=10)
                            axes[deg_idx, 2].axis('off')
                            
                            # Column 3: Difference map (enhanced - ground truth)
                            diff = np.abs(first_sample['enhanced'] - first_sample['highres'])
                            im = axes[deg_idx, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
                            axes[deg_idx, 3].set_title('Error Map', fontsize=10)
                            axes[deg_idx, 3].axis('off')
                            
                            # Add colorbar to last row
                            if deg_idx == n_degradations - 1:
                                plt.colorbar(im, ax=axes[deg_idx, 3], fraction=0.046, pad=0.04)
                    
                    # Add metrics as text
                    fig.text(0.02, 0.98, 
                            f'Patient: {patient_name} | Iteration: {global_iteration}', 
                            fontsize=12, fontweight='bold', va='top')
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.97])
                    
                    # Save figure
                    save_path = f"{split_base_path}/validation/{patient_name}_iter{global_iteration}_all_degradations.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Log to wandb if enabled
                    if wand_db_boolean:
                        wandb.log({
                            f"Validation_{patient_name}_MultiDeg": wandb.Image(save_path),
                            "iteration": global_iteration,
                            'tag': "Validation Multi-Degradation",
                        })
                
                # Restore training weights
                ema_helper.restore(model.named_parameters())
            
            # Calculate and log average metrics across all degradations
            pbar.write(f"\n{'='*60}")
            pbar.write(f"Validation Results at Iteration {global_iteration}")
            pbar.write(f"{'='*60}")
            
            best_psnr = 0
            best_ssim = 0
            
            for deg_type in DEGRADATION_TYPES:
                if len(degradation_metrics[deg_type]['PSNR']) > 0:
                    avg_psnr = np.mean(degradation_metrics[deg_type]['PSNR'])
                    avg_ssim = np.mean(degradation_metrics[deg_type]['SSIM'])
                    avg_mse = np.mean(degradation_metrics[deg_type]['MSE'])
                    avg_mae = np.mean(degradation_metrics[deg_type]['MAE'])
                    
                    pbar.write(f"{deg_type.upper():15s} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | MSE: {avg_mse:.6f} | MAE: {avg_mae:.6f}")
                    
                    # Store in validation metrics
                    validation_metrics_iter[f'{deg_type}_PSNR'] = avg_psnr
                    validation_metrics_iter[f'{deg_type}_SSIM'] = avg_ssim
                    validation_metrics_iter[f'{deg_type}_MSE'] = avg_mse
                    validation_metrics_iter[f'{deg_type}_MAE'] = avg_mae
                    
                    # Track best overall metrics (using bicubic as reference)
                    if deg_type == 'bicubic':
                        best_psnr = avg_psnr
                        best_ssim = avg_ssim
            
            pbar.write(f"{'='*60}\n")
            
            # Save detailed results to JSON
            results_json = {
                'iteration': global_iteration,
                'degradation_metrics': {
                    deg: {
                        'psnr': float(np.mean(degradation_metrics[deg]['PSNR'])) if degradation_metrics[deg]['PSNR'] else 0,
                        'ssim': float(np.mean(degradation_metrics[deg]['SSIM'])) if degradation_metrics[deg]['SSIM'] else 0,
                        'mse': float(np.mean(degradation_metrics[deg]['MSE'])) if degradation_metrics[deg]['MSE'] else 0,
                        'mae': float(np.mean(degradation_metrics[deg]['MAE'])) if degradation_metrics[deg]['MAE'] else 0,
                    }
                    for deg in DEGRADATION_TYPES
                }
            }
            
            os.makedirs(f"{split_base_path}/validation_metrics", exist_ok=True)
            with open(f"{split_base_path}/validation_metrics/iter_{global_iteration}_metrics.json", "w") as f:
                json.dump(results_json, f, indent=2)
            
            # Log to wandb
            if wand_db_boolean:
                wandb.log(validation_metrics_iter)
            
            # Model checkpointing based on bicubic metrics (standard reference)
            if best_psnr > 0:
                best_models_psnr.append((best_psnr, f"{model_path_val}/psnr/iter_{global_iteration}__split_{key}__.pth"))
                best_models_ssim.append((best_ssim, f"{model_path_val}/ssim/iter_{global_iteration}__split_{key}__.pth"))
                
                # Save best models
                best_models_psnr = deque(sorted(best_models_psnr, reverse=True)[:3], maxlen=3)
                best_models_ssim = deque(sorted(best_models_ssim, reverse=True)[:3], maxlen=3)
                
                top_models_psnr = set(m[1] for m in best_models_psnr)
                top_models_ssim = set(m[1] for m in best_models_ssim)
                
                if f"{model_path_val}/psnr/iter_{global_iteration}__split_{key}__.pth" in top_models_psnr:
                    save_checkpoint(model, optimizer, global_iteration, f"{model_path_val}/psnr/iter_{global_iteration}__split_{key}__.pth", ema_helper)
                    with open(f"{model_path_val}/Top3PSNR.pkl", "wb") as f:
                        pickle.dump(best_models_psnr, f)
                
                if f"{model_path_val}/ssim/iter_{global_iteration}__split_{key}__.pth" in top_models_ssim:
                    save_checkpoint(model, optimizer, global_iteration, f"{model_path_val}/ssim/iter_{global_iteration}__split_{key}__.pth", ema_helper)
                    with open(f"{model_path_val}/Top3SSIM.pkl", "wb") as f:
                        pickle.dump(best_models_ssim, f)
                
                # Clean up old models
                for file in os.listdir(f"{model_path_val}/psnr"):
                    file_path = os.path.join(f"{model_path_val}/psnr", file)
                    if file_path not in top_models_psnr and file.endswith('.pth'):
                        os.remove(file_path)
                
                for file in os.listdir(f"{model_path_val}/ssim"):
                    file_path = os.path.join(f"{model_path_val}/ssim", file)
                    if file_path not in top_models_ssim and file.endswith('.pth'):
                        os.remove(file_path)
                
                # Save CSV reports
                df = pd.DataFrame.from_dict(dict(best_models_psnr), orient='index')
                df.to_csv(f"{model_path_val}/Top3PSNR.csv", index_label="Iteration")
                
                df = pd.DataFrame.from_dict(dict(best_models_ssim), orient='index')
                df.to_csv(f"{model_path_val}/Top3SSIM.csv", index_label="Iteration")
                
                pbar.write(f'✓ Best Models - PSNR: {best_models_psnr[0][0]:.2f} dB | SSIM: {best_models_ssim[0][0]:.4f}')
            
    pbar.close()
    print(f'Split {key} Fast-DDPM training completed at {global_iteration} iterations')

    if wand_db_boolean:
        final_metrics = {
            "Final_Iterations": global_iteration,
            "Split": key,
            "tag": "Final Metrics"
        }
        if len(best_models_psnr) > 0:
            final_metrics["Final_Best_PSNR"] = best_models_psnr[0][0]
        if len(best_models_ssim) > 0:
            final_metrics["Final_Best_SSIM"] = best_models_ssim[0][0]
        wandb.log(final_metrics)
        run.finish()

print("\n" + "="*80)
print("FAST-DDPM ITERATION-BASED TRAINING COMPLETED")
print("="*80)
print(f"• Total iterations: {HP['n_iters']:,}")
print(f"• Checkpoint frequency: Every {HP['snapshot_freq']:,} iterations")
print(f"• Validation frequency: Every {HP.get('validation_freq', 5000000000):,} iterations")
print("="*80)