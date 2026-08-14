"""
Phase 2: Fast-DDPM with Anatomy + Contrast Conditioning
========================================================

Key Changes:
- Input: noisy_image [B,1,H,W] + anatomy [B,128] + contrast [B,128]
- Model input channels: 1 + 128 + 128 = 257
- Loss: sg_noise_estimation_loss (single guidance)
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision.utils import make_grid, save_image
import json
from torch.cuda.amp import autocast, GradScaler


# Import existing Fast-DDPM components
from diffusion_film import Model as DiffusionModel
from ema import EMAHelper
from denoising import sg_generalized_steps

# Import Phase 1 model
from train_phase1 import ContrastiveModel

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
phase1_checkpoint = ""
train_data_root = ""
val_data_root = ""
base_path = ""
os.makedirs(base_path, exist_ok=True)

# Training hyperparameters
n_iters = 5000000  # 5Mil iterations 
snapshot_freq = 100000  # Save every 100k iterations
validation_freq = 50000  # Validate every 50k iterations
batch_size = 1
learning_rate = 2e-4

# Fast-DDPM parameters
num_diffusion_timesteps = 1000 
fast_ddpm_timesteps = 10  # 10 steps for sampling
scheduler_type = 'non-uniform'  # or 'uniform'
beta_schedule = 'linear'
beta_start = 0.0001
beta_end = 0.02

modalities = ["t1", "t1ce", "t2", "flair"]

# Metrics
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

# --- AMP CONFIGURATION ---
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16  # use bfloat16 (safe for stable training)
scaler = GradScaler(enabled=USE_AMP)


# ==============================================================================
# LOAD PHASE 1 MODEL (FROZEN)
# ==============================================================================

def load_phase1_model(checkpoint_path, device):
    """Load frozen Phase 1 contrastive model"""
    print(f"Loading Phase 1 model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = ContrastiveModel(
        encoder_dim=768,
        proj_dim=config['proj_dim'],
        hidden_dim=512,
        num_modalities=len(config['modalities'])
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # FREEZE all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Phase 1 model loaded and frozen")
    return model, config

# ==============================================================================
# MODIFIED FAST-DDPM CONFIG (WITH CONDITIONING)
# ==============================================================================

class ConditionedFastDDPMConfig:
    def __init__(self):
        # Model config
        self.model = type('ModelConfig', (), {})()
        self.model.type = "sg"
        self.model.in_channels = 1+1
        self.model.out_ch = 1        
        self.model.ch = 128
        self.model.ch_mult = [1,1,2,2,4,4]
        self.model.num_res_blocks = 2
        self.model.attn_resolutions = [16]
        self.model.dropout = 0.0
        self.model.var_type = "fixedsmall"
        self.model.ema_rate = 0.999
        self.model.ema = True
        self.model.resamp_with_conv = True
        
        # Data config 
        self.data = type('DataConfig', (), {})()
        self.data.image_size = 256   
        self.data.channels = 1
        self.data.rescaled = True
        
        # Diffusion config
        self.diffusion = type('DiffusionConfig', (), {})()
        self.diffusion.beta_schedule = beta_schedule
        self.diffusion.beta_start = beta_start
        self.diffusion.beta_end = beta_end
        self.diffusion.num_diffusion_timesteps = num_diffusion_timesteps


# ==============================================================================
# EDGE MAP UTILITY
# ==============================================================================

def sobel_edge_map(img):
    """
    Compute per-image normalized Sobel edge magnitude.
    Contrast-invariant, suitable for disentangled diffusion conditioning.

    Args:
        img: [B, 1, H, W], values in [-1, 1] or [0, 1].
    Returns:
        edges_norm: [B, 1, H, W], normalized to [0, 1].
    """
    import torch.nn.functional as F
    sobel_x = torch.tensor(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        dtype=img.dtype,
        device=img.device
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]],
        dtype=img.dtype,
        device=img.device
    ).unsqueeze(0).unsqueeze(0)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    # --- Per-image normalization (contrast invariant) ---
    B = grad_mag.size(0)
    grad_min = grad_mag.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
    grad_max = grad_mag.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
    edges_norm = (grad_mag - grad_min) / (grad_max - grad_min + 1e-3)

    # Optional: local smoothing for stability
    edges_norm = F.avg_pool2d(edges_norm, kernel_size=3, stride=1, padding=1)

    return edges_norm



# ==============================================================================
# FAST-DDPM UTILITIES
# ==============================================================================

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Your original function"""
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
    """Your original: [0,1] → [-1,1]"""
    return 2 * x - 1.0


def inverse_data_transform_ddpm(x):
    """Your original: [-1,1] → [0,1]"""
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def save_checkpoint(model, optimizer, iteration, path, ema_helper=None):
    """Your original save function"""
    if '/' in path:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
    
    states = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if ema_helper is not None:
        states['ema_state_dict'] = ema_helper.state_dict()
    
    torch.save(states, path)


# ==============================================================================
# MODIFIED LOSS FUNCTION (WITH ANATOMY + CONTRAST CONDITIONING)
# ==============================================================================

# def pixel_conditioned_diffusion_loss(
#     model,
#     x_gt_image,       # [B,1,256,256] - PIXEL image
#     x_source_image,   # [B,1,256,256] - SOURCE image (used for edges)
#     anatomy_emb,      # [B,128]
#     contrast_code,    # [B,128]
#     t,                # [B]
#     e,                # [B,1,256,256] noise
#     betas
# ):
#     """
#     Diffusion in PIXEL space with anatomy + contrast conditioning.
#     ALIGNED with sg_noise_estimation_loss behavior.
#     """
#     B, _, H, W = x_gt_image.shape

#     # --- Structural edge map guidance ---
#     # edge_map = sobel_edge_map(x_source_image).detach()

    
#     cond = torch.cat([anatomy_emb, contrast_code], dim=1)  # [B, 256]

#     # Add noise to IMAGE (same as working script)
#     # a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
#     a_cum = (1 - betas).cumprod(dim=0)
#     t = t.clamp(0, len(a_cum)-1).long()
#     a = a_cum.index_select(0, t).view(-1, 1, 1, 1)

#     x_noisy = x_gt_image * a.sqrt() + e * (1.0 - a).sqrt()

#     # Concatenate
#     # model_input = torch.cat([x_noisy, edge_map], dim=1)
#     model_input = x_noisy

#     # Predict noise with FiLM conditioning
#     pred_noise = model(model_input, t.float(), cond=cond)

#     # Loss: EXACT same as working script
#     loss = (pred_noise - e).square().sum(dim=(1, 2, 3)).mean(dim=0)

#     return loss

def pixel_conditioned_diffusion_loss(
    model,
    x_gt_image,
    x_source_image,
    anatomy_emb,
    contrast_code,
    t,
    e,
    betas,
    cond_drop_prob=0.2  # NEW parameter
):
    """
    Diffusion loss with classifier-free guidance training.
    """
    B, _, H, W = x_gt_image.shape
    
    cond = torch.cat([anatomy_emb, contrast_code], dim=1)  # [B, 256]

    edge_map = sobel_edge_map(x_source_image).detach()
    
    # === CRITICAL: Classifier-free guidance ===
    if model.training:
        # Random conditioning dropout
        drop_mask = (torch.rand(B, device=cond.device) < cond_drop_prob)
        # Zero out conditioning for dropped samples
        cond = cond * (~drop_mask).float().view(B, 1)
    
    a_cum = (1 - betas).cumprod(dim=0)
    t = t.clamp(0, len(a_cum)-1).long()
    a = a_cum.index_select(0, t).view(-1, 1, 1, 1)
    
    x_noisy = x_gt_image * a.sqrt() + e * (1.0 - a).sqrt()
    model_input = torch.cat([x_noisy, edge_map], dim=1)
    
    pred_noise = model(model_input, t.float(), cond=cond)
    
    loss = (pred_noise - e).square().sum(dim=(1, 2, 3)).mean(dim=0)
    
    return loss


# def conditioned_noise_estimation_loss(
#     model,
#     anatomy_emb,  # [B, 128] anatomy condition
#     contrast_emb,  # [B, 128] contrast condition
#     x_gt,  # [B, 1, H, W] clean image
#     t,  # [B] timesteps
#     e,  # [B, 1, H, W] noise
#     b,  # betas
#     keepdim=False
# ):
#     """
#     Modified sg_noise_estimation_loss to condition on anatomy + contrast.
    
#     Original: model(torch.cat([x_img, x_noisy], dim=1), t)
#     New:      model(torch.cat([anatomy_spatial, contrast_spatial, x_noisy], dim=1), t)
#     """
#     B, _, H, W = x_gt.shape
    
#     # Compute alpha (from your code)
#     a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    
#     # Add noise to clean image: x_noisy = sqrt(a) * x_gt + sqrt(1-a) * noise
#     x_noisy = x_gt * a.sqrt() + e * (1.0 - a).sqrt()
    
#     # Broadcast embeddings to spatial dimensions
#     anatomy_spatial = anatomy_emb.view(B, 128, 1, 1).expand(-1, -1, H, W)
#     contrast_spatial = contrast_emb.view(B, 128, 1, 1).expand(-1, -1, H, W)
    
#     # Concatenate: [B, 257, H, W] = [B, 128+128+1, H, W]
#     conditioned_input = torch.cat([
#         anatomy_spatial,   # [B, 128, H, W]
#         contrast_spatial,  # [B, 128, H, W]
#         x_noisy           # [B, 1, H, W]
#     ], dim=1)
    
#     # Predict noise
#     output = model(conditioned_input, t.float())
    
#     # Loss
#     if keepdim:
#         return (e - output).square().sum(dim=(1, 2, 3))
#     else:
#         return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


# ==============================================================================
# MODIFIED SAMPLING (WITH ANATOMY + CONTRAST CONDITIONING)
# ==============================================================================

# def sample_conditioned_fast_ddpm(
#     model,
#     x_source_image,
#     anatomy_emb,
#     contrast_emb,
#     config,
#     betas,
#     timesteps=10,
#     scheduler_type='non-uniform',
#     device='cuda',
#     return_x0=False  # ← Already has this parameter
# ):
#     """Sample in PIXEL space with anatomy + contrast conditioning."""
#     model.eval()
#     with torch.no_grad():
#         B = anatomy_emb.shape[0]
#         H, W = config.data.image_size, config.data.image_size

#         edge_map = sobel_edge_map(x_source_image).detach()
#         x = torch.randn(B, 1, H, W, device=device)


#         if scheduler_type == 'non-uniform':
#             if timesteps == 5:
#                 seq = [0, 199, 499, 799, 999]
#             elif timesteps == 10:
#                 seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]
#             else:
#                 # Fallback for other timesteps
#                 num_1 = int(timesteps * 0.4)
#                 num_2 = int(timesteps * 0.6)
#                 stage_1 = np.linspace(0, 699, num_1+1)[:-1]
#                 stage_2 = np.linspace(699, 999, num_2)
#                 stage_1 = np.ceil(stage_1).astype(int)
#                 stage_2 = np.ceil(stage_2).astype(int)
#                 seq = np.concatenate((stage_1, stage_2)).tolist()
        
#         seq_next = [-1] + list(seq[:-1])
#         xs = [x]
#         x0_t = None  # ← ADD THIS to track x0
        
#         for i, j in zip(reversed(seq), reversed(seq_next)):
#             t = torch.ones(B, device=device).long() * i
#             next_t = torch.ones(B, device=device).long() * j
            
#             a_cum = (1 - betas).cumprod(dim=0)
#             t = t.clamp(0, len(a_cum)-1).long()
#             next_t = next_t.clamp(-1, len(a_cum)-1).long()

#             at = a_cum.index_select(0, t)
#             at_next = a_cum.index_select(0, next_t.clamp(min=0))
            
#             xt = xs[-1].to(device)
            
#             cond = torch.cat([anatomy_emb, contrast_emb], dim=1)
#             model_input = torch.cat([xt, edge_map], dim=1)
#             et = model(model_input, t.float(), cond=cond)
            
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # ← SAVE x0
#             c1 = 0.0
#             c2 = ((1 - at_next) - c1 ** 2).sqrt()
#             xt_next = at_next.sqrt() * x0_t + c2 * et
            
#             xs.append(xt_next.to('cpu'))
        
#         # ← CHANGE THIS RETURN STATEMENT
#         if return_x0:
#             return x0_t.to(device)  # Return clean prediction
#         else:
#             return xs[-1].to(device)  # Return noisy sample

def sample_conditioned_fast_ddpm(
    model,
    x_source_image,
    anatomy_emb,
    contrast_emb,
    config,
    betas,
    timesteps=10,
    scheduler_type='non-uniform',
    device='cuda',
    eta=0.0  # ← ADD eta parameter (default 0.0 like doc2)
):
    """Sample in PIXEL space with anatomy + contrast conditioning."""
    model.eval()
    with torch.no_grad():
        B = anatomy_emb.shape[0]
        H, W = config.data.image_size, config.data.image_size

        # edge_map = sobel_edge_map(x_source_image).detach()
        x = torch.randn(B, 1, H, W, device=device)
        edge_map = sobel_edge_map(x_source_image).detach()

        if scheduler_type == 'non-uniform':
            if timesteps == 5:
                seq = [0, 199, 499, 799, 999]
            elif timesteps == 10:
                seq = [0, 199, 399, 599, 699, 799, 849, 899, 949, 999]
            else:
                # Fallback for other timesteps
                num_1 = int(timesteps * 0.4)
                num_2 = int(timesteps * 0.6)
                stage_1 = np.linspace(0, 699, num_1+1)[:-1]
                stage_2 = np.linspace(699, 999, num_2)
                stage_1 = np.ceil(stage_1).astype(int)
                stage_2 = np.ceil(stage_2).astype(int)
                seq = np.concatenate((stage_1, stage_2)).tolist()
        
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []  # ← ADD x0_preds tracking like doc2
        
        # ← CHANGE: Use compute_alpha like doc2 (simpler indexing)
        beta_with_zero = torch.cat([torch.zeros(1).to(device), betas], dim=0)
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.ones(B, device=device).long() * i
            next_t = torch.ones(B, device=device).long() * j
            
            # ← CHANGE: Use compute_alpha style from doc2
            at = (1 - beta_with_zero).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
            at_next = (1 - beta_with_zero).cumprod(dim=0).index_select(0, next_t + 1).view(-1, 1, 1, 1)
            
            xt = xs[-1].to(device)
            
            cond = torch.cat([anatomy_emb, contrast_emb], dim=1)
            model_input = torch.cat([xt, edge_map], dim=1)
            # model_input = xt
            et = model(model_input, t.float(), cond=cond)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))  # ← ADD: Track x0 predictions
            
            # ← CHANGE: Use DDIM equation (12) with eta like doc2
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            # ← CHANGE: Add c1 * randn term (even though eta=0, keeps structure consistent)
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et
            
            xs.append(xt_next.to('cpu'))
        
        # ← CHANGE: Return tuple like doc2's sg_generalized_steps
        return xs, x0_preds



# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def train_phase2():
    """Main training loop (your style)"""
    
    print("="*80)
    print("PHASE 2: FAST-DDPM WITH ANATOMY + CONTRAST CONDITIONING")
    print("="*80)

    
    # 1. Load Phase 1 model (FROZEN)
    phase1_model, phase1_config = load_phase1_model(phase1_checkpoint, DEVICE)
    
    # 2. Create dataloaders
    print("\nCreating dataloaders...")
    from train_dataloader import create_contrastive_dataloaders
    train_loader, val_loader = create_contrastive_dataloaders(
        train_root=train_data_root,
        val_root=val_data_root,
        modalities=modalities,
        num_patients_per_batch=batch_size,
        num_slices_per_modality=1,
        target_size=(256, 256),
        num_workers=8,
        apply_mask=True
    )
    
    # 3. Create Fast-DDPM model
    print("Building Fast-DDPM model...")
    ddpm_config = ConditionedFastDDPMConfig()
    model = DiffusionModel(ddpm_config).to(DEVICE)
    model = torch.nn.DataParallel(model)
    
    # 4. Setup optimizer
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # NEW: Separate learning rates
    film_params = []
    base_params = []

    for name, param in model.named_parameters():
        if 'cond_scale' in name or 'cond_shift' in name:
            film_params.append(param)
            print(f"FiLM param: {name}")  # Debug: see which params
        else:
            base_params.append(param)

    print(f"\nFiLM params: {len(film_params)}")
    print(f"Base params: {len(base_params)}\n")

    optimizer = optim.Adam([
        {'params': base_params, 'lr': learning_rate},           # 2e-4
        {'params': film_params, 'lr': learning_rate * 10}       # 2e-3 (10× higher!)
    ], lr=learning_rate)


    # 5. Setup beta schedule
    betas = get_beta_schedule(
        beta_schedule=ddpm_config.diffusion.beta_schedule,
        beta_start=ddpm_config.diffusion.beta_start,
        beta_end=ddpm_config.diffusion.beta_end,
        num_diffusion_timesteps=ddpm_config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(DEVICE)
    
    # 6. Setup EMA
    ema_helper = EMAHelper(mu=ddpm_config.model.ema_rate)
    ema_helper.register(model)


    
    # 7. Tracking best models
    best_models_psnr = deque(maxlen=3)
    best_models_ssim = deque(maxlen=3)


    # ==============================================================================
    # (OPTIONAL) LOAD PRETRAINED CHECKPOINT FOR FINE-TUNING
    # ==============================================================================

    resume_checkpoint = ""  # <- your last checkpoint
    if os.path.exists(resume_checkpoint):
        print(f"Loading pretrained weights from: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=DEVICE)
        
        # Restore model
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        # Restore optimizer (optional — if continuing seamlessly)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state restored.")
        
        # Restore EMA (if present)
        if 'ema_state_dict' in checkpoint:
            ema_helper.load_state_dict(checkpoint['ema_state_dict'])
            print("EMA state restored.")
        
        # Optionally continue iteration counter
        start_iter = checkpoint.get('iteration', 0)
        global_iteration = start_iter
        print(f"Resuming training from iteration {global_iteration}.\n")
    else:
        print("No checkpoint found. Starting from scratch.")
        global_iteration = 0

    
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Total iterations: {n_iters:,}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Diffusion steps (train): {num_diffusion_timesteps}")
    print(f"Diffusion steps (sample): {fast_ddpm_timesteps}")
    print(f"Snapshot frequency: {snapshot_freq:,}")
    print(f"Validation frequency: {validation_freq:,}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")
    
    # 8. Training loop
    # global_iteration = 0
    train_iter = iter(train_loader)
    
    pbar = tqdm(initial=global_iteration, total=n_iters, 
                desc="Phase 2 Training")
    
    while global_iteration < n_iters:
        model.train()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch['images'].to(DEVICE)           # [B, 1, 256, 256]
        modality_ids = batch['modality_id'].to(DEVICE)
        patient_ids = batch['patient_id'].to(DEVICE)
        
        # Transform to [-1,1]
        images = data_transform_ddpm(images)

        # Randomly pair source and target from SAME patient
        B = images.size(0)
        unique_patients = torch.unique(patient_ids)
        
        # For each patient, randomly select source and target modalities
        source_indices = []
        target_indices = []
        
        for patient_id in unique_patients:
            # Get all indices for this patient
            patient_mask = (patient_ids == patient_id)
            patient_indices = torch.where(patient_mask)[0]
            
            if len(patient_indices) < 2:
                continue  # Need at least 2 modalities
            
            # All pairs
            for i in range(len(patient_indices)):
                for j in range(len(patient_indices)):
                    if i != j:  # Different modalities
                        source_indices.append(patient_indices[i])
                        target_indices.append(patient_indices[j])
        
        if len(source_indices) == 0:
            continue
        
        source_indices = torch.tensor(source_indices, device=DEVICE)
        target_indices = torch.tensor(target_indices, device=DEVICE)
        
        # Get source and target images
        source_images = images[source_indices]  # e.g., T1 images
        target_images = images[target_indices]  # e.g., T2 images
        target_mod_ids = modality_ids[target_indices]
        
        # Extract anatomy from SOURCE
        with torch.no_grad():
            features = phase1_model.encoder(source_images)
            anatomy_emb = phase1_model.anatomy_head(features)  # [num_pairs, 128]
        
        # Get contrast for TARGET from bank
        contrast_code = phase1_model.contrast_bank(target_mod_ids)  # [num_pairs, 128]

        # Sample timesteps
        num_pairs = len(target_images)
        if scheduler_type == 'non-uniform':
            t_intervals = torch.tensor([0,199,399,599,699,799,849,899,949,999])
        
        idx_1 = torch.randint(0, len(t_intervals), size=(num_pairs // 2 + 1,))
        idx_2 = len(t_intervals) - idx_1 - 1
        idx = torch.cat([idx_1, idx_2], dim=0)[:num_pairs]
        t = t_intervals[idx].to(DEVICE)

        # Sample noise
        e = torch.randn_like(target_images)

        # # Cross-modality loss: SOURCE anatomy + TARGET contrast → TARGET image
        # loss = pixel_conditioned_diffusion_loss(
        #     model, target_images, source_images, anatomy_emb, contrast_code, t, e, betas
        # )

        # # Backward
        # optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # optimizer.step()
        # ema_helper.update(model)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward + backward
        with autocast(dtype=AMP_DTYPE):
            loss = pixel_conditioned_diffusion_loss(
                model, target_images, source_images, anatomy_emb, contrast_code, t, e, betas
            )

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer + scaler step
        scaler.step(optimizer)
        scaler.update()
        ema_helper.update(model)


        # Logging
        global_iteration += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- FiLM DIAGNOSTIC (CORRECTED) ---
        if global_iteration % 1000 == 0:
            with torch.no_grad():
                # Use current batch data for diagnostic
                test_img = source_images[:1]  # Take first source image
                test_mod_id = target_mod_ids[:1]  # Take first target modality
                
                # Get embeddings
                feat = phase1_model.encoder(test_img)
                anat = phase1_model.anatomy_head(feat)
                cont = phase1_model.contrast_bank(test_mod_id)
                cond_test = torch.cat([anat, cont], dim=1)
                
                # Create test input
                edge_test = sobel_edge_map(test_img)
                test_t = torch.tensor([500], device=DEVICE)  # Mid-timestep
                test_noise = torch.randn_like(test_img)
                
                # Add noise at t=500
                a_test = (1 - betas).cumprod(dim=0)[500]
                test_noisy = test_img * a_test.sqrt() + test_noise * (1.0 - a_test).sqrt()
                
                # test_input = torch.cat([test_noisy, edge_test], dim=1)
                test_input = test_noisy
                
                # Compare with/without conditioning
                with autocast(dtype=AMP_DTYPE):
                    pred_with_cond = model(test_input, test_t.float(), cond=cond_test)
                    pred_without_cond = model(test_input, test_t.float(), cond=None)
                
                diff = (pred_with_cond - pred_without_cond).abs().mean().item()
                pbar.write(f"   FiLM effect @ iter {global_iteration}: {diff:.6f}")
                
                # Check FiLM weight magnitude
                try:
                    first_resblock = model.module.down[0].block[0]
                    if hasattr(first_resblock, 'cond_scale') and first_resblock.cond_scale is not None:
                        gamma_std = first_resblock.cond_scale.weight.std().item()
                        beta_std = first_resblock.cond_shift.weight.std().item()
                        pbar.write(f"   FiLM weights - gamma: {gamma_std:.4f}, beta: {beta_std:.4f}")
                except Exception as e:
                    pbar.write(f"   Could not check FiLM weights: {e}")

                
                # === Check if gradients exist ===
                first_resblock = model.module.down[0].block[0]
                if first_resblock.cond_scale.weight.grad is not None:
                    gamma_grad_norm = first_resblock.cond_scale.weight.grad.norm().item()
                    beta_grad_norm = first_resblock.cond_shift.weight.grad.norm().item()
                    pbar.write(f"   FiLM grad norms - gamma: {gamma_grad_norm:.6f}, beta: {beta_grad_norm:.6f}")
                else:
                    pbar.write(f"   FiLM gradients are NONE!")
                
                # === Check embedding variance ===
                anatomy_mean = anatomy_emb.mean().item()
                anatomy_std = anatomy_emb.std().item()
                contrast_mean = contrast_code.mean().item()
                contrast_std = contrast_code.std().item()
                pbar.write(f"   Embeddings - anatomy: μ={anatomy_mean:.4f}, σ={anatomy_std:.4f}")
                pbar.write(f"              contrast: μ={contrast_mean:.4f}, σ={contrast_std:.4f}")
                
                # === Check if embeddings vary across samples ===
                if len(anatomy_emb) > 1:
                    # Pairwise cosine similarity
                    from torch.nn.functional import cosine_similarity
                    sim = cosine_similarity(anatomy_emb[0:1], anatomy_emb[1:2], dim=1)
                    pbar.write(f"   Anatomy similarity (sample 0 vs 1): {sim.item():.4f}")
        # --- END FiLM DIAGNOSTIC ---

        # Checkpoints
        if global_iteration % snapshot_freq == 0:
            ckpt = f"{base_path}/checkpoints/iter_{global_iteration:07d}.pth"
            save_checkpoint(model, optimizer, global_iteration, ckpt, ema_helper)
            pbar.write(f"✓ Checkpoint saved at iteration {global_iteration}")
        
        if global_iteration % validation_freq == 0:
            pbar.write(f"Generating visuals at iteration {global_iteration}...")
            model.eval()
            ema_helper.ema(model)

            os.makedirs(f"{base_path}/visuals", exist_ok=True)

            with torch.no_grad(), autocast(dtype=AMP_DTYPE):
                val_batch = next(iter(val_loader))
                val_images = val_batch['images'].to(DEVICE)
                val_mod_ids = val_batch['modality_id'].to(DEVICE)
                val_patient_ids = val_batch['patient_id'].to(DEVICE)

                # NEW: Cycle through different source modalities
                source_mod_id = (global_iteration // validation_freq) % len(modalities)
                source_mod_name = modalities[source_mod_id]
                
                source_mask = (val_mod_ids == source_mod_id)
                if source_mask.any():
                    source_img = val_images[source_mask][:1]
                    source_transformed = data_transform_ddpm(source_img)

                    # Extract anatomy
                    features = phase1_model.encoder(source_transformed)
                    anatomy_emb = phase1_model.anatomy_head(features)

                    # Create visualization with labels
                    from PIL import Image, ImageDraw, ImageFont
                    import torchvision.transforms as transforms
                    
                    labeled_images = []
                    
                    # Add source image with label
                    source_pil = transforms.ToPILImage()(source_img.squeeze(0).cpu())
                    source_pil = source_pil.convert('RGB')
                    draw = ImageDraw.Draw(source_pil)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    draw.text((10, 10), f"INPUT: {source_mod_name.upper()}", fill=(255, 255, 0), font=font)
                    labeled_images.append(transforms.ToTensor()(source_pil))

                    # Generate all modalities
                    for mod_id in range(len(modalities)):
                        target_mod_name = modalities[mod_id]
                        
                        # Get contrast from bank
                        contrast_emb = phase1_model.contrast_bank(
                            torch.tensor([mod_id], device=DEVICE)
                        )

                        # Generate
                        xs, x0_preds = sample_conditioned_fast_ddpm(
                        model, source_transformed, anatomy_emb, contrast_emb,
                        ddpm_config, betas, timesteps=fast_ddpm_timesteps,
                        scheduler_type=scheduler_type, device=DEVICE, eta=0.0
                    )
                        synthetic = xs[-1].to(DEVICE) 


                        synthetic_disp = inverse_data_transform_ddpm(synthetic)

                        # Get GT
                        gt_mask = (val_mod_ids == mod_id)
                        if gt_mask.any():
                            gt_img_disp = val_images[gt_mask][:1]
                        else:
                            gt_img_disp = torch.zeros_like(source_img)

                        # Add labels to GT
                        gt_pil = transforms.ToPILImage()(gt_img_disp.squeeze(0).cpu())
                        gt_pil = gt_pil.convert('RGB')
                        draw = ImageDraw.Draw(gt_pil)
                        draw.text((10, 10), f"GT: {target_mod_name.upper()}", fill=(0, 255, 0), font=font)
                        labeled_images.append(transforms.ToTensor()(gt_pil))

                        # Add labels to Synthetic
                        syn_pil = transforms.ToPILImage()(synthetic_disp.squeeze(0).cpu())
                        syn_pil = syn_pil.convert('RGB')
                        draw = ImageDraw.Draw(syn_pil)
                        draw.text((10, 10), f"SYN: {target_mod_name.upper()}", fill=(255, 0, 0), font=font)
                        labeled_images.append(transforms.ToTensor()(syn_pil))

                    # Stack horizontally
                    visual_grid = torch.cat(labeled_images, dim=2)  # Concatenate along width
                    save_path = f"{base_path}/visuals/iter{global_iteration:07d}_source_{source_mod_name}.png"
                    save_image(visual_grid, save_path, normalize=False)
                    pbar.write(f"Saved → {save_path} (source: {source_mod_name})")

            ema_helper.restore(model)
            model.train()

        
        # Validation
        if global_iteration % validation_freq == 0:
            pbar.write(f"Running validation at iteration {global_iteration}...")
            
            model.eval()
            ema_helper.ema(model)
            
            # Track metrics per source modality
            patient_metrics = {
                'PSNR': [],
                'SSIM': [],
                'per_modality': {mod: {'PSNR': [], 'SSIM': []} for mod in modalities}
            }
            
            with torch.no_grad(), autocast(dtype=AMP_DTYPE):
                for val_idx, val_batch in enumerate(val_loader):
                    if val_idx >= 5:
                        break
                    
                    val_images = val_batch['images'].to(DEVICE)
                    val_mod_ids = val_batch['modality_id'].to(DEVICE)
                    val_patient_ids = val_batch['patient_id'].to(DEVICE)
                    
                    #Try each modality as source
                    unique_patients = torch.unique(val_patient_ids)
                    
                    for patient_id in unique_patients[:1]:  # Just first patient per batch
                        patient_mask = (val_patient_ids == patient_id)
                        patient_images = val_images[patient_mask]
                        patient_mod_ids = val_mod_ids[patient_mask]
                        
                        # Get one source modality (e.g., T1 = modality 0)
                        source_mod_id = 0  # Or cycle through all modalities
                        source_mask = (patient_mod_ids == source_mod_id)
                        
                        if not source_mask.any():
                            continue
                        
                        source_img = patient_images[source_mask][:1]
                        source_transformed = data_transform_ddpm(source_img)
                        
                        # Extract anatomy
                        features = phase1_model.encoder(source_transformed)
                        anatomy_emb = phase1_model.anatomy_head(features)
                        
                        # Generate all modalities
                        for target_mod_id in range(len(modalities)):
                            # Skip if generating same as source (optional)
                            # if target_mod_id == source_mod_id:
                            #     continue
                            
                            # Get contrast for target
                            contrast_emb = phase1_model.contrast_bank(
                                torch.tensor([target_mod_id], device=DEVICE)
                            )
                            
                            # Generate
                            xs, x0_preds = sample_conditioned_fast_ddpm(
                            model, source_transformed, anatomy_emb, contrast_emb,
                            ddpm_config, betas, timesteps=fast_ddpm_timesteps,
                            scheduler_type=scheduler_type, device=DEVICE, eta=0.0
                        )
                            synthetic = xs[-1].to(DEVICE)

                            
                            # Ensure [0, 1] range
                            synthetic = inverse_data_transform_ddpm(synthetic)
                            synthetic = torch.clamp(synthetic, 0.0, 1.0)
                            
                            # Get GT for this modality
                            gt_mask = (patient_mod_ids == target_mod_id)
                            if not gt_mask.any():
                                continue
                            
                            gt_img = patient_images[gt_mask][:1]
                            
                            # Ensure GT is also in [0, 1]
                            # (Your dataloader should already do this, but just in case)
                            gt_img = torch.clamp(gt_img, 0.0, 1.0)
                            
                            # Ensure same spatial size
                            if synthetic.shape != gt_img.shape:
                                synthetic = F.interpolate(
                                    synthetic,
                                    size=(gt_img.shape[2], gt_img.shape[3]),
                                    mode='bilinear',
                                    align_corners=False
                                )
                            
                            # Calculate metrics
                            try:
                                psnr_val = psnr(gt_img, synthetic).item()
                                ssim_val = ssim(gt_img, synthetic).item()
                                
                                # Sanity checks
                                if not (0 <= psnr_val <= 100):
                                    pbar.write(f"Warning: PSNR={psnr_val:.2f} out of range")
                                if not (0 <= ssim_val <= 1):
                                    pbar.write(f" Warning: SSIM={ssim_val:.4f} out of range")
                                
                                patient_metrics['PSNR'].append(psnr_val)
                                patient_metrics['SSIM'].append(ssim_val)
                                patient_metrics['per_modality'][modalities[target_mod_id]]['PSNR'].append(psnr_val)
                                patient_metrics['per_modality'][modalities[target_mod_id]]['SSIM'].append(ssim_val)
                                
                            except Exception as e:
                                pbar.write(f" Metric calculation error: {e}")
                                pbar.write(f"   GT range: [{gt_img.min():.3f}, {gt_img.max():.3f}]")
                                pbar.write(f"   Synthetic range: [{synthetic.min():.3f}, {synthetic.max():.3f}]")
                
                # Compute averages
                if len(patient_metrics['PSNR']) > 0:
                    avg_psnr = np.mean(patient_metrics['PSNR'])
                    avg_ssim = np.mean(patient_metrics['SSIM'])
                    
                    pbar.write(f" Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
                    
                    # Per-modality breakdown
                    pbar.write("   Per-modality metrics:")
                    for mod_name in modalities:
                        if len(patient_metrics['per_modality'][mod_name]['PSNR']) > 0:
                            mod_psnr = np.mean(patient_metrics['per_modality'][mod_name]['PSNR'])
                            mod_ssim = np.mean(patient_metrics['per_modality'][mod_name]['SSIM'])
                            pbar.write(f"     {mod_name.upper()}: PSNR={mod_psnr:.2f}, SSIM={mod_ssim:.4f}")
            
                
                # Save best models
                model_path = f"{base_path}/model_weights"
                os.makedirs(f"{model_path}/psnr", exist_ok=True)
                os.makedirs(f"{model_path}/ssim", exist_ok=True)
                
                best_models_psnr.append((avg_psnr, f"{model_path}/psnr/iter_{global_iteration}.pth"))
                best_models_ssim.append((avg_ssim, f"{model_path}/ssim/iter_{global_iteration}.pth"))
                
                best_models_psnr = deque(sorted(best_models_psnr, reverse=True)[:3], maxlen=3)
                best_models_ssim = deque(sorted(best_models_ssim, reverse=True)[:3], maxlen=3)
                
                # Save if in top 3
                if (avg_psnr, f"{model_path}/psnr/iter_{global_iteration}.pth") in best_models_psnr:
                    save_checkpoint(model, optimizer, global_iteration, 
                                    f"{model_path}/psnr/iter_{global_iteration}.pth", ema_helper)
                
                if (avg_ssim, f"{model_path}/ssim/iter_{global_iteration}.pth") in best_models_ssim:
                    save_checkpoint(model, optimizer, global_iteration,
                                    f"{model_path}/ssim/iter_{global_iteration}.pth", ema_helper)
            
            # Restore training weights
            ema_helper.restore(model)


    
    pbar.close()
    print("\n" + "="*80)
    print("PHASE 2 TRAINING COMPLETED!")
    print("="*80)
    print(f"Total iterations: {global_iteration:,}")
    if len(best_models_psnr) > 0:
        print(f"Best PSNR: {best_models_psnr[0][0]:.4f}")
    if len(best_models_ssim) > 0:
        print(f"Best SSIM: {best_models_ssim[0][0]:.4f}")
    print("="*80)


if __name__ == "__main__":
    train_phase2()