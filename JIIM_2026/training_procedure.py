# import data loader

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import pandas as pd
from collections import deque
import pickle
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pandas as pd
from training_hyperparameters import *
from splits_only_val import SPLITS
# from splits_only_test import SPLITS  #for testing only
from train_dataloader import train_dataset
from test_dataloader import test_dataset
from ESAU_net import ESAU
from residual_transformers import ResViT
from transformer_configs import get_resvit_b16_config, get_resvit_l16_config
from CGNet_arch import CascadedGaze
from training_helpers import post_processing_order
from tqdm import tqdm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision.utils import make_grid, save_image
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
import json

if wand_db_boolean:
    import wandb

DEVICE = HP['DEVICE']
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
'''
CT will always be the first argument returnned by the dataloader

'''

base_path = f"sessions/{EXPERIMENT_NAME}"

class SimpleDiscriminator(nn.Module):
    def __init__(self, spatial_dims=2, num_in_ch=1, num_feat=64, input_size=[256, 256]):
        super(SimpleDiscriminator, self).__init__()
        self.spatial_dims = spatial_dims
        self.input_size = input_size
        
        conv_module = nn.Conv2d
        norm_module = nn.LayerNorm
        
        # Conv blocks
        self.convs = nn.Sequential(
            # First conv layer
            conv_module(num_in_ch, num_feat, 3, 1, 1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            
            # Conv with stride=2
            conv_module(num_feat, num_feat, 3, 2, 1, bias=False),  # /2
            nn.BatchNorm2d(num_feat), 
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            
            # Conv-stride 1
            conv_module(num_feat, num_feat*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_feat*2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            
            conv_module(num_feat*2, num_feat*2, 3, 2, 1, bias=False),  # /4
            nn.BatchNorm2d(num_feat*2),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            
            # Conv-stride 2
            conv_module(num_feat*2, num_feat*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_feat*4),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            
            conv_module(num_feat*4, num_feat*4, 3, 2, 1, bias=False),  # /8
            nn.BatchNorm2d(num_feat*4),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            
            # Conv-stride 3
            conv_module(num_feat*4, num_feat*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_feat*8),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            
            conv_module(num_feat*8, num_feat*8, 3, 2, 1, bias=False),  # /16
            nn.BatchNorm2d(num_feat*8),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )
        
        # Add final 1x1 conv to get patch predictions
        self.final_conv = conv_module(num_feat*8, 1, kernel_size=1)
    
    def forward(self, x):
        if isinstance(x, dict):
            x = x['level_0']
        feat2d = self.convs(x)
        out = self.final_conv(feat2d)  # Output shape: [B, 1, 16, 16]
        return out


# Boilerplate model loading and saving for GAN
def save_checkpoint_gan(generator, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d, epoch, path):
    if '/' in path:
        name = path.split('/')[-1]
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'scheduler_g_state_dict': scheduler_g.state_dict(),
        'scheduler_d_state_dict': scheduler_d.state_dict()
    }, path)

def load_checkpoint_gan(generator, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d, checkpoint, device=None):
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint.get('generator_state_dict', {}))
    else:
        ValueError("Check point does not contain key for generator_state_dict")
    
    if 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint.get('discriminator_state_dict', {}))
    else:
        ValueError("Check point does not contain key for discriminator_state_dict")
        
    if 'optimizer_g_state_dict' in checkpoint:
        optimizer_g.load_state_dict(checkpoint.get('optimizer_g_state_dict',{}))
    else:
        ValueError("Check point does not contain key for optimizer_g_state_dict")
        
    if 'optimizer_d_state_dict' in checkpoint:
        optimizer_d.load_state_dict(checkpoint.get('optimizer_d_state_dict',{}))
    else:
        ValueError("Check point does not contain key for optimizer_d_state_dict")
        
    if 'scheduler_g_state_dict' in checkpoint:
        scheduler_g.load_state_dict(checkpoint.get('scheduler_g_state_dict',{}))
    else:
        ValueError("Check point does not contain key for scheduler_g_state_dict")
        
    if 'scheduler_d_state_dict' in checkpoint:
        scheduler_d.load_state_dict(checkpoint.get('scheduler_d_state_dict',{}))
    else:
        ValueError("Check point does not contain key for scheduler_d_state_dict")
        
def model_restore_state(check_point_path):
    model_ckpt_name = check_point_path.split('/')[-1]
    epoch = model_ckpt_name.split('__')[0]
    splitd = model_ckpt_name.split('__')[1]
    print(epoch)
    print(splitd)
    
    return int(epoch.split("_")[1]), int(splitd.split("_")[1])

def load_checkpoint_eval_gan(generator, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    return generator

def convert_to_dict(x):
    if isinstance(x, torch.Tensor):
        return {'level_0': x}
    return x

def define_pretrained(model_name):
    if model_name == 'vgg19':
        pretrained = torchvision.models.vgg19(weights=torchvision.models.vgg.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet152':
        pretrained = torchvision.models.resnet152(weights=torchvision.models.resnet.ResNet152_Weights)
    return pretrained

class PerceptionLoss2D(nn.Module):
    def __init__(
        self,
        feature_extractor,
        loss_fn=nn.L1Loss(),
        channel_dim: int = 3,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        separate_channel: bool = True,
        base_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.channel_dim = channel_dim
        
        if normalize_mean is not None:
            c = len(normalize_mean)
            self.normalize_mean = torch.Tensor(normalize_mean).view(1, c, 1, 1)
            self.normalize_std = torch.Tensor(normalize_std).view(1, c, 1, 1)
        else:
            self.normalize_mean = normalize_mean
            self.normalize_std = normalize_std
        
        self.separate_channel = separate_channel
        self.base_weight = base_weight
    
    def forward(self, out, target, return_record=False):
        if isinstance(out, dict):
            out = out['level_0']
        if isinstance(target, dict):
            target = target['level_0']
            
        if self.normalize_mean is not None:
            device = out.device
            self.normalize_mean = self.normalize_mean.to(device)
            self.normalize_std = self.normalize_std.to(device)
            out = (out - self.normalize_mean) / (self.normalize_std + 1e-5)
            target = (target - self.normalize_mean) / (self.normalize_std + 1e-5)
        
        # Handle channel dimension for 2D images
        b, c, h, w = out.shape
        if c != self.channel_dim or self.separate_channel:
            # Convert grayscale to RGB by repeating channels
            if c == 1:
                out = out.repeat(1, self.channel_dim, 1, 1)
                target = target.repeat(1, self.channel_dim, 1, 1)
        
        # Extract features
        o_features = self.feature_extractor(out)
        t_features = self.feature_extractor(target)
        
        # Calculate perceptual loss
        loss = 0
        for key in o_features.keys():
            loss += self.loss_fn(o_features[key], t_features[key]) / self.base_weight
        
        loss_record = loss.item()
        
        if return_record:
            return loss, loss_record
        else:
            return loss
        
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
        # Create VGG19 feature extractor
        model_name = 'vgg19'
        return_nodes = ['features.35']
        
        pretrained = define_pretrained(model_name).eval()
        feature_extractor = create_feature_extractor(pretrained, return_nodes)
        feature_extractor.to(device)
        
        self.perceptual_loss = PerceptionLoss2D(
            feature_extractor=feature_extractor,
            loss_fn=nn.L1Loss(),
            channel_dim=3,
            separate_channel=True,
            base_weight=1.0 
        )
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = self.l1_weight * l1 + self.perceptual_weight * perceptual
        return total_loss

# Modified Hybrid Loss with Adversarial Training
class AdversarialHybridLoss(nn.Module):
    def __init__(self, beta=1.0, gamma=0.0001): 
        super().__init__()
        self.beta = beta    # Weight for regression (Combined L1 + Perceptual)
        self.gamma = gamma  # Weight for adversarial loss
        
        # Use the combined loss for regression (L1 + Perceptual)
        self.combined_loss = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1, device=DEVICE)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_continuous, target, discriminator_fake_output=None, mode='generator'):
        # Combined regression loss (L1 + Perceptual)
        regression_loss = self.combined_loss(pred_continuous, target)
        
        if mode == 'generator' and discriminator_fake_output is not None:
            # Adversarial loss - generator wants discriminator to classify fake as real
            real_labels = torch.ones_like(discriminator_fake_output)
            adversarial_loss = self.adversarial_loss(discriminator_fake_output, real_labels)
            
            return (self.beta * regression_loss + 
                   self.gamma * adversarial_loss)
        else:
            return self.beta * regression_loss

# Metrics calculation
def calculate_val_metrics(output, target):
    mse_loss = nn.MSELoss()(output, target)
    psnr = 10 * torch.log10(1 / mse_loss)
    ssim_value = ssim(output, target, data_range=1.0, size_average=True)
    return psnr.item(), ssim_value.item()


split_resolved, epoch_resolved = True, True
if continue_path:
    split_resolved, epoch_resolved = False, False

for key, splitc in SPLITS.items():
    config = get_resvit_b16_config()
    model = ESAU("Converter",in_channels=1,n_channels=HP['model_params']['num_channels'],out_channels=1).to(DEVICE)
    
    #Use during trsting form loading weights
    # model.load_state_dict(torch.load('/', map_location=DEVICE)['generator_state_dict'])
    
    # Create discriminator
    discriminator = SimpleDiscriminator(spatial_dims=2, num_in_ch=2, num_feat=64, input_size=[256, 256]).to(DEVICE)
    
    # Optimizers for both generator and discriminator
    optimizer_G = optim.Adam(model.parameters(), lr=HP['learning_rate'])
    optimizer_D = optim.Adam(discriminator.parameters(), lr=HP['learning_rate'] * 0.5)  # Slower learning for discriminator
    
    scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=10, gamma=0.5)

    criterion = AdversarialHybridLoss(beta=1.0, gamma=0.0001)
    
    best_models_psnr = deque(maxlen=3)
    best_models_ssim = deque(maxlen=3)
    
    if not split_resolved:
        e, spll = model_restore_state(continue_path)
        if key < spll:
            continue
        checkpoint = torch.load(continue_path, map_location=DEVICE)
        load_checkpoint_gan(model, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D, checkpoint)
                    
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
                        name = f"{EXPERIMENT_NAME} Split-{key}",
                        config= HP,
                        resume="allow"
                        )

    print("Processing SPLIT ---------- " , key," ---------- ")
    split_base_path = f"{base_path}/SPLIT_{key}"
    os.makedirs(split_base_path,exist_ok=True)
    trr = train_dataset(splitc['train'])
    train_dataloader = DataLoader(trr, batch_size=HP['batch_size'], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True,
                            prefetch_factor=4)

    print(f"Split {key} - Train: {len(splitc['train'])}, Validation: {len(splitc['validation'])}")

    for epoch in range(HP['epochs']):
        if not epoch_resolved:
            if epoch < e:
                continue
            epoch_resolved = True
        
        ######################################
        # Training phase with adversarial training
        model.train()
        discriminator.train()
        
        train_dataloader_tqdm = tqdm(train_dataloader, desc=f"Split {key} Training Epoch {epoch+1}/{HP['epochs']}", leave=True)
        
        total_g_loss = 0
        total_d_loss = 0
        
        for idx, (ct_batch, mr_batch, mask) in enumerate(train_dataloader_tqdm):
            ct_batch, mr_batch, mask = ct_batch.to(DEVICE), mr_batch.to(DEVICE), mask.to(DEVICE)
            ct_batch = ct_batch.unsqueeze(1)
            mr_batch = mr_batch.unsqueeze(1)
            mask = mask.unsqueeze(1)
            
            batch_size = ct_batch.size(0)
            binary_mask = (mask > 0.5).float()
            mr_batch_masked = mr_batch * binary_mask  # Fixed: assign to new variable
            
            # ===============================
            # Train Discriminator
            # ===============================
            optimizer_D.zero_grad()
            
            # Real images
            real_pair = torch.cat([ct_batch, mr_batch_masked], dim=1)  # [B, 2, H, W]
            real_output = discriminator(real_pair)
            real_labels = torch.ones_like(real_output)
            d_real_loss = criterion.adversarial_loss(real_output, real_labels)
            
            # Fake images
            mr_fake = model(ct_batch)
            mr_fake_masked = mr_fake * binary_mask
            
            fake_pair = torch.cat([ct_batch, mr_fake_masked.detach()], dim=1)  # [B, 2, H, W]
            fake_output = discriminator(fake_pair)
            fake_labels = torch.zeros_like(fake_output)
            d_fake_loss = criterion.adversarial_loss(fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            # ===============================
            # Train Generator
            # ===============================
            optimizer_G.zero_grad()
            
            # Get discriminator output for generator training
            fake_pair = torch.cat([ct_batch, mr_fake_masked], dim=1)  # [B, 2, H, W]
            fake_output = discriminator(fake_pair)
            
            # Generator loss (regression + adversarial)
            g_loss = criterion(mr_fake_masked, mr_batch_masked, fake_output, mode='generator')  # Fixed: variable names
            
            g_loss.backward()
            optimizer_G.step()
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            train_dataloader_tqdm.set_postfix(
                G_loss=g_loss.item(),
                D_loss=d_loss.item()
            )

        # Move schedulers outside the batch loop
        scheduler_G.step()
        scheduler_D.step()

        avg_g_loss = total_g_loss / len(train_dataloader)
        avg_d_loss = total_d_loss / len(train_dataloader)
        
        if wand_db_boolean:
            wandb.log({
                "epoch": epoch,
                "Generator_Loss": avg_g_loss,
                "Discriminator_Loss": avg_d_loss,
                'tag': "Training Loss",
                'step': epoch
            })
        
        ##########################################################################################
        
        model.eval()
        discriminator.eval()
        
        # Store per-patient metrics
        patient_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        
        validation_metrics_epoch = {
            'step' : epoch,
            'ValidationPSNR': 0,
            'ValidationSSIM': 0,
            'ValidationMSE': 0,
            'ValidationMAE': 0,
            'epoch' : epoch
            }
        
        with torch.no_grad():
            model_path_val = f"{split_base_path}/model_weights"
            os.makedirs(f"{model_path_val}/ssim", exist_ok=True)
            os.makedirs(f"{model_path_val}/psnr", exist_ok=True)
            val_index = 0
            splitc['validation'] = [ele for ele in splitc['validation'] if '.DS_Store' not in ele]
            validation_loop = tqdm(splitc['validation'], desc=f"Split {key} Validation Loop {val_index+1}/{len(splitc['validation'])}", leave=True)
            
            for val_scan_index, scan in enumerate(validation_loop):
                save_array_val = []
                combined_grid_val = []
                validation_DS = test_dataset(scan, HP['training_type'])
                validation_loader = DataLoader(validation_DS, batch_size=HP['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,
                                            prefetch_factor=2)
                scan_name_val = scan.split('/')[-1]
                
                # Per-patient accumulation
                patient_psnr_total = 0.0
                patient_ssim_total = 0.0
                patient_mse_total = 0.0
                patient_mae_total = 0.0
                patient_slice_count = 0
                
                for idx, (ct, mr, mask) in enumerate(validation_loader):
                    # CT2MR
                    source_real = ct.to(DEVICE).unsqueeze(1)
                    target_real = mr.to(DEVICE).unsqueeze(1)
                    mask = mask.to(DEVICE).unsqueeze(1) 
                    
                    # Create binary mask
                    binary_mask = (mask > 0.5).float()
                    target_real_masked = target_real * binary_mask
                    
                    target_fake = model(source_real)  # [B, 1, H, W]
                    
                    for fn in post_processing_order:
                        target_fake = fn(target_fake)
                        
                    target_size = (target_real.shape[2], target_real.shape[3])
                    target_fake = F.interpolate(target_fake, size=target_size, mode=HP['inference_interpolation_mode'], align_corners=HP['inference_interpolation_allign_cornors']).to(DEVICE)
                    
                    target_fake_masked = target_fake * binary_mask  # Apply mask to fake target
                    target_fake_masked = torch.clamp(target_fake_masked, 0, 1)
                    
                    # Accumulate for this patient only
                    batch_size = source_real.shape[0]
                    patient_psnr_total += psnr(target_real_masked, target_fake_masked).mean().item() * batch_size
                    patient_ssim_total += ssim(target_real_masked, target_fake_masked).mean().item() * batch_size
                    patient_mse_total += F.mse_loss(target_real_masked, target_fake_masked).item() * batch_size
                    patient_mae_total += F.l1_loss(target_real_masked, target_fake_masked).item() * batch_size
                    patient_slice_count += batch_size
                    
                    # For visualization, you can choose masked or unmasked versions
                    # Option 1: Use masked versions for grids (recommended for consistency)
                    grid_fake = make_grid(target_fake_masked, nrow=target_fake_masked.shape[0], padding=2)
                    grid_real = make_grid(target_real_masked, nrow=target_real_masked.shape[0], padding=2)
                    
                    # Option 2: If you want to show unmasked for visualization purposes
                    # grid_fake = make_grid(target_fake, nrow=target_fake.shape[0], padding=2)
                    # grid_real = make_grid(target_real, nrow=target_real.shape[0], padding=2)
                    
                    # For saved arrays, use masked versions for consistency with metrics
                    combined = torch.cat([target_real_masked.squeeze(1), target_fake_masked.squeeze(1)], dim=-1)
                    save_array_val.append(combined)
                    combined_grid_val.append(torch.cat((grid_real, grid_fake), dim=1))
                    val_index += 1

                # Calculate per-patient averages and store
                patient_metrics['PSNR'].append(patient_psnr_total / patient_slice_count)
                patient_metrics['SSIM'].append(patient_ssim_total / patient_slice_count)
                patient_metrics['MSE'].append(patient_mse_total / patient_slice_count)
                patient_metrics['MAE'].append(patient_mae_total / patient_slice_count)
                    
                eval_scanvv = torch.cat(save_array_val, dim=0) 
                eval_scanvv = eval_scanvv.cpu().numpy()
                os.makedirs(f"{split_base_path}/validation", exist_ok=True)
                if combined_grid_val:
                    max_height = max(grid.shape[1] for grid in combined_grid_val)
                    padded_grids = []
                    for grid in combined_grid_val:
                        if grid.shape[1] < max_height:
                            padding = max_height - grid.shape[1]
                            grid = F.pad(grid, (0, 0, 0, padding), mode='constant', value=0)
                        padded_grids.append(grid)
                    combined_grid_val = torch.cat(padded_grids, dim=2)
                save_image(combined_grid_val, f"{split_base_path}/validation/{scan_name_val}_{epoch}.png")
                if wand_db_boolean:
                    wandb.log({f"Validation Generation Split_{key}": wandb.Image(f"{split_base_path}/validation/{scan_name_val}_{epoch}.png"),
                                   "step" : epoch,
                                   'tag' : "Validation Image",
                                   'caption' : f"{scan_name_val}_{epoch}_{key} Validation",
                                    "epoch" : epoch,
                                })
                        
        # Calculate final averages across patients (not slices)
        validation_metrics_epoch['ValidationPSNR'] = np.mean(patient_metrics['PSNR'])
        validation_metrics_epoch['ValidationSSIM'] = np.mean(patient_metrics['SSIM'])
        validation_metrics_epoch['ValidationMSE'] = np.mean(patient_metrics['MSE'])
        validation_metrics_epoch['ValidationMAE'] = np.mean(patient_metrics['MAE'])
        
        fold_validation_results = {
        'fold': key,
        'epoch': epoch,
        'avg_psnr': validation_metrics_epoch['ValidationPSNR'],
        'avg_ssim': validation_metrics_epoch['ValidationSSIM'], 
        'avg_mse': validation_metrics_epoch['ValidationMSE'],
        'avg_mae': validation_metrics_epoch['ValidationMAE'],
        'num_val_patients': len(splitc['validation'])}
        os.makedirs(f"{split_base_path}/validation_averages", exist_ok=True)
        with open(f"{split_base_path}/validation_averages/fold_{key}_epoch_{epoch}.json", "w") as f:
            json.dump(fold_validation_results, f, indent=2)

        samples = 0 # Creating div by 0 in case of an error
        
        best_models_psnr.append((validation_metrics_epoch['ValidationPSNR'], f"{model_path_val}/psnr/epoch_{epoch}__split_{key}__.pth"))
        best_models_ssim.append((validation_metrics_epoch['ValidationSSIM'], f"{model_path_val}/ssim/epoch_{epoch}__split_{key}__.pth"))
        
        if wand_db_boolean:
            wandb.log(validation_metrics_epoch)
        
        # Resume logic
        best_models_psnr = deque(sorted(best_models_psnr, reverse=True)[:3], maxlen=3)
        best_models_ssim = deque(sorted(best_models_ssim, reverse=True)[:3], maxlen=3)
        
        top_models_pnsr = set(m[1] for m in best_models_psnr)
        top_models_ssim = set(m[1] for m in best_models_ssim)
        
        if f"{model_path_val}/psnr/epoch_{epoch}__split_{key}__.pth" in top_models_pnsr:
            save_checkpoint_gan(model, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D, epoch, f"{model_path_val}/psnr/epoch_{epoch}__split_{key}__.pth")
            with open(f"{model_path_val}/Top3PSNR.pkl","wb") as f:
                pickle.dump(best_models_psnr, f)
        
        if f"{model_path_val}/ssim/epoch_{epoch}__split_{key}__.pth" in top_models_ssim:
            save_checkpoint_gan(model, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D, epoch, f"{model_path_val}/ssim/epoch_{epoch}__split_{key}__.pth")
            with open(f"{model_path_val}/Top3SSIM.pkl", "wb") as f:
                pickle.dump(best_models_ssim, f)
            
        # Delete if bad performer.
        for file in os.listdir(f"{model_path_val}/psnr"):
            file_path = os.path.join(f"{model_path_val}/psnr", file)
            if file_path not in top_models_pnsr:
                os.remove(file_path)
                
        for file in os.listdir(f"{model_path_val}/ssim"):
            file_path = os.path.join(f"{model_path_val}/ssim", file)
            if file_path not in top_models_ssim:
                os.remove(file_path)
            
        df = pd.DataFrame.from_dict(dict(best_models_psnr), orient='index')

        df.to_csv(f"{model_path_val}/Top3PSNR.csv", index_label="Epoch")
        
        df = pd.DataFrame.from_dict(dict(best_models_ssim), orient='index')
        df.to_csv(f"{model_path_val}/Top3SSIM.csv", index_label="Epoch")
        
        # Split completion
        print(f'Split {key} training completed. Best PSNR: {best_models_psnr[0][0]:.4f}, Best SSIM: {best_models_ssim[0][0]:.4f}')

    if wand_db_boolean:
        wandb.log({
                "Final_Best_PSNR": best_models_psnr[0][0],
                "Final_Best_SSIM": best_models_ssim[0][0], 
                "Split": key,
                "tag": "Final Metrics"
            })
        run.finish()   

