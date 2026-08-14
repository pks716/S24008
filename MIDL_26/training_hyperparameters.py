import cv2
import subprocess
session_name = subprocess.check_output(["tmux", "display-message", "-p", "#S"], text=True).strip()

wand_db_boolean = False
# PROJECT = "CT2_MR_Pelvis_only_T2W"
PROJECT = "fast_ldm_sr_acdc"
EXPERIMENT_NAME = 'fast_ldm_acdc_sr' # Changed to indicate Fast-DDPM
continue_path = ""

data_directory = "/home/pks/Desktop/Peeyush/cardiac_work/diffuison_work/acdc/Data/slices_acdc"

evaluating_run_boolean = True # Keep False while training

HP = {
    'DEVICE': 'cuda:0',
    'model_params': {
        'type': 'fast-ldm_acdc_sr',
        'num_channels': 128  # Changed to their exact setting
    },
    'System': "Ruby",
    'TMUX': session_name,

    "batch_size": 8,              
    "learning_rate": 0.0001,      
    "n_iters": 1000000,           
    "snapshot_freq": 100,       
    "validation_freq": 100, 
    
    # Fast-DDPM specific parameters 
    'fast_ddpm_timesteps': 10,     
    'scheduler_type': 'non-uniform',   # 'uniform' or 'non-uniform'
    'diffusion_beta_start': 0.0001,
    'diffusion_beta_end': 0.02,
    'diffusion_num_timesteps': 1000,
    'diffusion_beta_schedule': 'linear',
    'use_lora': False,
    
    # Optimizer settings
    'optimizer': 'Adam',
    'weight_decay': 0.000,         
    'beta1': 0.9,                 
    'amsgrad': False,              
    'eps': 0.00000001,             
    
    # EMA settings 
    'use_ema': True,
    'ema_rate': 0.999,             
    
    'loss_weights': {},
    'inference_interpolation_mode': 'bilinear',
    'inference_interpolation_allign_cornors': False,
    'training_type': 'CT2MR'
}

helper_parameters = {
    'align_corners': True
}

# Fast-DDPM Model Configuration
FAST_DDPM_CONFIG = {
    'model': {
        'type': 'sg',                    
        'in_channels': 8,                
        'out_ch': 4,                     
        'ch': 128,                       
        'ch_mult': [1, 1, 2, 2, 4, 4],  
        'num_res_blocks': 2,             
        'attn_resolutions': [16],       
        'dropout': 0.0,                 
        'var_type': 'fixedsmall',        
        'ema_rate': 0.999,               
        'ema': True,                     
        'resamp_with_conv': True         
    },
    'data': {
        'dataset': 'BRATS',              
        'image_size': 32,               
        'channels': 4,                   
        'logit_transform': False,        
        'uniform_dequantization': False, 
        'gaussian_dequantization': False,
        'random_flip': False,             
        'rescaled': True,                #(rescale to [-1, 1])
        'num_workers': 8                 
    },
    'diffusion': {
        'beta_schedule': 'linear',       # their exact setting
        'beta_start': 0.0001,            # their exact setting
        'beta_end': 0.02,                # their exact setting
        'num_diffusion_timesteps': 1000  # their exact setting
    }
}
