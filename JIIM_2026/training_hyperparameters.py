import cv2  

import subprocess

#Use Tmux for sessions
session_name = subprocess.check_output(["tmux", "display-message", "-p", "#S"], text=True).strip()


wand_db_boolean = False #for wandb logging

PROJECT = "CT2_MR_Pelvis"
EXPERIMENT_NAME = 'esau'
continue_path = ""


data_directory = '/'


evaluating_run_boolean = True 


HP = {
    'DEVICE' : 'cuda:0',

    'model_params':{
        'type': 'esau',
        'num_channels': 32
    },
    'System': "Ruby",
    'TMUX' : session_name,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "epochs": 100,
    'loss_weights' : {

    },
    'inference_interpolation_mode' : 'bilinear',
    'inference_interpolation_allign_cornors' : False,
    'training_type' : 'CT2MR' 
}

helper_parameters= {
    'align_corners' : True
}

