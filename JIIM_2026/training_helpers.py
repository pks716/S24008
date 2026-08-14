import cv2  
import numpy as np
from training_hyperparameters import helper_parameters

"""
Define Preprocessing steps here. Write numpy slice funtions. Expected function type:

def f_name(slice):
    BOC
    return processed_slice
"""

def place_holder_fn(ele):
    return ele


def rescale_to_float16(slicell):
    # By default this saves the slices as float64, which is not needed.
    return slicell.astype(np.float16)

def normalize(tensor_block):
    return tensor_block

def de_normalize(tensor_block):
    return tensor_block

# These methods will be executed from left to right
# place it as an ordered pair with arguments
# Do not remove place holder

post_processing_order = [de_normalize, place_holder_fn]

# pre_processing_order.append((rescale_to_float16,['ct', 'mri', 'mask']))

# for method, targets in pre_processing_order:
#     for ele in targets:
#         if ele not in ['ct', 'mri', 'mask']:
#             raise ValueError('Incorrect modalities specified in pre_processing_order, pre_processing.py')