import cv2
import numpy as np

full_dataset = True # False Limits to 5 totals scans
all_slices = True # False Limits to 10 slides per patient
data_directory = '/'
destination_directory = '/'
per_sample_normalize = True

# NEW PARAMETERS: Number of slices to exclude from start and end
exclude_start_slices = 0  # Number of slices to skip from the beginning
exclude_end_slices = 0   # Number of slices to skip from the end

view = {
'axial' : True,
'coronal' : False,
'saggital' : False
}
slice_size = (256,256)
interpolation_method = cv2.INTER_LINEAR
padding_size = 608

# Do not edit
view_index_hh = {
'axial' : 2,
'coronal' : 1,
'saggital' : 0
}
print([n for m, n in view.items()])