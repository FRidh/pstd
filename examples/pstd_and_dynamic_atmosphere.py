"""
Perform PSTD simulation using dynamic refractive index.
"""

import h5py
import numpy as np

from acoustics.pstd import PSTD as Model
from acoustics.pstd import Medium, PML

CFL = 0.3
soundspeed = 343.0
density = 1.296

filename = "../data/pstd_dynamic.hdf"
refractive_index_label = 'refractive_index'
output_label = 'output'

with h5py.File(filename, 'w') as datafile:
    
    
    refractive_index = datafile[refractive_index_label]
    medium = Medium(soundspeed, density, refractive_index)

    model = Model()

    output = datafile[output_label]
    receiver = Receiver()
    

