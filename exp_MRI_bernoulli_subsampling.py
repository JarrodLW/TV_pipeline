#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:17:07 2020

@author: jlw31
"""
from processing import regularised_recons_from_subsampled_data
from Utils import *
import os
import matplotlib.pyplot as plt

overwrite = False

directory = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday'
data_path = directory + '/Data/02-20_MRI_Melanie_Britton/7/fid'
data_vec = np.fromfile(data_path, dtype=np.int32)

data = recasting_fourier_as_complex(data_vec, 128, 256)
data /= np.amax(np.abs(data))

height, width = data.shape

reg_types = ['TV', 'TGV']
sample_rates = [0.1*(a+1) for a in range(10)]
#sample_rates = [1.]
reg_params = [10**(-a) for a in range(10)]
#reg_params = [0.0001]

for sample_rate in sample_rates:
    subsampling_matrix = bernoulli_mask(height, width, expected_sparsity=sample_rate)[0]
    for reg_type in reg_types:
        for reg_param in reg_params:
            
            filename = directory + '/Experiments/MRI_birmingham/hor_recon_sample_rate_' + str(sample_rate)\
                       + reg_type + "reg_param_" + str(reg_param) + '.png'
            if os.path.isfile(filename) and not overwrite:
                continue                
            
            recons = regularised_recons_from_subsampled_data(data, 'MRI',
                                                           reg_type, reg_param,
                                                           subsampling_arr=subsampling_matrix, niter=500)

            plt.figure()
            plt.imshow(np.abs(recons[0]), cmap=plt.cm.gray)
            plt.axis("off")
            plt.savefig(filename)
            plt.close()
