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
from skimage import io

overwrite = True

directory = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday'
#data_path = directory + '/Data/04-20_CT_Paul_Quinn/phase/sino_cleaned/sino_0050.tif'
data_path = directory + '/Experiments/CT_diamond/sino_0050_cleaned.tif'

data = np.array(io.imread(data_path), dtype=float)
#data /= np.amax(np.abs(data)) # maybe not te correct way to normalise...
data = (data - np.amin(data))/(np.amax(data) - np.amin(data))


height, width = data.shape

reg_types = ['TV', 'TGV']
#sample_rates = [0.1 * (a + 1) for a in range(1)]
sample_rates = [1.]
#reg_params = [10 ** (-a) for a in range(1)]
reg_params = [0.0001]

for sample_rate in sample_rates:
    num_walks = round(sample_rate*height)
    subsampling_matrix = horiz_rand_walk_mask(height, width, num_walks, allowing_inter=True, p=[0, 1., 0.])[0]
    for reg_type in reg_types:
        for reg_param in reg_params:

            filename = directory + '/Experiments/CT_diamond/recon_sample_rate_' + str(sample_rate) \
                       + reg_type + "reg_param_" + str(reg_param) + '.png'
            if os.path.isfile(filename) and not overwrite:
                continue

            recons = regularised_recons_from_subsampled_data(data, 'CT',
                                            reg_type, reg_param, subsampling_arr=subsampling_matrix,
                                                             recon_dims=(167, 167),
                                            niter=100, a_offset=0, a_range=2*np.pi,
                                            d_offset=0, d_width=40)


            masked_recon = circle_mask(167, 0.95)*recons[0]
            plt.figure()
            plt.imshow(masked_recon, cmap=plt.cm.gray)
            plt.axis("off")
            plt.savefig(filename)
            plt.close()
