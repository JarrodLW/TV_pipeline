#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:17:07 2020

@author: jlw31
"""
from processing import *
from Utils import *
import os
import matplotlib.pyplot as plt
from skimage import io

overwrite = True

directory = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday'
#data_path = directory + '/Data/04-20_CT_Paul_Quinn/phase/sino_cleaned/sino_0050.tif'
data_path_0 = directory + '/Data/04-20_CT_Paul_Quinn/phase/sino_cleaned/sino_0049.tif'
data_path_1 = directory + '/Data/04-20_CT_Paul_Quinn/phase/sino_cleaned/sino_0050.tif'
data_path_2 = directory + '/Data/04-20_CT_Paul_Quinn/phase/sino_cleaned/sino_0051.tif'

data_0 = np.array(io.imread(data_path_0), dtype=float)
data_1 = np.array(io.imread(data_path_1), dtype=float)
data_2 = np.array(io.imread(data_path_2), dtype=float)

data = np.zeros((3, *data_1.shape))
data[0, :, :] = data_0
data[1, :, :] = data_1
data[2, :, :] = data_2
#data /= np.amax(np.abs(data)) # maybe not te correct way to normalise...
#data = (data - np.amin(data))/(np.amax(data) - np.amin(data))


_, height, width = data.shape

reg_types = ['TV', 'TGV']
sample_rates = [0.1 * (a + 1) for a in range(10)]
reg_params = [10 ** (-a) for a in range(10)]

for reg_type in reg_types:

    model = VariationalRegClass('CT', reg_type)

    for sample_rate in sample_rates:
        num_walks = round(sample_rate*height)
        subsampling_matrix = horiz_rand_walk_mask(height, width, num_walks, allowing_inter=True, p=[0, 1., 0.])[0]

        for reg_param in reg_params:

            folder = directory + '/Experiments/CT_diamond/' + str(reg_type)+'_regularised_recons/recon_sample_rate_' \
                     + str(sample_rate) + "_reg_param_" + str(reg_param)
            #filename = +folder
            if os.path.isfile(folder + '/recon.png') and not overwrite:
                continue

            if not os.path.isdir(folder):
                os.system('mkdir '+ folder)

            recons = model.regularised_recons_from_subsampled_data(data, reg_param, subsampling_arr=subsampling_matrix,
                                                                   recon_dims=(167, 167), niter=500, a_offset=0,
                                                                   a_range=2*np.pi, d_offset=0, d_width=40)

            recon_numbers = ['0049', '0050', '0051']

            for i, recon_number in enumerate(recon_numbers):

                np.save(folder + '/recon_array_'+recon_number+'.npy', recons[i])

                plt.figure()
                plt.imshow(circle_mask(167, 0.95)*recons[i], cmap=plt.cm.gray)
                plt.axis("off")
                plt.savefig(folder + '/masked_recon_' + recon_number + '.png')
                plt.close()

                plt.figure()
                plt.imshow(recons[i], cmap=plt.cm.gray)
                plt.colorbar()
                plt.savefig(folder + '/recon_' + recon_number + '.png')
                plt.close()

