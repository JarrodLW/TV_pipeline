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

overwrite = False

directory = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday'
data_path = directory + '/Data/02-20_MRI_Melanie_Britton/7/fid'
data_vec = np.fromfile(data_path, dtype=np.int32)

data = recasting_fourier_as_complex(data_vec, 128, 256)
data /= np.amax(np.abs(data))

height, width = data.shape

reg_types = ['TV', 'TGV']
sample_rates = [0.1*(a+1) for a in range(10)]
reg_params = [10**(-a) for a in range(10)]
#scales = [5, 10, 15, 20] # this gives the spread of the clustered mask

for reg_param in reg_params:
    model = VariationalRegClass('MRI', reg_type)
    for sample_rate in sample_rates:
        subsampling_matrix_bernoulli = bernoulli_mask(height, width, expected_sparsity=sample_rate)[0]

        folder = directory + '/Experiments/MRI_birmingham/subsampled_data/'

        if not os.path.isdir(folder):
            os.system('mkdir ' + folder)

        np.save(folder + "mask_Bernoulli_sample_rate" + str(round(sample_rate, 4)) + ".npy",
                subsampling_matrix_bernoulli)

        for reg_type in reg_types:

            folder_Bernoulli = directory + '/Experiments/MRI_birmingham/'+ str(reg_type)+\
                       '_regularised_recons/Bernoulli_sampling/sample_rate_' + str(round(sample_rate, 4)) +'/'

            if not os.path.isdir(folder_Bernoulli):
                os.system('mkdir ' + folder_Bernoulli)

            if os.path.isfile(folder_Bernoulli + 'recon_reg_param' + str(reg_param) + '.png') and not overwrite:
                continue

            recons_bernoulli = model.regularised_recons_from_subsampled_data(data, reg_param,
                                                           subsampling_arr=subsampling_matrix_bernoulli, niter=500)

            np.save(folder_Bernoulli + "recon_array_reg_param_" + str(reg_param) + ".npy", recons_bernoulli[0])

            plt.figure()
            plt.imshow(np.abs(recons_bernoulli[0]), cmap=plt.cm.gray)
            plt.axis("off")
            plt.savefig(folder_Bernoulli + "recon_reg_param_" + str(reg_param) + ".png")
            plt.close()

