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

overwrite = True

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

for reg_type in reg_types:
    model = VariationalRegClass('MRI', reg_type)
    for sample_rate in sample_rates:
        num_walks = round(sample_rate * height)
        subsampling_matrix_0 = horiz_rand_walk_mask(height, width, num_walks,
                                                  distr='centre_clustered', allowing_inter=True,
                                                  p=[0., 1., 0.], scale=5)
        subsampling_matrix_1 = horiz_rand_walk_mask(height, width, num_walks,
                                                    distr='centre_clustered', allowing_inter=True,
                                                    p=[0., 1., 0.], scale=10)
        subsampling_matrix_2 = horiz_rand_walk_mask(height, width, num_walks,
                                                    distr='centre_clustered', allowing_inter=True,
                                                    p=[0., 1., 0.], scale=15)
        subsampling_matrix_3 = horiz_rand_walk_mask(height, width, num_walks,
                                                    distr='centre_clustered', allowing_inter=True,
                                                    p=[0., 1., 0.], scale=20)

        folder = directory + '/Experiments/MRI_birmingham/subsampled_data/'

        if not os.path.isdir(folder):
            os.system('mkdir ' + folder)

        np.save(folder + "mask_clustered_scale" + str(5) + "_sample_rate" + str(round(sample_rate, 4)) + ".npy",
                subsampling_matrix_0)
        np.save(folder + "mask_clustered_scale" + str(10) + "_sample_rate" + str(round(sample_rate, 4)) + ".npy",
                subsampling_matrix_1)
        np.save(folder + "mask_clustered_scale" + str(15) + "_sample_rate" + str(round(sample_rate, 4)) + ".npy",
                subsampling_matrix_2)
        np.save(folder + "mask_clustered_scale" + str(20) + "_sample_rate" + str(round(sample_rate, 4)) + ".npy",
                subsampling_matrix_3)

        for reg_param in reg_params:

            folder_clustered = directory + '/Experiments/MRI_birmingham/'+ str(reg_type)+\
                       '_regularised_recons/clustered_sampling/sample_rate_' + str(round(sample_rate, 4)) + '/'

            if os.path.isfile(folder_clustered + "recon_array_scale_" + str(20)
                              + "_reg_param_" + str(reg_param) + ".npy") and not overwrite:
                continue

            recons_clustered_0 = model.regularised_recons_from_subsampled_data(data, reg_param,
                                                           subsampling_arr=subsampling_matrix_0, niter=2000)
            recons_clustered_1 = model.regularised_recons_from_subsampled_data(data, reg_param,
                                                           subsampling_arr=subsampling_matrix_1, niter=2000)
            recons_clustered_2 = model.regularised_recons_from_subsampled_data(data, reg_param,
                                                           subsampling_arr=subsampling_matrix_2, niter=2000)
            recons_clustered_3 = model.regularised_recons_from_subsampled_data(data, reg_param,
                                                           subsampling_arr=subsampling_matrix_3, niter=2000)

            np.save(folder_clustered + "recon_array_scale_" + str(5) + "_reg_param_" + str(reg_param) + ".npy",
                    recons_clustered_0[0])
            np.save(folder_clustered + "recon_array_scale_" + str(10) + "_reg_param_" + str(reg_param) + ".npy",
                    recons_clustered_1[0])
            np.save(folder_clustered + "recon_array_scale_" + str(15) + "_reg_param_" + str(reg_param) + ".npy",
                    recons_clustered_2[0])
            np.save(folder_clustered + "recon_array_scale_" + str(20) + "_reg_param_" + str(reg_param) + ".npy",
                    recons_clustered_3[0])


            plt.figure()
            plt.imshow(np.abs(recons_clustered_0[0]), cmap=plt.cm.gray)
            plt.axis("off")
            plt.savefig(folder_clustered + "recon_scale_" +str(5) +"_reg_param_" + str(reg_param) + ".png")
            plt.close()

            plt.figure()
            plt.imshow(np.abs(recons_clustered_1[0]), cmap=plt.cm.gray)
            plt.axis("off")
            plt.savefig(folder_clustered + "recon_scale_" +str(10) +"_reg_param_" + str(reg_param) + ".png")
            plt.close()

            plt.figure()
            plt.imshow(np.abs(recons_clustered_2[0]), cmap=plt.cm.gray)
            plt.axis("off")
            plt.savefig(folder_clustered + "recon_scale_" + str(15) + "_reg_param_" + str(reg_param) + ".png")
            plt.close()

            plt.figure()
            plt.imshow(np.abs(recons_clustered_3[0]), cmap=plt.cm.gray)
            plt.axis("off")
            plt.savefig(folder_clustered + "recon_scale_" + str(20) + "_reg_param_" + str(reg_param) + ".png")
            plt.close()



