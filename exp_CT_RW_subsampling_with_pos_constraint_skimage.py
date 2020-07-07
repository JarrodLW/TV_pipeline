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
import datetime as dt

overwrite = False

#logfile = 'experiment_output_'+dt.datetime.now().isoformat()+'.txt'
#open(logfile, 'a').write('Starting experiment at: ' +dt.datetime.now().isoformat()+'\n')

directory = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday'
data_path_0 = directory + '/Data/04-20_CT_Paul_Quinn/phase/sino_cleaned/sino_0049.tif'
data_path_1 = directory + '/Data/04-20_CT_Paul_Quinn/phase/sino_cleaned/sino_0050.tif'
data_path_2 = directory + '/Data/04-20_CT_Paul_Quinn/phase/sino_cleaned/sino_0051.tif'

#open(logfile, 'a').write('Writing files to: '+directory+'\n')

data_0 = np.array(io.imread(data_path_0), dtype=float)
data_1 = np.array(io.imread(data_path_1), dtype=float)
data_2 = np.array(io.imread(data_path_2), dtype=float)

step = 1.5

list1 = ((np.arange(87830, 87862) - 87830) * step).tolist()
list2 = ((np.arange(87872, 87978) - 87872) * step + 58.5).tolist()
list3 = ((np.arange(87980, 88073) - 87872) * step + 58.5).tolist()

angle_list = list1 + list2 + list3

data_0, mask = pad_sino(data_0, step, 0, 240, angle_list)
data_1, _ = pad_sino(data_1, step, 0, 240, angle_list)
data_2, _ = pad_sino(data_2, step, 0, 240, angle_list)

data = np.zeros((3, *data_1.shape))
data[0, :, :] = data_0
data[1, :, :] = data_1
data[2, :, :] = data_2

_, height, width = data.shape

reg_types = ['TV']
#sample_rates = [0.1 * (a + 1) for a in range(10)]
#reg_params = [10 ** (-a) for a in range(10)]
sample_rates = [1.0]
reg_params = [0.]


for reg_type in reg_types:

    model = VariationalRegClass('CT', reg_type)

    for sample_rate in sample_rates:
        num_walks = round(sample_rate*height)
        subsampling_matrix = horiz_rand_walk_mask(height, width, num_walks, allowing_inter=True, p=[0, 1., 0.])[0]

        # indices at which we have measurements
        ind = np.nonzero((mask*subsampling_matrix)[:, 0])

        list_angle = (step * np.array(ind, dtype=float)) * np.pi / 180.0
        center = 83

        for reg_param in reg_params:

            folder = directory + '/Experiments/CT_diamond/' + str(reg_type)+'_regularised_recons_skimage/recon_sample_rate_' \
                     + str(round(sample_rate, 4)) + "_reg_param_" + str(reg_param)
            #filename = +folder

            recon_numbers = ['0049', '0050', '0051']

            if os.path.isfile(folder + '/recon_' + str(recon_numbers[-1]) + '_with_pos_constraint.png') and not overwrite:
                continue

            if not os.path.isdir(folder):
                os.system('mkdir '+ folder)

            recons = model.regularised_recons_from_subsampled_data(data, reg_param, subsampling_arr=mask*subsampling_matrix,
                                                                   recon_dims=(167, 167), niter=200, a_offset=0, enforce_positivity=True,
                                                                   a_range=2*np.pi, d_offset=0, d_width=40)

            for i, recon_number in enumerate(recon_numbers):

                np.save(folder + '/recon_array_'+recon_number+'_with_pos_constraint.npy', recons[i])

                recons_astra = recon_astra(data[i][ind], center, list_angle, 0.95, method="SIRT", num_iter=200)

                # plt.figure()
                # plt.imshow(circle_mask(167, 0.95)*recons[i], cmap=plt.cm.gray)
                # plt.axis("off")
                # plt.savefig(folder + '/masked_recon_' + recon_number + '_with_pos_constraint.png')
                # plt.close()
                #
                # plt.figure()
                # plt.imshow(recons[i], cmap=plt.cm.gray)
                # plt.colorbar()
                # plt.savefig(folder + '/recon_' + recon_number + '_with_pos_constraint.png')
                # plt.close()

                f = plt.figure()
                axarr = f.subplots(1, 3)
                im_0 = axarr[0].imshow(recons[i], cmap=plt.cm.gray)
                im_1 = axarr[1].imshow(circle_mask(167, 0.95)*recons[i], cmap=plt.cm.gray)
                im_2 = axarr[2].imshow(recon_astra, cmap=plt.cm.gray)
                axarr[0].axis("off")
                axarr[1].axis("off")
                axarr[2].axis("off")
                f.colorbar(im_0, ax=axarr[0])
                f.colorbar(im_1, ax=axarr[1])
                f.colorbar(im_2, ax=axarr[2])
                plt.savefig(folder + "recon_reg_param_" + recon_number + "_with_pos_constraint.png")
                plt.close()

# generating summary slides
#recon_numbers = ['0049', '0050', '0051']

#reg_types = ['TV']

for reg_type in reg_types:
    for recon_number in recon_numbers:
        for sample_rate in sample_rates:

            fig, axes = plt.subplots(5, 2)

            for j, reg_param in enumerate(reg_params):
                folder = directory + '/Experiments/CT_diamond/' + str(reg_type) + '_regularised_recons_skimage/recon_sample_rate_' \
                         + str(round(sample_rate, 4)) + "_reg_param_" + str(reg_param)

                recon = np.load(folder + '/recon_array_'+recon_number+'_with_pos_constraint.npy')

                axes[j//2, j % 2].imshow(circle_mask(167, 0.95)*recon, cmap=plt.cm.gray)
                axes[j//2, j % 2].axis('off')
                axes[j//2, j % 2].set_title('reg_param: '+str(reg_param))

            plt.tight_layout(w_pad=0.05)
            plt.savefig(directory + '/Experiments/CT_diamond/' + str(reg_type) + '_regularised_recons/Summary_slides/'
                                                                                 'recons_' + recon_number
                        + '_sample_rate_'+str(round(sample_rate, 4))+'_with_pos_constraint.png')
            plt.close()





