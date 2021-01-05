# Created: 07/12/2020.
# Based on noisy_Li_data_experiments_22102020_finer_hyperparam_search.py. Using the data dated 31/11/2020.

import numpy as np
import matplotlib.pyplot as plt
from processing import *
import json
import matplotlib.pyplot as plt
import os
import odl
import myOperators as ops
from Utils import *

dir = 'dTV/7Li_1H_MRI_Data_31112020/'

def unpacking_fourier_coeffs(arr):

    fourier_real_im = arr[:, 1:65]
    fourier_real_im = fourier_real_im[::2, :]

    fourier_real = fourier_real_im[:, 1::2]
    fourier_im = fourier_real_im[:, ::2]
    fourier = fourier_real + fourier_im * 1j

    return fourier

f_coeff_list = []

for i in range(2, 34):
    f_coeffs = np.reshape(np.fromfile(dir + 'Li2SO4/'+str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)

# f_coeffs_averaged = np.average(np.asarray(f_coeff_list), axis=0)
# my_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs_averaged)))
# image = np.abs(my_recon)

#reg_params = np.logspace(np.log10(2e3), np.log10(1e5), num=20)
reg_params = np.logspace(3., 4.5, num=20)
output_dims = [int(32), int(64)]
#output_dims = [int(32)]
Li_fourier_coeffs =f_coeff_list

run_exp = True
#plot_results = True

if run_exp:

    regularised_recons = {}
    exp = 0
    for i, Li_fourier in enumerate(Li_fourier_coeffs):
        regularised_recons['measurement=' + str(i)] = {}
        model = VariationalRegClass('MRI', 'TV')
        for reg_param in reg_params:
            regularised_recons['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)] = {}
            for output_dim in output_dims:

                print("Experiment_" + str(exp))
                exp+=1

                data = np.zeros((output_dim, output_dim), dtype='complex')
                data[output_dim//2 - 16 :output_dim//2 + 16, output_dim//2 - 16 :output_dim//2 + 16] = Li_fourier
                data = np.fft.fftshift(data)
                subsampling_matrix = np.zeros((output_dim, output_dim))
                subsampling_matrix[output_dim//2 - 16 :output_dim//2 + 16, output_dim//2 - 16 :output_dim//2 + 16] = 1
                subsampling_matrix = np.fft.fftshift(subsampling_matrix)

                recons = model.regularised_recons_from_subsampled_data(data, reg_param, subsampling_arr=subsampling_matrix, niter=5000)
                regularised_recons['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]['output_size=' + str(output_dim)] = \
                    [np.real(recons[0]).tolist(), np.imag(recons[0]).tolist()]

    save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/Experiments/MRI_birmingham/Results_MRI_dTV'
    json.dump(regularised_recons,
              open(save_dir + '/Robustness_31112020_TV_512_new.json', 'w'))

# with open('dTV/Results_MRI_dTV/Robustness_31112020_TV_512.json') as f:
#     d = json.load(f)
