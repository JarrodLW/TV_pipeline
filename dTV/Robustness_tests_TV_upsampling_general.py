# Created: 11/01/2021.
# Based on noisy_Li_data_experiments_22102020_finer_hyperparam_search.py. Using the data dated 31/11/2020.
# This consolidates all TV experiments, for both datasets and all averages in a single (messy) script.

import numpy as np
import matplotlib.pyplot as plt
from processing import *
import json
import matplotlib.pyplot as plt
import os
import odl
import myOperators as ops
from Utils import *
import sys

dir = 'dTV/7Li_1H_MRI_Data_31112020/'
n = int(sys.argv[1]) # 512, 1024, 2048, etc
dataset = sys.argv[2] # string, has to be either 'Li2SO4' or 'Li_LS'

def unpacking_fourier_coeffs(arr):

    fourier_real_im = arr[:, 1:65]
    fourier_real_im = fourier_real_im[::2, :]

    fourier_real = fourier_real_im[:, 1::2]
    fourier_im = fourier_real_im[:, ::2]
    fourier = fourier_real + fourier_im * 1j

    return fourier

f_coeff_list = []

for i in range(2, 34):
    f_coeffs = np.reshape(np.fromfile(dir + dataset +'/'+str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)

if n !=512:
    f_coeff_arr = np.asarray(f_coeff_list)
    f_coeff_list_grouped = []
    num = n//512
    for i in range(num):
        data_arr = np.roll(f_coeff_arr, i, axis=0)
        for ele in range(len(f_coeff_list)//num):
            f_coeff_list_grouped.append(np.sum(data_arr[num*ele:num*(ele+1)], axis=0)/num)

    f_coeff_list = f_coeff_list_grouped

reg_params = np.concatenate((np.asarray([0.001, 1., 10**0.5, 10., 10**1.5, 10**2]), np.logspace(3., 4.5, num=20)))
#output_dims = [int(32), int(64)]
output_dims = [int(32)]
#Li_fourier_coeffs = f_coeff_list
Li_fourier_coeffs = [f_coeff_list[0]]

save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/Experiments/MRI_birmingham/Results_MRI_dTV'

if dataset=='Li2SO4':
    with open(save_dir + '/New/results/TV_results/Robustness_31112020_TV_' + str(n) + '_new.json') as f:
        d = json.load(f)

elif dataset=='Li_LS':
    with open(save_dir + '/New/results_Li_LS/TV_results_Li_LS/Robustness_31112020_TV_' + str(n) + '_Li_LS_new.json') as f:
        d = json.load(f)

f.close()

run_exp = True

if run_exp:

    regularised_recons = {}
    exp = 0
    for i, Li_fourier in enumerate(Li_fourier_coeffs):
        #regularised_recons['measurement=' + str(i)] = {}
        model = VariationalRegClass('MRI', 'TV')
        for reg_param in reg_params:

            if 'reg_param=' + '{:.1e}'.format(reg_param) not in d['measurement=' + str(i)].keys():
                d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)] = {}

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
                    d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]['output_size=' + str(output_dim)] = \
                        [np.real(recons[0]).tolist(), np.imag(recons[0]).tolist()]

    if dataset == 'Li2SO4':
        json.dump(d,
                  open(save_dir + '/New/results/TV_results/Robustness_31112020_TV_' + str(n) + '_new.json', 'w'))

    elif dataset == 'Li_LS':
        json.dump(d,
                  open(save_dir + '/New/results_Li_LS/TV_results_Li_LS/Robustness_31112020_TV_' + str(n) + '_Li_LS_new.json', 'w'))
