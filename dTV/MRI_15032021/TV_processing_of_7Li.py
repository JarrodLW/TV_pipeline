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
import datetime as dt

dir = 'dTV/MRI_15032021/Data_15032021/Li_data/'
#n = int(sys.argv[1]) # 512, 1024, 2048, etc
#dataset = sys.argv[2] # string, has to be either 'Li2SO4' or 'Li_LS'
n = 2048

f_coeff_list = []

for i in range(3, 35):
    f_coeffs = np.reshape(np.fromfile(dir +str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs)
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
Li_fourier_coeffs = f_coeff_list

save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/' \
           'Experiments/MRI_birmingham/Results_15032021/TV_results'

# save_dir = 'dTV/MRI_15032021/Results_15032021'
filename = save_dir + '/TV_7Li_15032021_' + str(n) + '.json'

if os.path.isfile(filename):

    print("About to read previous datafile: "+filename+" at "+dt.datetime.now().isoformat())
    with open(filename, 'r') as f:
        d = json.load(f)
    print("Loaded previous datafile at "+dt.datetime.now().isoformat())

    f.close()

else:
    d = {}

run_exp = True

if run_exp:

    regularised_recons = {}
    exp = 0
    for i, Li_fourier in enumerate(Li_fourier_coeffs):
        model = VariationalRegClass('MRI', 'TV')

        if 'measurement=' + str(i) not in d.keys():
            d['measurement=' + str(i)] = {}

        for reg_param in reg_params:

            if 'reg_param=' + '{:.1e}'.format(reg_param) not in d['measurement=' + str(i)].keys():
                d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)] = {}

            for output_dim in output_dims:

                if 'output_size=' + str(output_dim) not in d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)].keys():

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

    # if dataset == 'Li2SO4':
    #     outputfile = save_dir + '/New/results/TV_results/Robustness_31112020_TV_' + str(n) + '_new.json'
    # elif dataset == 'Li_LS':
    #     outputfile = save_dir + '/New/results_Li_LS/TV_results_Li_LS/Robustness_31112020_TV_' + str(n) + '_Li_LS_new.json'

    print("About to write to datafile: " + filename + " at " + dt.datetime.now().isoformat())
    json.dump(d, open(filename, 'w'))
    print("Written outputfile at " + dt.datetime.now().isoformat())