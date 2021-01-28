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
#reg_params = np.logspace(3., 4.5, num=20)
#reg_params = [10**3]
reg_params = np.logspace(2., np.log10(5*10**3), num=15)
#output_dims = [int(32), int(64)]
output_dims = [int(32)]
Li_fourier = np.average(np.asarray(f_coeff_list), axis=0)

naive_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier)))

run_exp = True
#plot_results = True

recon_arr = np.zeros((15, 32, 32))

if run_exp:

    regularised_recons = {}
    exp = 0

    model = VariationalRegClass('MRI', 'TV')
    for k, reg_param in enumerate(reg_params):
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

            recon_arr[k, :, :] = recons[0]

            # regularised_recons['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]['output_size=' + str(output_dim)] = \
            #     [np.real(recons[0]).tolist(), np.imag(recons[0]).tolist()]

    # save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/Experiments/MRI_birmingham/Results_MRI_dTV'
    # json.dump(regularised_recons,
    #           open(save_dir + '/Robustness_31112020_TV_512_new.json', 'w'))

np.save('/Users/jlw31/Desktop/Presentations:Reports/dTV results/31112020_results/TV_regularised_recons_Li2SO4.npy', recon_arr)

recon_arr_16384 = np.load('/Users/jlw31/Desktop/Presentations:Reports/dTV results/31112020_results/TV_regularised_recons_Li2SO4.npy')

fig, axs = plt.subplots(5, 3, figsize=(6, 10))

for k in range(15):

    axs[k // 3, k % 3].imshow(np.abs(recon_arr_16384[k, :, :]), cmap=plt.cm.gray)
    axs[k // 3, k % 3].axis("off")

plt.figure()
plt.imshow(np.abs(recons[0]).T[:, ::-1], cmap=plt.cm.gray)
plt.axis("off")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(naive_recon), cmap=plt.cm.gray)