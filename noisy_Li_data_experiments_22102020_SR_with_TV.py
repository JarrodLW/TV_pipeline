# Created: 16/11/2020.
# Based on noisy_Li_data_experiments_22102020_finer_hyperparam_search.py. Using the data dated 22/10/2020.

import numpy as np
import matplotlib.pyplot as plt
from processing import *
import json
import matplotlib.pyplot as plt
import os
import odl
import myOperators as ops
from Utils import *

dir = 'dTV/7Li_1H_MRI_Data_22102020/'

avgs = ['512', '1024', '2048', '4096', '8192']
Li_fourier_coeffs = []

for avg in avgs:
    # files from Bearshare folder, labelled 7Li_Axial_512averages_1mmslicethickness etc.
    fourier_Li_real_im_padded = np.reshape(np.fromfile(dir + '1mm_7Li_'+avg+'_avgs/fid', dtype=np.int32), (64, 128))
    fourier_Li_real_im = fourier_Li_real_im_padded[:, 1:65]
    fourier_Li_real_im = fourier_Li_real_im[::2, :]

    fourier_Li_real = fourier_Li_real_im[:, 1::2]
    fourier_Li_im = fourier_Li_real_im[:, ::2]
    fourier_Li = fourier_Li_real + fourier_Li_im*1j
    Li_fourier_coeffs.append(fourier_Li)

my_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier_coeffs[-1])))
image = np.abs(my_recon)

#plt.figure()
#plt.imshow(image, cmap=plt.cm.gray)

reg_types = ['TV', 'TGV']
reg_params = np.logspace(np.log10(2e3), np.log10(1e7), num=20)

output_dims = [int(64), int(128)]
#output_dims = [int(128)]
#Li_fourier_coeffs = [Li_fourier_coeffs[-1]]

run_exp = False
plot_results = True

if run_exp:

    regularised_recons = {}
    exp = 0
    for i, Li_fourier in enumerate(Li_fourier_coeffs):
        regularised_recons['avgs=' + str(avgs[i])] = {}
        for reg_type in reg_types:
            regularised_recons['avgs=' + str(avgs[i])]['reg_type=' + reg_type] = {}
            model = VariationalRegClass('MRI', reg_type)
            for reg_param in reg_params:
                regularised_recons['avgs=' + str(avgs[i])]['reg_type=' + reg_type][
                    'reg_param=' + '{:.1e}'.format(reg_param)] = {}
                for output_dim in output_dims:

                    print("Experiment_" + str(exp))
                    exp+=1

                    data = np.zeros((output_dim, output_dim), dtype='complex')
                    data[output_dim//2 - 16 :output_dim//2 + 16, output_dim//2 - 16 :output_dim//2 + 16] = Li_fourier
                    data = np.fft.fftshift(data)
                    subsampling_matrix = np.zeros((output_dim, output_dim))
                    subsampling_matrix[output_dim//2 - 16 :output_dim//2 + 16, output_dim//2 - 16 :output_dim//2 + 16] = 1
                    subsampling_matrix = np.fft.fftshift(subsampling_matrix)

                    recons_bernoulli = model.regularised_recons_from_subsampled_data(data, reg_param, subsampling_arr=subsampling_matrix, niter=5000)
                    regularised_recons['avgs=' + str(avgs[i])]['reg_type=' + reg_type]['reg_param=' + '{:.1e}'.format(reg_param)]['output_size=' + str(output_dim)] = \
                        [np.real(recons_bernoulli[0]).tolist(), np.imag(recons_bernoulli[0]).tolist()]

    json.dump(regularised_recons, open('dTV/Results_MRI_dTV/TV_recons_multiple_avgs_22102020_SR.json', 'w'))

if plot_results:

    with open('dTV/Results_MRI_dTV/TV_recons_multiple_avgs_22102020_SR.json') as f:
        d = json.load(f)

    #dir_save = '/Users/jlw31/Desktop/Presentations:Reports/dTV results/Applications_of_dTV'

    for avg in avgs:
        d2 = d['avgs='+avg]
        for reg_type in reg_types:
            d3 = d2['reg_type='+reg_type]

            fig, axs = plt.subplots(4, 5, figsize=(5, 4))
            for output_dim in output_dims:
                for i, reg_param in enumerate(reg_params):
                    d4 = d3['reg_param='+'{:.1e}'.format(reg_param)]
                    recon = np.asarray(d4['output_size=' + str(output_dim)]).astype('float64')
                    image = np.abs(recon[0] + 1j*recon[1])

                    image_rotated_flipped = image[:, ::-1].T[:, ::-1]

                    axs[i//5, i % 5].imshow(image_rotated_flipped, cmap=plt.cm.gray)
                    axs[i//5, i % 5].axis("off")

                fig.tight_layout(w_pad=0.4, h_pad=0.4)
                plt.savefig("dTV/Results_MRI_dTV/SR_with_TV_22102020_data_8192_avgs_32_to_" + str(output_dim) + ".pdf")
