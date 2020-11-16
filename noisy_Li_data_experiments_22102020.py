# Created: 26/10/2020.
# Based on noisy_Li_data_experiments.py. Using the data dated 22/10/2020.

import numpy as np
import matplotlib.pyplot as plt
from processing import *
import json
import matplotlib.pyplot as plt
import os
import odl
import myOperators as ops
import tifffile as tf

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
plt.imshow(image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.real(Li_fourier_coeffs[0]), cmap=plt.cm.gray)
plt.show()

plt.figure()
plt.imshow(np.imag(Li_fourier_coeffs[0]), cmap=plt.cm.gray)
plt.show()

plt.figure()
plt.imshow(np.abs(Li_fourier_coeffs[0]), cmap=plt.cm.gray)
plt.show()

reg_types = ['TV', 'TGV']
reg_params = [1., 2., 5., 10., 20., 50., 10**2, 2*10**2, 5*10**2, 10**3, 2*10**3, 5*10**3, 10**4, 2*10**4, 5*10**4,
              10**5, 2*10**5, 5*10**5, 10**6, 2*10**6, 5*10**6, 10**7]

regularised_recons = {}
for i, Li_fourier in enumerate(Li_fourier_coeffs):
    regularised_recons['avgs=' + str(avgs[i])] = {}
    for reg_type in reg_types:
        regularised_recons['avgs=' + str(avgs[i])]['reg_type=' + reg_type] = {}
        model = VariationalRegClass('MRI', reg_type)
        for reg_param in reg_params:

            recons_bernoulli = model.regularised_recons_from_subsampled_data(np.fft.fftshift(Li_fourier), reg_param, niter=5000)
            regularised_recons['avgs=' + str(avgs[i])]['reg_type=' + reg_type]['reg_param=' + '{:.1e}'.format(reg_param)] = np.abs(recons_bernoulli[0]).tolist()

json.dump(regularised_recons, open('dTV/Results_MRI_dTV/TV_TGV_recons_multiple_avgs_22102020.json', 'w'))


with open('dTV/Results_MRI_dTV/TV_TGV_recons_multiple_avgs_22102020.json') as f:
    d = json.load(f)

dir = '/Users/jlw31/Desktop/Presentations:Reports/dTV results/Applications_of_dTV'

#dict_best_matching_image = {}
for avg in avgs:
    d2 = d['avgs='+avg]
    #dict_best_matching_image['avgs='+avg] = {}
    for reg_type in reg_types:
        d3 = d2['reg_type='+reg_type]

        fig, axs = plt.subplots(4, 5, figsize=(5, 4))

        #recon_array = np.zeros((len(reg_params), fourier_Li.shape[0], fourier_Li.shape[1])).astype('float64')
        #ind_best = 0
        #L2_dist = np.inf
        for i, reg_param in enumerate(reg_params[:-2]):
            recon = np.asarray(d3['reg_param='+'{:.1e}'.format(reg_param)]).astype('float64')
            #recon_array[i, :, :] = recon

            axs[i//5, i % 5].imshow(recon[:, ::-1].T[:, ::-1], cmap=plt.cm.gray)
            axs[i//5, i % 5].axis("off")

        plt.savefig(dir + "/" + reg_type + "_reg_22102020_data_" + avg +"_avgs.pdf")

            #L2_dist_new = np.sum(np.square(np.asarray(d3['reg_param='+'{:.1e}'.format(reg_param)]).T[:, ::-1] - image))
            # there's an annoying rotation that I need to fix at some point
            #if L2_dist_new < L2_dist:
            #    ind_best = i

            #L2_dist = L2_dist_new
        #dict_best_matching_image['avgs='+avg]['reg_type=' + reg_type] = recon_array[ind_best]

        #tf.imwrite('dTV/Results_MRI_dTV/'+reg_type+'reg_'+avg+'_avgs_22102020.tif', recon_array, photometric='minisblack')


d2 = d['avgs='+avgs[-1]]
d3 = d2['reg_type='+'TV']
recon = np.asarray(d3['reg_param=1.0e+00']).astype('float64')
plt.figure()
plt.imshow(recon, cmap=plt.cm.gray)

my_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier_coeffs[-1])))
image = np.abs(my_recon)
plt.figure()
plt.imshow(image, cmap=plt.cm.gray)

np.sum(np.square(recon.T[:, ::-1] - image))

