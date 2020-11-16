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

Li_fourier_coeffs_av = (Li_fourier_coeffs[0] + 2*Li_fourier_coeffs[1] + 4*Li_fourier_coeffs[2] * 8*Li_fourier_coeffs[3] + 16*Li_fourier_coeffs[4])/31

my_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier_coeffs[-1])))
image = np.abs(my_recon)

my_recon_2 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier_coeffs_av)))
image_2 = np.abs(my_recon_2)

plt.figure()
plt.imshow(image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(image_2, cmap=plt.cm.gray)

reg_types = ['TV']
reg_params = np.logspace(np.log10(2e3), np.log10(5e4), num=20)

regularised_recons = {}
exp = 0
for i, Li_fourier in enumerate(Li_fourier_coeffs):
    regularised_recons['avgs=' + str(avgs[i])] = {}
    for reg_type in reg_types:
        regularised_recons['avgs=' + str(avgs[i])]['reg_type=' + reg_type] = {}
        model = VariationalRegClass('MRI', reg_type)
        for reg_param in reg_params:

            print("Experiment_" + str(exp))
            exp+=1

            recons_bernoulli = model.regularised_recons_from_subsampled_data(np.fft.fftshift(Li_fourier), reg_param, niter=5000)
            regularised_recons['avgs=' + str(avgs[i])]['reg_type=' + reg_type]['reg_param=' + '{:.1e}'.format(reg_param)] = \
                [np.real(recons_bernoulli[0]).tolist(), np.imag(recons_bernoulli[0]).tolist()]

json.dump(regularised_recons, open('dTV/Results_MRI_dTV/TV_recons_multiple_avgs_22102020_finer_hyperparam_full_recon.json', 'w'))


with open('dTV/Results_MRI_dTV/TV_TGV_recons_multiple_avgs_22102020_finer_hyperparam.json') as f:
    d = json.load(f)

dir_save = '/Users/jlw31/Desktop/Presentations:Reports/dTV results/Applications_of_dTV'

# the following TV-assisted recon is quite good. We'll use it as a reference
d2 = d['avgs=8192']
d3 = d2['reg_type=TV']
ref_recon = np.asarray(d3['reg_param='+'{:.1e}'.format(reg_params[17])]).astype('float64')

best_recons = []

for avg in avgs:
    d2 = d['avgs='+avg]
    for reg_type in reg_types:
        d3 = d2['reg_type='+reg_type]

        fig, axs = plt.subplots(4, 5, figsize=(5, 4))

        psnr = 0
        for i, reg_param in enumerate(reg_params):
            recon = np.asarray(d3['reg_param='+'{:.1e}'.format(reg_param)]).astype('float64')

            _, psnr_new, _ = recon_error(recon/np.sqrt(np.sum(np.square(recon))), ref_recon/np.sqrt(np.sum(np.square(ref_recon))))

            recon_rotated_flipped = recon[:, ::-1].T[:, ::-1]

            if psnr_new > psnr:
                best_recon = recon_rotated_flipped

            psnr = psnr_new

            axs[i//5, i % 5].imshow(recon_rotated_flipped, cmap=plt.cm.gray)
            axs[i//5, i % 5].axis("off")

        best_recons.append(best_recon)

        plt.savefig(dir_save + "/" + reg_type + "_reg_22102020_data_" + avg +"_avgs_fine_hyperparam.pdf")

plt.figure()
recon_512 = np.asarray(d["avgs=512"]['reg_type=TV']['reg_param='+'{:.1e}'.format(reg_params[11])])
plt.imshow(recon_512[:, ::-1].T[:, ::-1], cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
recon_1024 = np.asarray(d["avgs=1024"]['reg_type=TV']['reg_param='+'{:.1e}'.format(reg_params[12])])
plt.imshow(recon_1024[:, ::-1].T[:, ::-1], cmap=plt.cm.gray)
plt.axis("off")


plt.figure()
plt.imshow(recon[:, ::-1].T[:, ::-1])
plt.figure()
plt.imshow(image[::-1, :])

image_H = np.reshape(np.fromfile(dir+'1mm_1H_high_res/2dseq', dtype=np.uint16), (128, 128))
image_Li_8192 = np.reshape(np.fromfile(dir+'1mm_7Li_8192_avgs/2dseq', dtype=np.uint16), (32, 32))
image_Li_4096 = np.reshape(np.fromfile(dir+'1mm_7Li_4096_avgs/2dseq', dtype=np.uint16), (32, 32))
image_Li_2048 = np.reshape(np.fromfile(dir+'1mm_7Li_2048_avgs/2dseq', dtype=np.uint16), (32, 32))
image_Li_1024 = np.reshape(np.fromfile(dir+'1mm_7Li_1024_avgs/2dseq', dtype=np.uint16), (32, 32))
image_Li_512 = np.reshape(np.fromfile(dir+'1mm_7Li_512_avgs/2dseq', dtype=np.uint16), (32, 32))

plt.figure()
plt.imshow(image_Li_8192.T[::-1, :], cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
plt.imshow(image_Li_4096.T[::-1, :], cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
plt.imshow(image_Li_2048.T[::-1, :], cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
plt.imshow(image_Li_1024.T[::-1, :], cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
plt.imshow(image_Li_512.T[::-1, :], cmap=plt.cm.gray)
plt.axis("off")


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(image_Li_8192.T[::-1, :], cmap=plt.cm.gray)
ax1.axis("off")
ax2.imshow(best_recons[4], cmap=plt.cm.gray)
ax2.axis("off")
ax3.imshow(image_H.T[::-1, :], cmap=plt.cm.gray)
ax3.axis("off")
plt.savefig(dir_save + "/TV_recon_best_22102020_avgs_8192.png")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(image_Li_4096.T[::-1, :], cmap=plt.cm.gray)
ax1.axis("off")
ax2.imshow(best_recons[3], cmap=plt.cm.gray)
ax2.axis("off")
ax3.imshow(image_H.T[::-1, :], cmap=plt.cm.gray)
ax3.axis("off")
plt.savefig(dir_save + "/TV_recon_best_22102020_avgs_4096.png")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(image_Li_2048.T[::-1, :], cmap=plt.cm.gray)
ax1.axis("off")
ax2.imshow(best_recons[2], cmap=plt.cm.gray)
ax2.axis("off")
ax3.imshow(image_H.T[::-1, :], cmap=plt.cm.gray)
ax3.axis("off")
plt.savefig(dir_save + "/TV_recon_best_22102020_avgs_2048.png")
