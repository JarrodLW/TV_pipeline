# The purpose of this script is to identify the location of the capillaries in noisy data.

# Created 21/06/2021. Implementation of dTV-guided MRI model, suitable for solution via PDHG.

import odl
import numpy as np
import myOperators as ops
import dTV.myFunctionals as fctls
from skimage.transform import resize
import matplotlib.pyplot as plt
import dTV.Ptycho_XRF_project.misc as misc
from scipy.io import loadmat
from Utils import *
from time import time

# grabbing guide image
image_H_high_res = np.load('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res_filtered.npy')


avgs = ['512', '1024', '2048', '4096', '8192']
f_coeff_list = []
dir_Li = 'dTV/MRI_15032021/Data_24052021/Li_data/'
Li_range = range(8, 40)
for i in Li_range:
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), (80, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, 40)
    f_coeff_list.append(f_coeffs_unpacked)

f_coeff_arr = np.asarray(f_coeff_list)
f_coeff_arr_combined = np.zeros((len(avgs), 32, 40, 40), dtype='complex')

for avg_ind in range(len(avgs)):

    num = 2**avg_ind

    for i in range(num):
        data_arr = np.roll(f_coeff_arr, i, axis=0)
        for ele in range(len(f_coeff_list)//num):
            f_coeff_arr_combined[avg_ind, ele+i*len(f_coeff_list)//num, :, :] = np.sum(data_arr[num*ele:num*(ele+1)], axis=0)/num

Li_fourier = f_coeff_arr_combined[2, 0, :, :]

Li_fourier_padded = np.zeros((120, 120), dtype='complex')
Li_fourier_padded[40:80, 40:80] = Li_fourier

fourier_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier)))
fourier_upsampled_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier_padded)))

plt.figure()
plt.imshow(image_H_high_res, cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(fourier_recon), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(fourier_upsampled_recon), cmap=plt.cm.gray)

