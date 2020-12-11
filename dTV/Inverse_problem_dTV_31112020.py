# created 11/12/2020
# based on "Inverse_problem_dTV_22102020.py"

import h5py
import numpy as np
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json
import dTV.myAlgorithms as algs
import matplotlib.pyplot as plt
import os
import odl
#import dTV.myOperators as ops
import myOperators as ops
from Utils import *
from skimage.measure import block_reduce
from processing import *

dir_H = 'dTV/7Li_1H_MRI_Data_31112020/1H_Li2SO4/'

for i in range(1, 8):
    image_H = np.reshape(np.fromfile(dir_H+'/'+str(i)+'/pdata/1/2dseq', dtype=np.uint16), (32, 32))
    plt.figure()
    plt.imshow(np.abs(image_H), cmap=plt.cm.gray)


image_H = np.reshape(np.fromfile(dir_H+'5/pdata/1/2dseq', dtype=np.uint16), (128, 128))
plt.figure()
plt.imshow(np.abs(image_H), cmap=plt.cm.gray)

image_H = np.reshape(np.fromfile(dir_H+'6/pdata/1/2dseq', dtype=np.uint16), (128, 128))
plt.figure()
plt.imshow(np.abs(image_H), cmap=plt.cm.gray)


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

f_coeffs_averaged = np.average(np.asarray(f_coeff_list), axis=0)
my_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs_averaged)))
image = np.abs(my_recon)

plt.figure()
plt.imshow(image, cmap=plt.cm.gray)

reg_param = 500
model = VariationalRegClass('MRI', 'TV')
recons = model.regularised_recons_from_subsampled_data(np.fft.fftshift(f_coeffs_averaged), reg_param, subsampling_arr=None, niter=5000)

recon_rotated = recons[0].T[:, ::-1]

plt.figure()
plt.imshow(np.abs(recons[0]), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(recon_rotated), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(recon_rotated.T[::-1, :]), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(image_H.T[::-1, :]), cmap=plt.cm.gray)