import numpy as np
import json
import matplotlib.pyplot as plt
import os
import odl
import myOperators as ops
from Utils import *
import sys
import datetime as dt
from skimage.measure import block_reduce
import dTV.myAlgorithms as algs
import dTV.myFunctionals as fctls
import datetime as dt

height = 32
width = 32
complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                              shape=[height, width], dtype='complex', interp='linear')
image_space = complex_space.real_space ** 2
Yaff = odl.tensor_space(6)

X = odl.ProductSpace(image_space, Yaff)

# grbbing data
f_coeff_list = []

for i in range(3, 35):
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)

Li_fourier = np.average(np.asarray(f_coeff_list), axis=0)

fourier_data_real = np.real(Li_fourier)
fourier_data_im = np.imag(Li_fourier)

# defining the forward op - I should do the subsampling in a more efficient way
fourier_transf = ops.RealFourierTransform(image_space)
data_height, data_width = fourier_data_real.shape

subsampling_arr = np.zeros((height, width))
subsampling_arr[height//2 - data_height//2: height//2 + data_height//2, width//2 - data_width//2: width//2 + data_width//2] = 1
subsampling_arr = np.fft.fftshift(subsampling_arr)
subsampling_arr_doubled = np.array([subsampling_arr, subsampling_arr])

forward_op = fourier_transf.range.element(subsampling_arr_doubled) * fourier_transf

padded_fourier_data_real = np.zeros((height, width))
padded_fourier_data_im = np.zeros((height, width))
padded_fourier_data_real[height//2 - data_height//2: height//2
                                                     + data_height//2, width//2 - data_width//2: width//2 + data_width//2] = fourier_data_real

padded_fourier_data_im[height // 2 - data_height // 2: height // 2
                                                         + data_height // 2, width // 2 - data_width // 2: width // 2 + data_width // 2] = fourier_data_im

data_odl = forward_op.range.element([np.fft.fftshift(padded_fourier_data_real), np.fft.fftshift(padded_fourier_data_im)])

# checking that shifts are consistent in TV discrepancy

data = np.load('/Users/jlw31/Desktop/debugging_fft_shifts.npy')
synth_data = data[0]
fully_averaged_data = data[1]
GT_proxy_data = data[2]

synth_data_complex = synth_data[0] + 1j*synth_data[1]
fully_averaged_data_complex = fully_averaged_data[0] + 1j*fully_averaged_data[1]
GT_proxy_data_complex = GT_proxy_data[0] + 1j*GT_proxy_data[1]

plt.figure()
plt.imshow(np.abs(synth_data_complex))

plt.figure()
plt.imshow(np.abs(fully_averaged_data_complex))

plt.figure()
plt.imshow(np.abs(GT_proxy_data_complex))

np.sqrt(np.sum(np.square(np.abs(synth_data_complex - fully_averaged_data_complex))))
np.sqrt(np.sum(np.square(np.abs(fully_averaged_data_complex - GT_proxy_data_complex))))
np.sqrt(np.sum(np.square(np.abs(GT_proxy_data_complex - synth_data_complex))))

np.sqrt(np.sum(np.square(np.abs(synth_data_complex))))
np.sqrt(np.sum(np.square(np.abs(fully_averaged_data_complex))))
np.sqrt(np.sum(np.square(np.abs(GT_proxy_data_complex))))

dTV_data = np.load('/Users/jlw31/Desktop/dTV_debugging_fft_shifts.npy')

raw_data = dTV_data[0, 0] + 1j*dTV_data[0, 1]
fourier_diff = dTV_data[1, 0] + 1j*dTV_data[1, 1]
raw_data_minus_GT = dTV_data[2, 0] + 1j*dTV_data[2, 1]
raw_data_minus_GT_TV = dTV_data[3, 0] + 1j*dTV_data[3, 1]
GT_TV = dTV_data[4, 0] + 1j*dTV_data[4, 1]

plt.figure()
plt.imshow(np.abs(raw_data), cmap=plt.cm.gray)
plt.title("raw data")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(fourier_diff), cmap=plt.cm.gray)
plt.title("diff between raw data and recon synth data")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(raw_data_minus_GT), cmap=plt.cm.gray)
plt.title("diff between raw data and GT, dTV")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(raw_data - np.fft.fftshift(fully_averaged_data_complex)), cmap=plt.cm.gray)
plt.title("diff between raw data and GT, TV")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(raw_data_minus_GT_TV), cmap=plt.cm.gray)
plt.title("diff between raw data and GT proxy, dTV")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(raw_data - GT_TV), cmap=plt.cm.gray)
plt.title("diff between raw data and GT proxy, dTV, take 2")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(raw_data - np.fft.fftshift(GT_proxy_data_complex)), cmap=plt.cm.gray)
plt.title("diff between raw data and GT proxy, TV")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(GT_TV), cmap=plt.cm.gray)
plt.title("GT proxy data")
plt.colorbar()

plt.figure()
plt.imshow(np.real(GT_TV), cmap=plt.cm.gray)
plt.title("GT proxy data, real part")
plt.colorbar()

plt.figure()
plt.imshow(np.imag(GT_TV), cmap=plt.cm.gray)
plt.title("GT proxy data, imaginary part")
plt.colorbar()

plt.figure()
plt.imshow(np.real(np.fft.fftshift(GT_proxy_data_complex)), cmap=plt.cm.gray)
plt.title("GT proxy, TV, real part")
plt.colorbar()

plt.figure()
plt.imshow(np.imag(np.fft.fftshift(GT_proxy_data_complex)), cmap=plt.cm.gray)
plt.title("GT proxy, TV, imaginary part")
plt.colorbar()

plt.figure()
plt.imshow(np.imag(np.fft.fftshift(GT_proxy_data_complex)) - np.imag(GT_TV), cmap=plt.cm.gray)
plt.title("GT proxy, diff real-imaginary")
plt.colorbar()

plt.figure()
plt.imshow(np.real(np.fft.fftshift(GT_proxy_data_complex)) - np.real(GT_TV), cmap=plt.cm.gray)
plt.title("GT proxy, diff imaginary-real")
plt.colorbar()


#
image_H_low_res = np.load('dTV/MRI_15032021/Results_15032021/pre_registered_H_image_low_res.npy')
GT_TV_image = np.load(dir + 'Results_15032021/example_TV_recon_15032021.npy')
GT_TV_image = np.abs(GT_TV_image[0] + 1j*GT_TV_image[1])

plt.figure()
plt.imshow(image_H_low_res, cmap=plt.cm.gray)

plt.figure()
plt.imshow(GT_TV_image, cmap=plt.cm.gray)

recon_error(image_H_low_res/np.amax(image_H_low_res), GT_TV_image/np.amax(GT_TV_image))


#
rec_abs = np.abs(recon[0] + 1j*recon[1])
#fourier = forward_op(forward_op.domain.element([recon[0], recon[1]]))
fourier_complex = np.fft.fft2(recon[0] + 1j*recon[1])
#fourier_complex = fourier[0].asarray() + 1j * fourier[1].asarray()
fourier_shift = np.fft.ifftshift(fourier_complex)
fourier_shift_subsampled = fourier_shift[64 - 16:64 + 16, 64 - 16:64 + 16]
rec_fourier = np.fft.ifft2(np.fft.fftshift(fourier_shift_subsampled))

plt.imshow(np.abs(rec_fourier), cmap=plt.cm.gray)

rec_fourier_normalised = rec_fourier/np.sqrt(np.sum(np.square(np.abs(rec_fourier))))
GT_TV_image_normalised = GT_TV_image/np.sqrt(np.sum(np.square(GT_TV_image)))
recon_error(np.abs(rec_fourier_normalised), GT_TV_image_normalised)[2]

plt.figure()
plt.imshow(np.abs(diff_shift_subsampled), cmap=plt.cm.gray)

recon_error(image_H_low_res/np.sqrt(np.sum(np.square(image_H_low_res))), GT_TV_image_normalised)[2]
