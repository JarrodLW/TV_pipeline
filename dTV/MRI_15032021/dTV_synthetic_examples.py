# Created: 30/03/2021.

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
from skimage.transform import resize
import imageio

phantom1 = imageio.imread('dTV/MRI_15032021/Data_15032021/Phantom_data/Phantom_circle_resolution1.png')
#phantom2 = imageio.imread('dTV/MRI_15032021/Data_15032021/Phantom_data/Phantom_circle_resolution2.png')

phantom1_128 = resize(phantom1, (128, 128))
phantom1_64 = resize(phantom1, (64, 64))
phantom1_32 = resize(phantom1, (32, 32))

sinfo_32 = -0.7*phantom1_32**2 + 0.1*phantom1_32 -1
sinfo_64 = -phantom1_64

plt.figure()
plt.imshow(phantom1, cmap=plt.cm.gray)

# plt.figure()
# plt.imshow(phantom2, cmap=plt.cm.gray)

plt.figure()
plt.imshow(phantom1_128, cmap=plt.cm.gray)

plt.figure()
plt.imshow(sinfo_32, cmap=plt.cm.gray)

plt.figure()
plt.imshow(sinfo_64, cmap=plt.cm.gray)

## generating synthetic data

height = 32
width = 32
complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.], shape=[height, width], dtype='complex')
image_space = complex_space.real_space ** 2

fourier_transf = ops.RealFourierTransform(image_space)

phantom1_32_odl = image_space.element([phantom1_32, np.zeros((32, 32))])
phantom1_32_data = fourier_transf(phantom1_32_odl)

phantom1_32_data_arr = phantom1_32_data.asarray()
norm_phantom1_data = np.sqrt(np.sum(np.square(np.abs(phantom1_32_data_arr))))

## modifying the phantom

# unpacking the Li data
dir_Li = 'dTV/MRI_15032021/Data_15032021/Li_data/'

f_coeff_list = []

for i in range(3, 35):
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)

f_coeff_arr = np.asarray(f_coeff_list)
fully_averaged = np.average(f_coeff_arr, axis=0)
Li_data_odl = fourier_transf.range.element([np.real(np.fft.fftshift(fully_averaged)), np.imag(np.fft.fftshift(fully_averaged))])
rec_Li = fourier_transf.inverse(Li_data_odl)

norm_fully_averaged_data = np.sqrt(np.sum(np.square(np.abs(fully_averaged))))

# rescaling the data to the same scale as fully-averaged Li data
phantom1_32_data *= (norm_fully_averaged_data/norm_phantom1_data)

phantom_32_odl_rescaled = fourier_transf.inverse(phantom1_32_data)
phantom_32_odl_rescaled.show()

# modulating so that the real and imaginary parts are of equal weight
phantom1_32_data_arr_complex = phantom1_32_data.asarray()[0] + 1j*phantom1_32_data.asarray()[1]
phantom1_32_data_arr_modulated = np.exp(1j*0.75)*phantom1_32_data_arr_complex

np.sqrt(np.sum(np.square(np.real(phantom1_32_data_arr_modulated))))
np.sqrt(np.sum(np.square(np.imag(phantom1_32_data_arr_modulated))))
phantom1_32_data_modulated = fourier_transf.range.element([np.real(phantom1_32_data_arr_modulated), np.imag(phantom1_32_data_arr_modulated)])

fourier_transf.inverse(phantom1_32_data_modulated).show()

# sanity check
np.sqrt(np.sum(np.square(np.abs(rec_Li.asarray()))))
np.sqrt(np.sum(np.square(np.abs(phantom_32_odl_rescaled.asarray()))))
phantom1_32_data_modulated.show()
Li_data_odl.show()

# adding noise consistent with 1024-average noise
white_noise = odl.phantom.noise.white_noise(fourier_transf.range, mean=0, stddev=500)
phantom1_32_data_modulated_noisy = phantom1_32_data_modulated + white_noise

fourier_transf.inverse(phantom1_32_data_modulated).show()

naive_recon_from_noisy = fourier_transf.inverse(phantom1_32_data_modulated_noisy).asarray()

## dTV recons

alpha = 8000.
eta = 0.01
gamma = 0.995
strong_cvx = 1e-5
niter_prox = 20
niter = 100

sinfo = sinfo_32
data_odl = phantom1_32_data_modulated_noisy

height, width = sinfo.shape
complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                      shape=[height, width], dtype='complex', interp='linear')
image_space = complex_space.real_space ** 2
Yaff = odl.tensor_space(6)

X = odl.ProductSpace(image_space, Yaff)

# defining the forward op - I should do the subsampling in a more efficient way
fourier_transf = ops.RealFourierTransform(image_space)
forward_op = fourier_transf

#data_height, data_width = fourier_data_real.shape
# subsampling_arr = np.zeros((height, width))
# # subsampling_arr[height//2 - data_height//2: height//2 + data_height//2, width//2 - data_width//2: width//2 + data_width//2] = 1
# # subsampling_arr = np.fft.fftshift(subsampling_arr)
# # subsampling_arr_doubled = np.array([subsampling_arr, subsampling_arr])
# #
# # forward_op = fourier_transf.range.element(subsampling_arr_doubled) * fourier_transf
# #
# # padded_fourier_data_real = np.zeros((height, width))
# # padded_fourier_data_im = np.zeros((height, width))
# # padded_fourier_data_real[height//2 - data_height//2: height//2
# #                                                      + data_height//2, width//2 - data_width//2: width//2 + data_width//2]=fourier_data_real
# #
# # padded_fourier_data_im[height // 2 - data_height // 2: height // 2
# #                                                          + data_height // 2, width // 2 - data_width // 2: width // 2 + data_width // 2] = fourier_data_im
# #
# # data_odl = forward_op.range.element([np.fft.fftshift(padded_fourier_data_real), np.fft.fftshift(padded_fourier_data_im)])



# Set some parameters and the general TV prox options
prox_options = {}
prox_options['name'] = 'FGP'
prox_options['warmstart'] = True
prox_options['p'] = None
prox_options['tol'] = None
prox_options['niter'] = niter_prox

reg_affine = odl.solvers.ZeroFunctional(Yaff)
x0 = X.element([forward_op.adjoint(data_odl), X[1].zero()])

f = fctls.DataFitL2Disp(X, data_odl, forward_op)

print("start: " + dt.datetime.now().isoformat())
reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                                gamma=gamma, eta=eta, NonNeg=False, strong_convexity=strong_cvx,
                                                                prox_options=prox_options)

g = odl.solvers.SeparableSum(reg_im, reg_affine)

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
      odl.solvers.CallbackShow(step=5))

L = [1, 1e+2]
ud_vars = [0]

# %%
palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
palm.run(niter)

print("end: " + dt.datetime.now().isoformat())

recon = palm.x[0].asarray()

plt.figure()
plt.imshow(phantom1_32, cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
plt.imshow(sinfo_32, cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
plt.imshow(np.abs(recon[0] + 1j*recon[1]), cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
plt.imshow(np.abs(naive_recon_from_noisy[0] + 1j*naive_recon_from_noisy[1]), cmap=plt.cm.gray)
plt.axis("off")
