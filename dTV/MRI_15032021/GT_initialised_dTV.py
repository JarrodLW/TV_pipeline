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

dir = 'dTV/MRI_15032021/Data_15032021/Li_data/'
n = 1024

image_H_high_res = np.load('dTV/MRI_15032021/Results_15032021/pre_registered_H_image_high_res_take_2.npy')

f_coeff_list = []

for i in range(3, 35):
    f_coeffs = np.reshape(np.fromfile(dir + str(i) + '/fid', dtype=np.int32), (64, 128))
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

#f_coeff_list = f_coeff_list[0: 10]

fully_averaged = np.average(f_coeff_arr, axis=0)
fully_averaged_shifted = np.fft.fftshift(fully_averaged)
recon_fully_averaged = np.fft.fftshift(np.fft.ifft2(fully_averaged_shifted))

sinfos = {}
sinfos['high_res'] = image_H_high_res

alpha = 8000.
eta = 0.01
gamma = 0.995
strong_cvx = 1e-5
niter_prox = 20
niter = 100

Yaff = odl.tensor_space(6)
exp=0

for i, Li_fourier in enumerate(f_coeff_list):

    fourier_data_real = np.real(Li_fourier)
    fourier_data_im = np.imag(Li_fourier)

    for dict_key in sinfos.keys():

        sinfo = sinfos[dict_key]
        height, width = sinfo.shape
        init_image = np.array([resize(np.real(recon_fully_averaged), (height, width)),
                               resize(np.imag(recon_fully_averaged), (height, width))])

        complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                              shape=[height, width], dtype='complex', interp='linear')
        image_space = complex_space.real_space ** 2

        X = odl.ProductSpace(image_space, Yaff)

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
                                                             + data_height//2, width//2 - data_width//2: width//2 + data_width//2]=fourier_data_real

        padded_fourier_data_im[height // 2 - data_height // 2: height // 2
                                                                 + data_height // 2, width // 2 - data_width // 2: width // 2 + data_width // 2] = fourier_data_im

        data_odl = forward_op.range.element([np.fft.fftshift(padded_fourier_data_real), np.fft.fftshift(padded_fourier_data_im)])

        # Set some parameters and the general TV prox options
        prox_options = {}
        prox_options['name'] = 'FGP'
        prox_options['warmstart'] = True
        prox_options['p'] = None
        prox_options['tol'] = None
        prox_options['niter'] = niter_prox

        reg_affine = odl.solvers.ZeroFunctional(Yaff)
        #x0 = X.element([forward_op.adjoint(data_odl), X[1].zero()])
        x0 = X.element([forward_op.adjoint(forward_op.range.element(init_image)), X[1].zero()])

        f = fctls.DataFitL2Disp(X, data_odl, forward_op)

        print("Experiment_" + str(exp))
        exp += 1

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
        ud_vars = [0, 1]

        # %%
        #palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
        palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
        palm.run(niter)

        print("end: " + dt.datetime.now().isoformat())

        recon = palm.x[0].asarray()
        synth_fourier = forward_op(forward_op.domain.element([recon[0], recon[1]]))
        synth_fourier_shifted = np.fft.fftshift(synth_fourier.asarray()[0] + 1j*synth_fourier.asarray()[1])
        fourier_32 = synth_fourier_shifted[height // 2 - data_height // 2: height // 2
                                                                 + data_height // 2, width // 2 - data_width // 2: width // 2 + data_width // 2]

plt.figure()
plt.imshow(np.abs(recon[0] + 1j*recon[1]), cmap=plt.cm.gray)

# plt.figure()
# plt.imshow(resize(np.abs(recon[0] + 1j*recon[1]), (32, 32)), cmap=plt.cm.gray)

recon_32 = np.fft.fftshift(np.fft.ifft2(fourier_32))
plt.imshow(np.abs(recon_32), cmap=plt.cm.gray)
