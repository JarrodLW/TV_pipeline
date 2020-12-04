# created 02/11/2020
# based on "MRI_experiments_dTV_2.py"

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

dir = 'dTV/7Li_1H_MRI_Data_22102020/'

avgs = ['512', '1024', '2048', '4096', '8192']
Li_fourier_coeffs = []

image_H = np.reshape(np.fromfile(dir+'1mm_1H_high_res/2dseq', dtype=np.uint16), (128, 128))

for avg in avgs:
    # files from Bearshare folder, labelled 7Li_Axial_512averages_1mmslicethickness etc.
    fourier_Li_real_im_padded = np.reshape(np.fromfile(dir + '1mm_7Li_'+avg+'_avgs/fid', dtype=np.int32), (64, 128))
    fourier_Li_real_im = fourier_Li_real_im_padded[:, 1:65]
    #fourier_Li_real_im = fourier_Li_real_im_padded[:, 5:69]
    fourier_Li_real_im = fourier_Li_real_im[::2, :]

    fourier_Li_real = fourier_Li_real_im[:, 1::2]
    fourier_Li_im = fourier_Li_real_im[:, ::2]
    fourier_Li = fourier_Li_real + fourier_Li_im*1j
    Li_fourier_coeffs.append(fourier_Li)

fourier_data_real = np.real(Li_fourier_coeffs[2])
fourier_data_im = np.imag(Li_fourier_coeffs[2])

naive_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_data_real+1j*fourier_data_im)))

gamma = 0.995
strong_cvx = 1e-2
niter_prox = 20
niter = 150


alphas = [50, 10**2, 5*10**2, 10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6]
etas = [0.01]

Yaff = odl.tensor_space(6)

sinfo_high_res = image_H.T
sinfo_med_res = block_reduce(sinfo_high_res, block_size=(2, 2), func=np.mean)
sinfo_low_res = block_reduce(sinfo_high_res, block_size=(4, 4), func=np.mean)

sinfos = {}
#sinfos['high_res'] = sinfo_high_res
sinfos['med_res'] = sinfo_med_res
#sinfos['low_res'] = sinfo_low_res

#fourier_data_real = np.random.normal(-2000, 1e4, size=fourier_data_real.shape)
#fourier_data_im = np.random.normal(600, 1e4, size=fourier_data_im.shape)

for dict_key in sinfos.keys():

    sinfo = sinfos[dict_key]

    height, width = sinfo.shape
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

    #forward_op = fourier_transf
    #data_odl = forward_op.range.element([np.fft.fftshift(fourier_data_real), np.fft.fftshift(fourier_data_im)])

    # Set some parameters and the general TV prox options
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = niter_prox

    reg_affine = odl.solvers.ZeroFunctional(Yaff)
    #x0 = X.zero()
    x0 = X.element([forward_op.adjoint(data_odl), X[1].zero()])

    f = fctls.DataFitL2Disp(X, data_odl, forward_op)

    dTV_regularised_recons = {}
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:
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
            palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
            print('Experiment '+'alpha='+str(alpha)+' eta='+str(eta))

            palm.run(niter)

            recon = palm.x[0].asarray()

            diff = forward_op(forward_op.domain.element([recon[0], recon[1]])) - data_odl
            diff = diff[0].asarray() + 1j*diff[1].asarray()
            diff_shift = np.fft.ifftshift(diff)
            diff_shift_subsampled = diff_shift[sinfo.shape[0]//2-16:sinfo.shape[0]//2+16,
                                    sinfo.shape[1]//2-16:sinfo.shape[1]//2+16]

            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['recon'] = recon.tolist()
            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['affine_params'] = palm.x[1].asarray().tolist()
            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['fourier_diff'] = [np.real(diff_shift_subsampled).tolist(),
                                                                                         np.imag(diff_shift_subsampled).tolist()]

json.dump(dTV_regularised_recons, open('dTV/Results_MRI_dTV/dTV_recons_2048_avgs_22102020_SR_to_64.json', 'w'))

with open('dTV/Results_MRI_dTV/dTV_recons_2048_avgs_22102020_SR_to_64.json') as f:
    d = json.load(f)

fig, axs = plt.subplots(10, 4, figsize=(5, 4))

for i, alpha in enumerate(alphas):
    recon = np.asarray(dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['recon'])
    recon = recon[0] + 1j*recon[1]
    im = np.abs(recon)

    fourier_diff = np.asarray(dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['fourier_diff'])
    fourier_diff = fourier_diff[0] + 1j*fourier_diff[1]
    abs_fourier_diff = np.abs(fourier_diff)

    axs[i, 0].imshow(sinfo_med_res, cmap=plt.cm.gray)
    axs[i, 0].axis("off")
    axs[i, 1].imshow(np.abs(naive_recon), cmap=plt.cm.gray)
    axs[i, 1].axis("off")
    axs[i, 2].imshow(im, cmap=plt.cm.gray)
    axs[i, 2].axis("off")
    axs[i, 3].imshow(abs_fourier_diff, cmap=plt.cm.gray)
    axs[i, 3].axis("off")





# fourier_rec_odl = fourier_transf.inverse(data_odl).asarray()
# fourier_rec_odl = fourier_rec_odl[0] + 1j*fourier_rec_odl[1]
# plt.figure()
# plt.imshow(np.abs(fourier_rec_odl), cmap=plt.cm.gray)
#
# fourier_rec = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_data_real + 1j*fourier_data_im)))
# plt.figure()
# plt.imshow(np.abs(fourier_rec), cmap=plt.cm.gray)
#
# fourier_rec_odl_norm = np.sqrt(np.sum(np.square(np.abs(fourier_rec_odl))))
# fourier_rec_norm = np.sqrt(np.sum(np.square(np.abs(fourier_rec))))
#
# np.sum(np.square((np.abs(fourier_rec_odl)/fourier_rec_odl_norm) - (np.abs(fourier_rec)/fourier_rec_norm)))
#
# plt.figure()
# plt.imshow(sinfo_low_res, cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(image_H.T)
#
# plt.figure()
# plt.imshow(sinfo)
#
# fourier_data = fourier_data_real + 1j*fourier_data_im
# plt.imshow(np.abs(fourier_data), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(fourier_data_real, cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(fourier_data_im, cmap=plt.cm.gray)
