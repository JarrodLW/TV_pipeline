# Created: 14/01/2021.
# Based on noisy_Li_data_experiments_22102020_finer_hyperparam_search.py. Using the data dated 31/11/2020.
# This is meant to consolidate all dTV experiments, for both datasets and all averages in a single (messy) script.
# Need to extend to allow switching between datasets.

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

alpha = float(sys.argv[1])
#alpha=10.

run_expt = False
plot = True

Li_fourier = np.fft.fftshift(np.load('dTV/MRI_15032021/Results_24052021/32768_data.npy'))
naive_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier)))

#plt.imshow(np.abs(Li_fourier), cmap=plt.cm.gray)

# plt.figure()
# plt.imshow(np.abs(naive_recon), cmap=plt.cm.gray)

# plt.figure()
# plt.imshow(np.abs(Fourier_upsampled_recon), cmap=plt.cm.gray)

low_res_shape = (80, 128)
Li_range = range(8, 40)
low_res_data_width = 40
image_H_high_res = np.load('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res.npy')
image_H_high_res = resize(image_H_high_res, (120, 120))
image_H_med_res = resize(image_H_high_res, (80, 80))
image_H_low_res = resize(image_H_high_res, (40, 40))

# plt.figure()
# plt.imshow(image_H_high_res, cmap=plt.cm.gray)

sinfos = {}
sinfos['high_res'] = image_H_high_res
#sinfos['med_res'] = image_H_med_res
#sinfos['low_res'] = image_H_low_res

etas = np.logspace(-3., -1, num=5).tolist()
gammas = [0.9, 0.925, 0.95, 0.975, 0.99, 0.995]
etas = [0.01]
gammas = [0.99]
strong_cvx = 1e-5
niter_prox = 20
#niter = 300
niter = 100

Yaff = odl.tensor_space(6)
exp = 0

fourier_data_real = np.real(Li_fourier)
fourier_data_im = np.imag(Li_fourier)

d = {}

save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/' \
               'Experiments/MRI_birmingham/Results_24052021/dTV_processing_of_32768'

# save_dir = 'dTV/MRI_15032021/Results_15032021'
filename = save_dir + '/dTV_with_regis_alpha_'+str(alpha)+'.json'

if run_expt:
    for dict_key in sinfos.keys():

        sinfo = sinfos[dict_key]

        d['output_size=' + str(sinfo.shape[0])] = {}

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

        for gamma in gammas:
            d['output_size=' + str(sinfo.shape[0])]['gamma=' + str(gamma)] = {}

            for eta in etas:
                d['output_size=' + str(sinfo.shape[0])]['gamma=' + str(gamma)]['eta=' + str(eta)] = {}

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
                palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
                #palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
                palm.run(niter)

                print("end: " + dt.datetime.now().isoformat())

                recon = palm.x[0].asarray()
                affine_params = palm.x[1].asarray()
                diff = forward_op(forward_op.domain.element([recon[0], recon[1]])) - data_odl
                diff = diff[0].asarray() + 1j * diff[1].asarray()
                diff_shift = np.fft.ifftshift(diff)
                diff_shift_subsampled = diff_shift[sinfo.shape[0] // 2 - low_res_data_width//2:sinfo.shape[0] // 2 + low_res_data_width//2,
                                        sinfo.shape[1] // 2 - low_res_data_width//2:sinfo.shape[1] // 2 + low_res_data_width//2]

                d['output_size=' + str(sinfo.shape[0])]['gamma=' + str(gamma)]['eta=' + str(eta)]['recon'] = recon.tolist()
                d['output_size=' + str(sinfo.shape[0])]['gamma=' + str(gamma)]['eta=' + str(eta)]['fourier_diff'] = [
                    np.real(diff_shift_subsampled).tolist(),
                    np.imag(diff_shift_subsampled).tolist()]
                d['output_size=' + str(sinfo.shape[0])]['gamma=' + str(gamma)]['eta=' + str(eta)]['affine_params'] = \
                    affine_params.tolist()

            # print("About to write to datafile: " + outputfile + " at " + dt.datetime.now().isoformat())
            # json.dump(d, open(outputfile, 'w'))
            # print("Written outputfile at " + dt.datetime.now().isoformat())

    print("About to write to datafile: " + filename + " at " + dt.datetime.now().isoformat())
    json.dump(d, open(filename, 'w'))
    print("Written outputfile at " + dt.datetime.now().isoformat())

if plot:
    # Fourier upsampling

    with open(filename, 'r') as f:
        d = json.load(f)
    print("Loaded previous datafile at " + dt.datetime.now().isoformat())

    d2 = d['output_size=120']

    recon_images = np.zeros((len(etas), len(gammas), 120, 120))
    fourier_diff_images = np.zeros((len(etas), len(gammas), 40, 40))

    for i, eta in enumerate(etas):
        for j, gamma in enumerate(gammas):

            d3 = d2['gamma=0.95']
            d4 = d3['eta=0.01']
            recon = np.asarray(d4['recon'])
            f_diff = np.asarray(d4['fourier_diff'])

            recon_images[i, j, :, :] = np.abs(recon[0] + 1j*recon[1])
            fourier_diff_images[i, j, :, :] = np.abs(f_diff[0] + 1j*f_diff[1])


    f, axarr = plt.subplots(10, 6, figsize=(6, 10))

    for i, eta in enumerate(etas):
        for j, gamma in enumerate(gammas):

            plot = axarr[2*i, j].imshow(recon_images[i, j], vmax=np.amax(recon_images), interpolation='none',
                                 cmap=plt.cm.gray)
            axarr[2 * i, j].axis("off")
            if i == 0:
                axarr[0, j].set_title(r"$\gamma$ = "+str(gamma), fontsize=5, weight="bold")

                if j==-1:
                    f.colorbar(plot, ax=axarr[0, j], shrink=0.6)

            if j == 0:
                axarr[2*i, 0].text(-0.2, 0.5, r"$\eta$ = "+'{:.1e}'.format(eta), fontsize=5, weight="bold",
                                   horizontalalignment='center', verticalalignment='center', rotation=90,
                                   transform=axarr[2*i, 0].transAxes)

            axarr[2*i+1, j].imshow(fourier_diff_images[i, j], vmax=np.amax(fourier_diff_images), interpolation='none',
                                   cmap=plt.cm.gray)
            axarr[2*i+1, j].axis("off")

    f.suptitle("dTV recons with "+r"$\alpha$ = "+ '{:.1e}'.format(alpha), fontsize=10)
    plt.tight_layout(w_pad=0.3, h_pad=0.3, rect=[0, 0, 1, 0.96])
    plt.savefig(save_dir + '/recons_dTV_with_regis_alpha_'+str(alpha)+'.pdf')

    output_dim = 120
    padded_data = np.zeros((output_dim, output_dim), dtype='complex')
    padded_data[(output_dim - 40) // 2:(output_dim + 40) // 2,
    (output_dim - 40) // 2:(output_dim + 40) // 2] = Li_fourier
    Fourier_upsampled_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(padded_data)))