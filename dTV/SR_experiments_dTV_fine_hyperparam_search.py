# created 23/10. Based on
# Here we use dTV to super-resolve a low-res H image using a high-res H image, 32x32 -> 128x128, automated registration,
# doing a more targeted hyperparameter sweep than before.

import numpy as np
import matplotlib.pyplot as plt
import json
import dTV.myOperators as ops
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
from processing import *
from Utils import *
import dTV.myDeform.linearized as defs

# high resolution H image
fourier_H_real_im = np.reshape(np.fromfile('dTV/MRI_data/fid_H', dtype=np.int32), (128, 256))
fourier_H_real = fourier_H_real_im[:, ::2]
fourier_H_im = fourier_H_real_im[:, 1::2]
fourier_H = fourier_H_real + fourier_H_im*1j
my_recon_H = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_H)))
image = np.abs(my_recon_H)

height, width = image.shape
image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height, width], dtype='float')
image_odl = image_space.element(image)

# downsampling
coarse_image_space_2 = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height//4, width//4], dtype='float')
subsampling_op_2 = ops.Subsampling(image_space, coarse_image_space_2)

# the given raw data for for lower-resolution image
f = open('dTV/MRI_data/1H_lowRes_imaginaryRaw_noZeros', 'r')
fourier_data_im = np.genfromtxt(f, delimiter=' ').T
f = open('dTV/MRI_data/1H_lowRes_realRaw_noZeros', 'r')
fourier_data_real = np.genfromtxt(f, delimiter=' ').T

fourier_data = (fourier_data_real + fourier_data_im*1j)
recon_from_unpacked_data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_data)))
low_res_image = np.abs(recon_from_unpacked_data)
low_res_image_odl = coarse_image_space_2.element(low_res_image)

TV_recon = False
dTV_recon = True

# super-resolving the downsampled images by dTV
if dTV_recon:

    strong_cvx = 1e-2
    niter_prox = 20
    niter = 1000

    alphas = [0.1, 0.2, 0.5, 1., 2., 5., 10., 20.]
    etas = [10**-4, 5*10**-4, 10**-3, 5*10**-3, 10**-2]
    gammas = [0.9, 0.92, 0.94, 0.96, 0.98, 0.995]

    Yaff = odl.tensor_space(6)

    # Create the forward operator
    forward_op = subsampling_op_2

    data_odl = low_res_image_odl
    sinfo = image_odl

    # space of optimised variables
    X = odl.ProductSpace(image_space, Yaff)

    # Set some parameters and the general TV prox options
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = niter_prox

    reg_affine = odl.solvers.ZeroFunctional(Yaff)
    # x0 = X.zero()
    x0 = X.element([subsampling_op_2.adjoint(data_odl), X[1].zero()])

    f = fctls.DataFitL2Disp(X, data_odl, forward_op)

    dTV_regularised_recons = {}
    exp = 0
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:
            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = {}
            for gamma in gammas:

                reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                                    gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                                    prox_options=prox_options)

                g = odl.solvers.SeparableSum(reg_im, reg_affine)

                L = [1, 1e+2]
                ud_vars = [0, 1]

                print('experiment '+str(exp))
                # %%
                palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
                palm.run(niter)

                recon = palm.x[0].asarray()
                affine_params = palm.x[1].asarray()

                dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]['gamma=' + '{:.1e}'.format(gamma)] = recon.tolist()

                exp+=1

    json.dump(dTV_regularised_recons, open('dTV/dTV_regularised_SR_32_to_128_with_regis_refined_hyper_sweep.json', 'w'))