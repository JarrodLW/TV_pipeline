# created on 16/11/2020. Based on "SR_experiments_dTV_ _avgs.py" file and "Inverse_problem_dTV_22102020"

import numpy as np
import matplotlib.pyplot as plt
import json
import dTV.myOperators as ops
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
from processing import *
from Utils import *
import dTV.myDeform.linearized as defs
from skimage.measure import block_reduce

dir = 'dTV/7LI_1H_MRI_Data_22102020/'

image_H = np.reshape(np.fromfile(dir+'1mm_1H_high_res/2dseq', dtype=np.uint16), (128, 128))
#plt.figure()
#plt.imshow(image_H, cmap=plt.cm.gray)

#image_Li = np.reshape(np.fromfile(dir+'1mm_7Li_512_avgs/2dseq', dtype=np.uint16), (32, 32))
#plt.figure()
#plt.imshow(image_Li, cmap=plt.cm.gray)

# files from Bearshare folder, labelled 7Li_Axial_512averages_1mmslicethickness etc.
fourier_Li_real_im_padded = np.reshape(np.fromfile(dir + '1mm_7Li_8192_avgs/fid', dtype=np.int32), (64, 128))
fourier_Li_real_im = fourier_Li_real_im_padded[:, 1:65]
fourier_Li_real_im = fourier_Li_real_im[::2, :]

fourier_data_real = fourier_Li_real_im[:, 1::2]
fourier_data_im = fourier_Li_real_im[:, ::2]
fourier_Li = fourier_Li_real + fourier_Li_im*1j

gamma = 0.995
strong_cvx = 1e-2
niter_prox = 20
niter = 500

alphas = [10000.]
etas = [0.01]

Yaff = odl.tensor_space(6)

sinfo_high_res = image_H.T
sinfo_med_res = block_reduce(sinfo_high_res, block_size=(2, 2), func=np.mean)
sinfo_low_res = block_reduce(sinfo_high_res, block_size=(4, 4), func=np.mean)

sinfos = {}
#sinfos['high_res'] = sinfo_high_res
sinfos['med_res'] = sinfo_med_res
#sinfos['low_res'] = sinfo_low_res

# running dTV
dTV_recon = True

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
            palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
            print('Experiment '+'alpha='+str(alpha)+' eta='+str(eta))

            palm.run(niter)

            recon = palm.x[0].asarray()


with open('dTV/Results_MRI_dTV/dTV_regularised_SR_8192_avgs_22102020.json') as f:
    d = json.load(f)

d2 = d['32_to_64']
