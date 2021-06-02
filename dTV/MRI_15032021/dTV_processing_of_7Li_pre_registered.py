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

date = '24052021'
#date = '15032021'

dir_Li = 'dTV/MRI_15032021/Data_' + date + '/Li_data/'
#dir_H = 'dTV/MRI_15032021/Data_' + date + '/H_data/'

if date=='15032021':
    low_res_shape = (64, 128)
    Li_range = range(3, 35)
    low_res_data_width = 32

    image_H_low_res = np.load('dTV/MRI_15032021/Results_15032021/pre_registered_H_image_low_res.npy')
    image_H_high_res = np.load('dTV/MRI_15032021/Results_15032021/pre_registered_H_image_high_res.npy')
    image_H_med_res = resize(image_H_high_res, (64, 64))

elif date=='24052021':
    low_res_shape = (80, 128)
    Li_range = range(8, 40)
    low_res_data_width = 40
    image_H_high_res = np.load('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res.npy')
    image_H_med_res = resize(image_H_high_res, (80, 80))
    image_H_low_res = resize(image_H_high_res, (40, 40))

n = int(sys.argv[1]) # 512, 1024, 2048, etc
#n = 512

f_coeff_list = []

for i in Li_range:
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), low_res_shape)
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, low_res_data_width)
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

#f_coeff_list = [f_coeff_list[0]]  # DELETE THIS!!!!

sinfos = {}
sinfos['high_res'] = image_H_high_res
sinfos['med_res'] = image_H_med_res
sinfos['low_res'] = image_H_low_res

alphas = np.concatenate((np.asarray([0.001, 1., 10**0.5, 10., 10**1.5, 10**2]), np.logspace(2.5, 4.75, num=20)))
#alphas = np.asarray([8000.]) # DELETE THIS!
eta = 0.01
gamma = 0.995
strong_cvx = 1e-5
niter_prox = 20
niter = 100

Yaff = odl.tensor_space(6)
exp = 0

save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/' \
           'Experiments/MRI_birmingham/Results_'+date+'/dTV_results_pre_registered'
run_exp = True

outputfile = save_dir + '/dTV_7Li_15032021_' + str(n) + '_pre_registered.json'

if os.path.isfile(outputfile):

    print("About to read previous datafile: " + outputfile + " at "+dt.datetime.now().isoformat())
    with open(outputfile, 'r') as f:
        d = json.load(f)
    print("Loaded previous datafile at "+dt.datetime.now().isoformat())

    f.close()

else:
    print("Could not find: " + outputfile)
    d = {}


for i, Li_fourier in enumerate(f_coeff_list):

    fourier_data_real = np.real(Li_fourier)
    fourier_data_im = np.imag(Li_fourier)

    if 'measurement=' + str(i) not in d.keys():
        d['measurement=' + str(i)] = {}

    #d['measurement=' + str(i)] = {}

    for dict_key in sinfos.keys():

        sinfo = sinfos[dict_key]

        if 'output_size=' + str(sinfo.shape[0]) not in d['measurement=' + str(i)].keys():
            d['measurement=' + str(i)]['output_size=' + str(sinfo.shape[0])] = {}

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

        for alpha in alphas:

            if 'alpha=' + '{:.1e}'.format(alpha) not in d['measurement=' + str(i)]['output_size=' + str(sinfo.shape[0])].keys():
            #dTV_regularised_recons['measurement=' + str(i)]['output_size=' + str(sinfo.shape[0])]['alpha=' + '{:.1e}'.format(alpha)] = {}
                d['measurement=' + str(i)]['output_size=' + str(sinfo.shape[0])]['alpha=' + '{:.1e}'.format(alpha)] = {}

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
                ud_vars = [0]

                # %%
                palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
                #palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
                palm.run(niter)

                print("end: " + dt.datetime.now().isoformat())

                recon = palm.x[0].asarray()
                diff = forward_op(forward_op.domain.element([recon[0], recon[1]])) - data_odl
                diff = diff[0].asarray() + 1j * diff[1].asarray()
                diff_shift = np.fft.ifftshift(diff)
                diff_shift_subsampled = diff_shift[sinfo.shape[0] // 2 - 16:sinfo.shape[0] // 2 + 16,
                                        sinfo.shape[1] // 2 - 16:sinfo.shape[1] // 2 + 16]

                d['measurement=' + str(i)]['output_size=' + str(sinfo.shape[0])]['alpha=' + '{:.1e}'.format(alpha)]['recon'] = recon.tolist()
                d['measurement=' + str(i)]['output_size=' + str(sinfo.shape[0])]['alpha=' + '{:.1e}'.format(alpha)]['affine_params'] = palm.x[1].asarray().tolist()
                d['measurement=' + str(i)]['output_size=' + str(sinfo.shape[0])]['alpha=' + '{:.1e}'.format(alpha)]['fourier_diff'] = [
                    np.real(diff_shift_subsampled).tolist(),
                    np.imag(diff_shift_subsampled).tolist()]

print("About to write to datafile: " + outputfile + " at " + dt.datetime.now().isoformat())
json.dump(d, open(outputfile, 'w'))
print("Written outputfile at " + dt.datetime.now().isoformat())
