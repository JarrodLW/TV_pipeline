# Created: 21/01/2021.
# The purpose of this script is to run dTV experiments with an initialisation coming from TV

import numpy as np
import matplotlib.pyplot as plt
from processing import *
import json
import matplotlib.pyplot as plt
import os
import odl
import myOperators as ops
from Utils import *
import sys
import datetime as dt
from skimage.measure import block_reduce
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs


dir = 'dTV/7Li_1H_MRI_Data_31112020/'
n = int(sys.argv[1]) # 512, 1024, 2048, etc
#dataset = sys.argv[2] # string, has to be either 'Li2SO4' or 'Li_LS'

# These reg params are those at which the GT discrepancy is minimised in TV experiments
init_recon_reg_params = ['{:.1e}'.format(7.4*10**3), '{:.1e}'.format(5.1*10**3), '{:.1e}'.format(3.0*10**3),
                         '{:.1e}'.format(2.1*10**3), '{:.1e}'.format(1.0*10**3)]

init_recon_reg_param = init_recon_reg_params[int(np.log2(n//512))]

dir_H = 'dTV/7Li_1H_MRI_Data_31112020/1H_Li2SO4/'
image_H = np.reshape(np.fromfile(dir_H+'6/pdata/1/2dseq', dtype=np.uint16), (128, 128))

def unpacking_fourier_coeffs(arr):

    fourier_real_im = arr[:, 1:65]
    fourier_real_im = fourier_real_im[::2, :]

    fourier_real = fourier_real_im[:, 1::2]
    fourier_im = fourier_real_im[:, ::2]
    fourier = fourier_real + fourier_im * 1j

    return fourier

f_coeff_list = []

for i in range(2, 34):
    f_coeffs = np.reshape(np.fromfile(dir + dataset +'/'+str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
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

#reg_params = np.concatenate((np.asarray([0.001, 1., 10**0.5, 10., 10**1.5, 10**2]), np.logspace(3., 4.5, num=20)))
#output_dims = [int(32)]
#output_dims = [int(32), int(64)]
f_coeff_list = [f_coeff_list[10]] #### Need to remove this

save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/Experiments/MRI_birmingham/' \
           'Results_MRI_dTV/New/results/TV_results'

with open(save_dir+'/Robustness_31112020_TV_'+str(n)+'_new.json') as f:
    D=json.load(f)

f.close()

sinfo_high_res = image_H.T
sinfo_med_res = block_reduce(sinfo_high_res, block_size=(2, 2), func=np.mean)
sinfo_low_res = block_reduce(sinfo_high_res, block_size=(4, 4), func=np.mean)

sinfos = {}
#sinfos['high_res'] = sinfo_high_res
sinfos['med_res'] = sinfo_med_res
sinfos['low_res'] = sinfo_low_res

#alphas = [50, 10**2, 5*10**2, 10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6]
#alphas = np.logspace(2.5, 4.75, num=20)
#alphas = np.concatenate((np.asarray([0.001, 1., 10**0.5, 10., 10**1.5, 10**2]), np.logspace(2.5, 4.75, num=20)))
alphas = [2.1*10**3]
eta = 0.01

gamma = 0.995
strong_cvx = 1e-5
niter_prox = 20
niter = 200

Yaff = odl.tensor_space(6)
exp = 0

d = {}

for i, Li_fourier in enumerate(f_coeff_list):

    D2 = D['measurement='+str(i)]
    D3 = D2['reg_param='+init_recon_reg_param]

    d['measurement=' + str(i)] = {}

    fourier_data_real = np.real(Li_fourier)
    fourier_data_im = np.imag(Li_fourier)

    #dTV_regularised_recons['measurement=' + str(i)] = {}

    for dict_key in sinfos.keys():

        sinfo = sinfos[dict_key]
        height, width = sinfo.shape

        init_recon = np.asarray(D3['output_size='+height])
        d['measurement=' + str(i)]['output_size=' + str(height)] = {}

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
        x0 = X.element([forward_op.domain.element(init_recon), X[1].zero()])

        f = fctls.DataFitL2Disp(X, data_odl, forward_op)

        for alpha in alphas:
            #sub_dict = d['measurement=' + str(i)]['output_size=' + str(sinfo.shape[0])]

            #if 'alpha=' + '{:.1e}'.format(alpha) not in sub_dict.keys():
             #   d['measurement=' + str(i)]['output_size=' + str(sinfo.shape[0])]['alpha=' + '{:.1e}'.format(alpha)] = {}

            d['measurement=' + str(i)]['output_size=' + str(height)]['alpha=' + '{:.1e}'.format(alpha)] = {}

            print("Experiment_" + str(exp))
            exp += 1

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
            palm.run(niter)

            recon = palm.x[0].asarray()
            diff = forward_op(forward_op.domain.element([recon[0], recon[1]])) - data_odl
            diff = diff[0].asarray() + 1j * diff[1].asarray()
            diff_shift = np.fft.ifftshift(diff)
            diff_shift_subsampled = diff_shift[sinfo.shape[0] // 2 - 16:sinfo.shape[0] // 2 + 16,
                                    sinfo.shape[1] // 2 - 16:sinfo.shape[1] // 2 + 16]

            d['measurement=' + str(i)]['output_size=' + str(height)]['alpha=' + '{:.1e}'.format(alpha)][
                'recon'] = recon.tolist()
            d['measurement=' + str(i)]['output_size=' + str(height)]['alpha=' + '{:.1e}'.format(alpha)][
                'affine_params'] = palm.x[1].asarray().tolist()
            d['measurement=' + str(i)]['output_size=' + str(height)]['alpha=' + '{:.1e}'.format(alpha)][
                'fourier_diff'] = [
                np.real(diff_shift_subsampled).tolist(),
                np.imag(diff_shift_subsampled).tolist()]

outputfile = save_dir + '/New/results/TV_initialised_dTV_results/Robustness_31112020_TV_init_dTV_' + str(n) + '.json'

print("About to write to datafile: " + outputfile + " at " + dt.datetime.now().isoformat())
json.dump(d, open(outputfile, 'w'))
print("Written outputfile at " + dt.datetime.now().isoformat())
