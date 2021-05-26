# Created: 30/03/2021.
# Using a better-registered H image

import numpy as np
import json
import matplotlib.pyplot as plt
import os
import odl
import myOperators as ops
from Utils import *
import dTV.myAlgorithms as algs
import dTV.myFunctionals as fctls
import datetime as dt
from skimage.transform import resize, rescale

dir = 'dTV/MRI_15032021/Data_24052021/Li_data/'
n = 1024

image_H_high_res = np.load('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res.npy')
image_H_med_res = resize(image_H_high_res, (80, 80))
image_H_low_res = resize(image_H_high_res, (40, 40))

plt.figure()
plt.imshow(image_H_high_res, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(image_H_low_res, cmap=plt.cm.gray)
plt.colorbar()

f_coeff_list = []

for i in range(8, 40):
    f_coeffs = np.reshape(np.fromfile(dir + str(i) + '/fid', dtype=np.int32), (80, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, 40)
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

f_coeff_list = [f_coeff_list[0]]

sinfos = {}
sinfos['high_res'] = image_H_high_res
#sinfos['med_res'] = image_H_med_res
#sinfos['low_res'] = image_H_low_res

alphas = np.concatenate((np.asarray([0.001, 1., 10**0.5, 10., 10**1.5, 10**2]), np.logspace(2.5, 4.75, num=20)))
alphas = alphas[10: 20]
eta = 0.01
gamma = 0.995
strong_cvx = 1e-5
niter_prox = 20
niter = 100

Yaff = odl.tensor_space(6)
exp = 0

save_dir = 'dTV/MRI_15032021/Results_24052021'
run_exp = True

outputfile = save_dir + '/dTV_7Li_24052021_' + str(n) + '_pre_registered.json'

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

# plotting
d = json.load(open('dTV/MRI_15032021/Results_24052021/dTV_7Li_24052021_1024_pre_registered.json', 'r'))

#d2 = d['measurement=0']
#d3 = d2['output_size=80']
#d4 = d3['alpha=3.7e+03']
#d4 = d3['alpha=4.8e+03']
#recon_small_im = rescale(recon_im, 0.33, anti_aliasing=False)

fully_averaged = np.load('dTV/MRI_15032021/Results_24052021/fully_averaged_Li_recon.npy')
TV_reg_fully_averaged = np.load('dTV/MRI_15032021/Results_24052021/example_TV_reg_Li_fully_averaged_lambda_1000.npy')

SSIM_vals = np.zeros((32, len(alphas)))
SSIM_of_normalised = np.zeros((32, len(alphas)))
rel_l2_errors = np.zeros((32, len(alphas)))
rel_l2_errors_of_normalised = np.zeros((32, len(alphas)))
rel_l1_errors = np.zeros((32, len(alphas)))

for j, alpha in enumerate(alphas):
    for measurement in range(1):

        d2 = d['measurement='+str(measurement)]
        d3 = d2['output_size=80']
        d4 = d3['alpha='+'{:.1e}'.format(alpha)]
        recon = np.asarray(d4['recon'])
        recon_im = np.abs(recon[0] + 1j*recon[1])
        recon_small_im = resize(recon_im, (40, 40))
        recon_small_im_normalised = recon_small_im / np.sqrt(np.sum(np.square(recon_small_im)))
        TV_reg_fully_averaged_normalised = TV_reg_fully_averaged / np.sqrt(np.sum(np.square(TV_reg_fully_averaged)))
        SSIM_vals[measurement, j] = recon_error(recon_small_im, TV_reg_fully_averaged)[2]
        SSIM_of_normalised[measurement, j] = recon_error(recon_small_im_normalised,
                      TV_reg_fully_averaged_normalised)[2]
        rel_l2_errors[measurement, j] = recon_error(recon_small_im, TV_reg_fully_averaged)[0]\
                                        /np.sqrt(np.sum(np.square(TV_reg_fully_averaged)))
        rel_l2_errors_of_normalised[measurement, j] = recon_error(recon_small_im_normalised, TV_reg_fully_averaged_normalised)[0]
        rel_l1_errors[measurement, j] = np.sum(np.abs(recon_small_im - TV_reg_fully_averaged))/np.sum(np.abs(TV_reg_fully_averaged))

fig, axs = plt.subplots(4, 5, figsize=(15, 6))
#for k, ax in enumerate(axs.flat):
for i in range(2):
    for j in range(5):

        d2 = d['measurement=' + str(0)]
        d3 = d2['output_size=80']
        d4 = d3['alpha=' + '{:.1e}'.format(alphas[5*i + j])]
        recon = np.asarray(d4['recon'])
        f_diff = np.asarray(d4['fourier_diff'])
        f_diff_abs = np.abs(f_diff[0] + 1j*f_diff[1])
        recon_im = np.abs(recon[0] + 1j * recon[1])
        recon_small_im = resize(recon_im, (40, 40))

        pcm = axs[2*i, j].imshow(recon_small_im, cmap=plt.cm.gray)
        axs[2*i, j].axis("off")

        # pcm = axs[2 * i, j].imshow(recon_im, cmap=plt.cm.gray)
        # axs[2 * i, j].axis("off")

        pcm = axs[2*i + 1, j].imshow(f_diff_abs, cmap=plt.cm.gray)
        axs[2*i + 1, j].axis("off")

plt.tight_layout()



plt.figure()
plt.plot(np.log10(alphas), np.average(SSIM_vals, axis=0))

plt.figure()
plt.plot(np.log10(alphas), np.average(SSIM_of_normalised, axis=0))

plt.figure()
plt.plot(np.log10(alphas), np.average(rel_l2_errors, axis=0))

plt.figure()
plt.plot(np.log10(alphas), np.average(rel_l2_errors_of_normalised, axis=0))

plt.figure()
plt.plot(np.log10(alphas), np.average(rel_l1_errors, axis=0))

# SSIM_2 = recon_error(recon_small_im / np.sqrt(np.sum(np.square(recon_small_im))),
#                      TV_reg_fully_averaged / np.sqrt(np.sum(np.square(TV_reg_fully_averaged))))[2]
# SSIM_of_normalised.append(SSIM_2)

plt.figure()
plt.imshow(np.abs(f_diff[0] + 1j*f_diff[1]), cmap=plt.cm.gray)

plt.figure()
plt.imshow(image_H_med_res, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recon_im, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recon_small_im, cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(fully_averaged), cmap=plt.cm.gray)

plt.figure()
plt.imshow(TV_reg_fully_averaged, cmap=plt.cm.gray)

SSIM = recon_error(recon_small_im, TV_reg_fully_averaged)[2]
SSIM_2 = recon_error(recon_small_im/np.sqrt(np.sum(np.square(recon_small_im))),
            TV_reg_fully_averaged/np.sqrt(np.sum(np.square(TV_reg_fully_averaged))))[2]
SSIM_3 = recon_error(recon_small_im/np.sqrt(np.sum(np.square(recon_small_im))),
                     image_H_low_res/np.sqrt(np.sum(np.square(image_H_low_res))))[2]
SSIM_4 = recon_error(TV_reg_fully_averaged/np.sqrt(np.sum(np.square(TV_reg_fully_averaged))),
                     image_H_low_res / np.sqrt(np.sum(np.square(image_H_low_res))))[2]


