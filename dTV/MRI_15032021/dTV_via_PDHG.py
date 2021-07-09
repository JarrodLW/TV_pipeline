# Created 21/06/2021. Implementation of dTV-guided MRI model, suitable for solution via PDHG.

import odl
import numpy as np
import myOperators as ops
import dTV.myFunctionals as fctls
from skimage.transform import resize
import matplotlib.pyplot as plt
import dTV.Ptycho_XRF_project.misc as misc
from scipy.io import loadmat
from Utils import *
from time import time
import datetime as dt
import sys
import json
import os

method = str(sys.argv[1])
upsample_factor = int(sys.argv[2])
avg = int(sys.argv[3])
date = str(sys.argv[4])

#date='21062021'

run_expt = False
plot = True

# grabbing guide image
image_H_high_res = np.load('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res_filtered.npy')

f_coeff_list = []
dir_Li = 'dTV/MRI_15032021/Data_'+date+'/Li_data/'
Li_range = range(8, 40)
for i in Li_range:
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), (80, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, 40)
    f_coeff_list.append(f_coeffs_unpacked)

full_avg_Fourier_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.average(np.asarray(f_coeff_list), axis=0))))

# plt.figure()
# plt.imshow(np.abs(full_avg_Fourier_recon), cmap=plt.cm.gray)

circular_mask = np.roll(circle_mask(40, 0.43), 1, axis=1)

# plt.figure()
# plt.imshow(circular_mask, cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(circular_mask*np.abs(full_avg_Fourier_recon), cmap=plt.cm.gray)

f_coeff_arr = np.asarray(f_coeff_list)
f_coeff_arr_combined = np.zeros((32, 40, 40), dtype='complex')

avg_ind = int(np.log2(avg/512))
num = 2**avg_ind

for i in range(num):
    data_arr = np.roll(f_coeff_arr, i, axis=0)
    for ele in range(len(f_coeff_list)//num):
        f_coeff_arr_combined[ele+i*len(f_coeff_list)//num, :, :] = np.sum(data_arr[num*ele:num*(ele+1)], axis=0)/num

height, width = (40, 40)
height *= upsample_factor
width *= upsample_factor

if method=='TV':
    sinfo = None

elif method=='dTV':
    sinfo = resize(image_H_high_res, (height, width))

save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/' \
               'Experiments/MRI_birmingham/Results_'+date+'/PDHG_results'

if method == 'TV':
    filename = save_dir + '/TV_'+str(avg)+'_avgs_upsample_factor_'+str(upsample_factor)+'.json'
elif method == 'dTV':
    filename = save_dir + '/dTV_' + str(avg) + '_avgs_upsample_factor_'+str(upsample_factor)+'.json'

if os.path.isfile(filename):

    print("About to read previous datafile: " + filename + " at "+dt.datetime.now().isoformat())
    with open(filename, 'r') as f:
        d = json.load(f)
    print("Loaded previous datafile at "+dt.datetime.now().isoformat())

    f.close()

else:
    print("Could not find: " + filename)
    d = {}

alphas = np.linspace(0, 50, num=21)
if run_expt:
    niter = 2000
    exp = 0
    for i in range(32):
        exp+=1
        print("Experiment number: "+str(exp))

        if 'measurement=' + str(i) not in d.keys():
            d['measurement=' + str(i)] = {}

        Li_fourier = np.fft.fftshift(f_coeff_arr_combined[i, :, :])

        for alpha in alphas:

            if 'output_size=' + str(height) not in d['measurement=' + str(i)].keys():
                d['measurement=' + str(i)]['output_size=' + str(height)] = {}

            if 'reg_param=' + '{:.1e}'.format(alpha) not in d['measurement=' + str(i)]['output_size=' + str(height)].keys():
                d['measurement=' + str(i)]['output_size=' + str(height)]['reg_param='+'{:.1e}'.format(alpha)] = {}

                complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                                      shape=[height, width], dtype='complex', interp='linear')
                image_space = complex_space.real_space ** 2

                # defining the forward op - I should do the subsampling in a more efficient way
                fourier_transf = ops.RealFourierTransform(image_space)
                data_height, data_width = Li_fourier.shape

                data_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1., 1.],
                                                      shape=[data_height, data_width])**2

                horiz_ind = np.concatenate((np.sort(list(np.arange(data_height//2))*int(data_width)),
                                          np.sort(list(np.arange(height - data_height//2, height))*int(data_width))))
                vert_ind = (list(np.arange(data_width//2))+list(np.arange(width - data_width//2, width)))*int(data_height)
                sampling_points = [vert_ind, horiz_ind]
                emb = misc.Embedding(data_space[0], fourier_transf.range[0], sampling_points=sampling_points, adjoint=None)
                subsampling = odl.DiagonalOperator(emb.adjoint, emb.adjoint)
                forward_op = subsampling*fourier_transf

                data_odl = forward_op.range.element([np.real(Li_fourier), np.imag(Li_fourier)])

                # building dTV
                gamma = 0.95
                eta = 0.01
                grad_basic = odl.Gradient(image_space[0], method='forward', pad_mode='symmetric')
                pd = [odl.discr.diff_ops.PartialDerivative(image_space[0], i, method='forward', pad_mode='symmetric') for i in range(2)]
                cp = [odl.operator.ComponentProjection(image_space, i) for i in range(2)]

                if sinfo is None:
                    grad = odl.BroadcastOperator(*[pd[i] * cp[j] for i in range(2) for j in range(2)])

                else:
                    vfield = gamma * fctls.generate_vfield_from_sinfo(sinfo, grad_basic, eta)
                    inner = odl.PointwiseInner(image_space, vfield) * grad_basic
                    grad = odl.BroadcastOperator(*[pd[i] * cp[j] - vfield[i] * inner * cp[j] for i in range(2) for j in range(2)])
                    grad.vfield = vfield

                #alpha = 5.
                #alpha = 15
                #alpha = 20.
                #alpha = 25.
                #alpha = 50.
                semi_norm = alpha*odl.solvers.GroupL1Norm(grad.range, exponent=2)

                forward_op_norm = odl.power_method_opnorm(forward_op)
                grad_norm = odl.power_method_opnorm(grad)
                normalised_forward_op = (1/forward_op_norm)*forward_op
                normalised_grad = (1/grad_norm)*grad
                op = odl.BroadcastOperator(normalised_forward_op, normalised_grad)

                datafit_func = odl.solvers.L2NormSquared(forward_op.range).translated(data_odl)*forward_op_norm

                beta = 1e-5
                f = beta*odl.solvers.L2NormSquared(image_space)
                g = odl.solvers.SeparableSum(datafit_func, semi_norm*grad_norm)

                stepsize_ratio = 100
                op_norm = 1.1 * odl.power_method_opnorm(op)
                tau = np.sqrt(stepsize_ratio) * 1.0 / op_norm  # Step size for the primal variable
                sigma = 1.0 / (np.sqrt(stepsize_ratio) * op_norm)  # Step size for the dual variable

                x = op.domain.zero()

                # obj = f + g*op
                # cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                #                   odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                #                   odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                #                   odl.solvers.CallbackPrint(obj, fmt='f(x)={0:.4g}') &
                #                   odl.solvers.CallbackShowConvergence(obj) &
                #                   odl.solvers.CallbackShow(step=10))

                t0 = time()
                odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=None)
                t1 = time()
                print("Experiment completed in "+ str(t1-t0))

                recon = x.asarray()
                synth_data = np.asarray([np.fft.fftshift(forward_op(recon).asarray()[0]),
                              np.fft.fftshift(forward_op(recon).asarray()[1])])
                data_shifted = np.asarray([np.fft.fftshift(data_odl.asarray()[0]),
                                           np.fft.fftshift(data_odl.asarray()[1])])
                f_diff = synth_data - data_shifted

                d['measurement=' + str(i)]['output_size=' + str(height)]['reg_param=' + '{:.1e}'.format(alpha)][
                    'recon'] = recon.tolist()
                d['measurement=' + str(i)]['output_size=' + str(height)][
                    'reg_param=' + '{:.1e}'.format(alpha)][
                    'synth_data'] = synth_data.tolist()
                d['measurement=' + str(i)]['output_size=' + str(height)]['reg_param=' + '{:.1e}'.format(alpha)][
                    'fourier_diff'] = f_diff.tolist()

    print("About to write to datafile: " + filename + " at " + dt.datetime.now().isoformat())
    json.dump(d, open(filename, 'w'))
    print("Written outputfile at " + dt.datetime.now().isoformat())

if plot:
    TV_fully_averaged = np.load("dTV/MRI_15032021/Results_24052021/example_TV_recon_with_PDHG_on_32768.npy")
    TV_fully_averaged_image = np.abs(TV_fully_averaged[0] + 1j*TV_fully_averaged[1])

    print("About to read datafile: " + filename + " at " + dt.datetime.now().isoformat())
    with open(filename, 'r') as f:
        d = json.load(f)
    print("Loaded datafile at " + dt.datetime.now().isoformat())

    measurements = [0, 5, 10, 15, 20, 25]
    output_size = int(upsample_factor*40)

    # for obtaining downsampled images
    complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                      shape=[40, 40], dtype='complex', interp='linear')
    image_space = complex_space.real_space ** 2

    # defining the forward op - I should do the subsampling in a more efficient way
    fourier_transf = ops.RealFourierTransform(image_space)

    bias_variance_vals = np.zeros((2, len(alphas)))
    masked_bias_variance_vals = np.zeros((2, len(alphas)))
    bias_variance_vals_complex = np.zeros((2, len(alphas)))
    discrepancies = np.zeros((len(alphas), 32))

    for j, alpha in enumerate(alphas):

        recon_images = np.zeros((32, output_size, output_size))
        f_diff_images = np.zeros((32, 40, 40))
        downsampled_recon_images = np.zeros((32, 40, 40))
        fourier_recon_images = np.zeros((32, 40, 40))
        downsampled_recons = np.zeros((32, 2, 40, 40))

        for i in range(32):
            d2 = d['measurement='+str(i)]
            d3 = d2['output_size='+str(output_size)]
            d4 = d3['reg_param='+ '{:.1e}'.format(alpha)]

            recon = np.asarray(d4['recon'])
            recon_image = np.abs(recon[0] + 1j*recon[1])

            f_diff = np.asarray(d4['fourier_diff'])
            f_diff_image = np.abs(f_diff[0] + 1j*f_diff[1])

            #discrep = np.sqrt(np.sum(np.square(f_diff_image)))
            synth_data = np.asarray(d4["synth_data"])
            synth_data_odl = fourier_transf.range.element([np.fft.fftshift(synth_data[0]), np.fft.fftshift(synth_data[1])])
            #downsampled_recon = np.fft.fftshift(np.fft.ifft2(synth_data[0] + 1j*synth_data[1]))
            downsampled_recon = fourier_transf.inverse(synth_data_odl).asarray()
            downsampled_recon_image = np.abs(downsampled_recon[0] + 1j*downsampled_recon[1])

            data = f_coeff_arr_combined[i, :, :]
            fourier_recon = np.fft.fftshift(np.fft.ifft2(data))
            fourier_recon_image = np.abs(fourier_recon)

            recon_images[i, :, :] = recon_image
            f_diff_images[i, :, :] = f_diff_image
            downsampled_recon_images[i, :, :] = downsampled_recon_image
            downsampled_recons[i, 0, :, :] = downsampled_recon[0]
            downsampled_recons[i, 1, :, :] = downsampled_recon[1]
            fourier_recon_images[i, :, :] = fourier_recon_image

            discrepancies[j, i] = np.sqrt(np.sum(np.square(f_diff_image)))

        f, axarr = plt.subplots(6, 6, figsize=(6, 6))

        if sinfo is None:
            sinfo = np.zeros((height, width))

        for i, measurement in enumerate(measurements):
            axarr[i, 0].imshow(sinfo, cmap=plt.cm.gray, interpolation='None')
            axarr[i, 0].axis("off")
            axarr[i, 1].imshow(fourier_recon_images[measurement], cmap=plt.cm.gray, interpolation='None')
            axarr[i, 1].axis("off")
            axarr[i, 2].imshow(TV_fully_averaged_image, cmap=plt.cm.gray, interpolation='None')
            axarr[i, 2].axis("off")
            axarr[i, 3].imshow(recon_images[measurement], cmap=plt.cm.gray, interpolation='None')
            axarr[i, 3].axis("off")
            axarr[i, 4].imshow(downsampled_recon_images[measurement], cmap=plt.cm.gray, interpolation='None')
            axarr[i, 4].axis("off")
            pcm = axarr[i, 5].imshow(f_diff_images[measurement], cmap=plt.cm.gray, interpolation='None')
            axarr[i, 5].axis("off")

        axarr[0, 0].set_title("Guide")
        axarr[0, 1].set_title("Fourier")
        axarr[0, 2].set_title("Target")
        axarr[0, 3].set_title("dTV")
        axarr[0, 4].set_title("Subs.")
        axarr[0, 5].set_title("Resid.")
        f.colorbar(pcm, ax=[axarr[-1, -1]], shrink=0.75)
        #plt.tight_layout()
        plt.savefig(save_dir+"/"+method+"_results/"+str(avg)+"_avgs/upsample_factor_"+str(upsample_factor)+"_reg_param_" + '{:.1e}'.format(alpha)+".pdf")
        plt.close()

        print("number of recon images: "+str(recon_images.shape[0]))

        average_recon_image = np.average(downsampled_recon_images, axis=0)
        stdev_image = np.sqrt(np.average((downsampled_recon_images - average_recon_image)**2, axis=0))
        bias_image = average_recon_image - TV_fully_averaged_image

        plt.imshow(stdev_image, cmap=plt.cm.gray)
        plt.colorbar()
        plt.savefig(save_dir + "/" + method + "_results/" + str(avg) + "_avgs/stdev_image_upsample_factor_" + str(
            upsample_factor) + "_reg_param_" + '{:.1e}'.format(alpha) + ".pdf")
        plt.close()

        plt.imshow(bias_image, cmap=plt.cm.gray)
        plt.colorbar()
        plt.savefig(save_dir + "/" + method + "_results/" + str(avg) + "_avgs/stdev_image_upsample_factor_" + str(
            upsample_factor) + "_reg_param_" + '{:.1e}'.format(alpha) + ".pdf")
        plt.close()

        variance = np.sum(stdev_image**2)
        bias = np.sqrt(np.sum(np.square(bias_image)))

        masked_variance = np.sum(circular_mask*stdev_image**2)
        masked_bias = np.sqrt(np.sum(np.square(circular_mask*bias_image)))

        # average_recon = np.average(downsampled_recons, axis=0)
        # variance_complex = np.average(np.sum(np.abs(downsampled_recons - average_recon) ** 2, axis=(1, 2, 3)))
        # bias_complex = np.sqrt(np.sum(np.square(average_recon - np.asarray([np.real(full_avg_Fourier_recon),
        #                                                                     np.imag(full_avg_Fourier_recon)]))))

        norm_averaged_recon_image = np.sqrt(np.sum(np.square(average_recon_image)))
        norm_TV_image = np.sqrt(np.sum(np.square(TV_fully_averaged_image)))
        print("norm of averaged recon image: "+str(norm_averaged_recon_image))
        print("norm of GT: " + str(norm_TV_image))

        bias_variance_vals[0, j] = bias
        bias_variance_vals[1, j] = variance

        masked_bias_variance_vals[0, j] = masked_bias
        masked_bias_variance_vals[1, j] = masked_variance

        # bias_variance_vals_complex[0, j] = bias_complex
        # bias_variance_vals_complex[1, j] = variance_complex

    np.save(save_dir+"/"+method+"_results/"+str(avg)+"_avgs/upsample_factor_"+str(upsample_factor)+"/"+method+"_upsample_factor_"+str(upsample_factor)+"_bias_variance_"+str(avg)+"_avgs.npy", bias_variance_vals)
    np.save(save_dir + "/" + method + "_results/" + str(avg) + "_avgs/upsample_factor_"+str(upsample_factor)+"/" + method + "_upsample_factor_" + str(
        upsample_factor) + "_masked_bias_variance_" + str(avg) + "_avgs.npy", masked_bias_variance_vals)
    np.save(save_dir + "/" + method + "_results/" + str(avg) + "_avgs/upsample_factor_"+str(upsample_factor)+"/" + method + "_upsample_factor_" + str(
        upsample_factor) + "_discrepancies_" + str(avg) + "_avgs.npy", discrepancies)
    # np.save(save_dir + "/" + method + "_results/" + str(avg) + "_avgs/" + method + "_upsample_factor_" + str(
    #     upsample_factor) + "_bias_variance_complex_" + str(avg) + "_avgs.npy", bias_variance_vals_complex)
    #np.save(save_dir+"/"+method+"_results/"+str(avg)+"_avgs/"+method+"_upsample_factor_"+str(upsample_factor)+"_discrepancies_"+str(avg)+"_avgs.npy", discrepancies)
