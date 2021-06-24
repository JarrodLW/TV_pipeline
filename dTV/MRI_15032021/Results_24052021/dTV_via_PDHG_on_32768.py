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

save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/' \
               'Experiments/MRI_birmingham/Results_24052021/PDHG_results'

if method == 'TV':
    filename = save_dir + '/TV_'+str(avg)+'_avgs.json'
elif method == 'dTV':
    filename = save_dir + '/dTV_' + str(avg) + '_avgs.json'

if os.path.isfile(filename):

    print("About to read previous datafile: " + filename + " at "+dt.datetime.now().isoformat())
    with open(filename, 'r') as f:
        d = json.load(f)
    print("Loaded previous datafile at "+dt.datetime.now().isoformat())

    f.close()

else:
    print("Could not find: " + filename)
    d = {}

if run_expt:
    alphas = np.linspace(0, 50, num=21)
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
