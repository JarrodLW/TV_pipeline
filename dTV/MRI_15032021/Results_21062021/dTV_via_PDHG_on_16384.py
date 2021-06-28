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

data = np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_21062021/16384_data.npy')
height = width = 40

# calculating Morozov noise level
fourier_recon = np.fft.fftshift(np.fft.ifft2(data))

plt.imshow(np.abs(fourier_recon), cmap=plt.cm.gray)

window = fourier_recon[:10, 30:]
Morozov_level = 40*4*np.sqrt(np.sum(np.square(np.abs(window))))

# TV-regularised recons

alphas = np.linspace(0, 10, num=11)
niter = 500
exp = 0

d = {}

sinfo=None
filename = 'dTV/MRI_15032021/Results_21062021/TV_with_PDHG_on_16384.npy'

for alpha in alphas:
    exp += 1
    print("Experiment number: " + str(exp))

    d['reg_param=' + '{:.1e}'.format(alpha)] = {}

    complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                          shape=[height, width], dtype='complex', interp='linear')
    image_space = complex_space.real_space ** 2

    # defining the forward op - I should do the subsampling in a more efficient way
    fourier_transf = ops.RealFourierTransform(image_space)
    data_height, data_width = data.shape

    data_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1., 1.],
                                          shape=[data_height, data_width])**2

    horiz_ind = np.concatenate((np.sort(list(np.arange(data_height//2))*int(data_width)),
                              np.sort(list(np.arange(height - data_height//2, height))*int(data_width))))
    vert_ind = (list(np.arange(data_width//2))+list(np.arange(width - data_width//2, width)))*int(data_height)
    sampling_points = [vert_ind, horiz_ind]
    emb = misc.Embedding(data_space[0], fourier_transf.range[0], sampling_points=sampling_points, adjoint=None)
    subsampling = odl.DiagonalOperator(emb.adjoint, emb.adjoint)
    forward_op = subsampling*fourier_transf

    data_odl = forward_op.range.element([np.real(data), np.imag(data)])

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

    obj = f + g*op
    cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                      odl.solvers.CallbackPrint(obj, fmt='f(x)={0:.4g}') &
                      odl.solvers.CallbackShowConvergence(obj) &
                      odl.solvers.CallbackShow(step=25))

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

    d['reg_param=' + '{:.1e}'.format(alpha)][
        'recon'] = recon.tolist()
    d['reg_param=' + '{:.1e}'.format(alpha)][
        'synth_data'] = synth_data.tolist()
    d['reg_param=' + '{:.1e}'.format(alpha)][
        'fourier_diff'] = f_diff.tolist()

print("About to write to datafile: " + filename + " at " + dt.datetime.now().isoformat())
json.dump(d, open(filename, 'w'))
print("Written outputfile at " + dt.datetime.now().isoformat())

# plotting

print("About to read datafile: " + filename + " at " + dt.datetime.now().isoformat())
with open(filename, 'r') as f:
    d = json.load(f)
print("Loaded datafile at " + dt.datetime.now().isoformat())

f.close()

rec_images = np.zeros((11, 40, 40))
f_diff_images = np.zeros((11, 40, 40))
discreps = np.zeros(11)

for i, alpha in enumerate(alphas):
    rec = np.asarray(d['reg_param=' + '{:.1e}'.format(alpha)]['recon'])
    rec_image = np.abs(rec[0] + 1j*rec[1])
    f_diff = np.asarray(d['reg_param=' + '{:.1e}'.format(alpha)]['fourier_diff'])
    f_diff_image = np.abs(f_diff[0] + 1j*f_diff[1])
    discrep = np.sqrt(np.sum(np.square(f_diff_image)))

    rec_images[i, :, :] = rec_image
    f_diff_images[i, :, :] = f_diff_image
    discreps[i] = discrep

f, axarr = plt.subplots(4, 5, figsize=(10, 8))
for i, alpha in enumerate(alphas[:-1]):

    rec_image = rec_images[i, :, :]
    f_diff_image = f_diff_images[i, :, :]
    discrep_ratio = discreps[i]/Morozov_level

    axarr[2 *(i // 5), i%5].imshow(rec_image, cmap=plt.cm.gray)
    axarr[2 * (i // 5), i % 5].axis("off")
    axarr[2 * (i // 5), i % 5].set_title("discrep:" + '{:.1e}'.format(discrep_ratio))
    axarr[1 + 2 * (i // 5), i % 5].imshow(f_diff_image, vmax=np.amax(f_diff_images), cmap=plt.cm.gray)
    axarr[1 + 2 * (i // 5), i % 5].axis("off")

plt.tight_layout()
plt.savefig("dTV/MRI_15032021/Results_21062021/TV_with_PDHG_on_16384_plot.png")

example_rec = np.asarray(d['reg_param=' + '{:.1e}'.format(alphas[5])]['recon'])
example_f_diff = np.asarray(d['reg_param=' + '{:.1e}'.format(alphas[5])]['fourier_diff'])
example_synth_data = np.asarray(d['reg_param=' + '{:.1e}'.format(alphas[5])]['synth_data'])

np.save("dTV/MRI_15032021/Results_21062021/example_TV_recon_with_PDHG_on_16384.npy", example_rec)
np.save("dTV/MRI_15032021/Results_21062021/example_TV_recon_with_PDHG_on_16384_synth_data.npy", example_synth_data)
