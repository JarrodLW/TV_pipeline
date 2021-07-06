# Created 2/07/2021. Running PDHG dTV and TV on synthetic examples

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
import imageio

phantom1 = imageio.imread('dTV/MRI_15032021/Data_15032021/Phantom_data/Phantom_circle_resolution1.png')
#phantom2 = imageio.imread('dTV/MRI_15032021/Data_15032021/Phantom_data/Phantom_circle_resolution2.png')

#
phantom1_120 = resize(phantom1, (120, 120))
#phantom1_80 = resize(phantom1, (80, 80))
#phantom1_40 = resize(phantom1, (40, 40))

#
sinfo_120 = -0.7*phantom1_120**2 + 0.1*phantom1_120 -1
sinfo_80 = resize(sinfo_120, (80, 80))
sinfo_40 = resize(sinfo_120, (40, 40))

# importing fully-averaged data for scale
data = np.load('dTV/MRI_15032021/Results_24052021/32768_data.npy')
#np.sqrt(np.sum(np.square(np.real(data))))
#np.sqrt(np.sum(np.square(np.imag(data))))
data_norm = np.sqrt(np.sum(np.square(np.abs(data))))

# computing synthetic data

complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.], shape=[120, 120], dtype='complex')
image_space = complex_space.real_space ** 2

fourier_transf = ops.RealFourierTransform(image_space)

data_120 = fourier_transf(image_space.element([phantom1_120, np.zeros((120, 120))])).asarray()
data_120 = np.asarray([np.fft.fftshift(data_120[0]), np.fft.fftshift(data_120[1])])
data_40 = np.zeros([2, 40, 40])
data_40[0, :, :] = data_120[0, 40:80, 40:80]
data_40[1, :, :] = data_120[1, 40:80, 40:80]
data_40 *= data_norm/np.sqrt(np.sum(np.square(np.abs(data_40))))
data_40_complex = np.exp(1j*0.75)*(data_40[0] + 1j*data_40[1])
data_40 = np.asarray([np.real(data_40_complex), np.imag(data_40_complex)])

scale = 500
data = data_40 + np.asarray([np.random.normal(loc=0.0, scale=scale, size=(40, 40)),
                             np.random.normal(loc=0.0, scale=scale, size=(40, 40))])
data = np.asarray([np.fft.fftshift(data[0]), np.fft.fftshift(data[1])])

naive_recon = np.fft.fftshift(np.fft.ifft2(data[0] + 1j*data[1]))

plt.figure()
plt.imshow(np.abs(naive_recon), cmap=plt.cm.gray)

GT_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data_40[0]) + 1j*np.fft.fftshift(data_40[1])))

plt.figure()
plt.imshow(np.abs(GT_recon), cmap=plt.cm.gray)

#

#sinfo = sinfo_120
sinfo = sinfo_40
#sinfo = None

if sinfo is None:
    height = width = 120

else:
    height, width = sinfo.shape
#
niter = 300
complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                                      shape=[height, width], dtype='complex', interp='linear')
image_space = complex_space.real_space ** 2

# defining the forward op - I should do the subsampling in a more efficient way
fourier_transf = ops.RealFourierTransform(image_space)
_, data_height, data_width = data.shape

data_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1., 1.],
                                      shape=[data_height, data_width])**2

horiz_ind = np.concatenate((np.sort(list(np.arange(data_height//2))*int(data_width)),
                          np.sort(list(np.arange(height - data_height//2, height))*int(data_width))))
vert_ind = (list(np.arange(data_width//2))+list(np.arange(width - data_width//2, width)))*int(data_height)
sampling_points = [vert_ind, horiz_ind]
emb = misc.Embedding(data_space[0], fourier_transf.range[0], sampling_points=sampling_points, adjoint=None)
subsampling = odl.DiagonalOperator(emb.adjoint, emb.adjoint)
forward_op = subsampling*fourier_transf

data_odl = forward_op.range.element(data)

# building dTV
#gamma = 0.95
gamma = 0.995
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
alpha = 0.5
alpha = 15.
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

obj = f + g*op
cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                  odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                  odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                  odl.solvers.CallbackPrint(obj, fmt='f(x)={0:.4g}') &
                  odl.solvers.CallbackShowConvergence(obj) &
                  odl.solvers.CallbackShow(step=10))

t0 = time()
odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=cb)
t1 = time()
print("Experiment completed in "+ str(t1-t0))

recon = x.asarray()
synth_data = np.asarray([np.fft.fftshift(forward_op(recon).asarray()[0]),
              np.fft.fftshift(forward_op(recon).asarray()[1])])
data_shifted = np.asarray([np.fft.fftshift(data_odl.asarray()[0]),
                           np.fft.fftshift(data_odl.asarray()[1])])
f_diff = synth_data - data_shifted

# downsampled
complex_space_low_res = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                                      shape=[40, 40], dtype='complex', interp='linear')
image_space_low_res = complex_space_low_res.real_space ** 2

# defining the forward op - I should do the subsampling in a more efficient way
fourier_transf_low_res = ops.RealFourierTransform(image_space_low_res)

downsampled = fourier_transf_low_res.inverse(forward_op(recon)).asarray()

plt.figure()
plt.imshow(phantom1_120, cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(recon[0] + 1j*recon[1]), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(downsampled[0] + 1j*downsampled[1]), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(f_diff[0] + 1j*f_diff[1]), cmap=plt.cm.gray)
plt.colorbar()
