# created 28/09/20
# Aim: to super-resolve Li reconstruction using high-res H reconstruction

import numpy as np
import skimage
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json
import matplotlib.pyplot as plt
import os
import odl
import dTV.myOperators as ops
from Utils import *
import dTV.myDeform.linearized as defs
from skimage.transform import resize
from skimage.measure import block_reduce
from time import time


fourier_H_real_im = np.reshape(np.fromfile('dTV/Results_MRI_dTV/fid_H', dtype=np.int32), (128, 256))

# for some reason the fid_7 data comes padded: all even rows are zero and the first and 65-128th columns are zeros
# once the zeros are removed, you get a 32x64 array that can then be unpacked into real-im parts as before
fourier_Li_real_im_padded = np.reshape(np.fromfile('dTV/Results_MRI_dTV/fid_Li_actual', dtype=np.int32), (64, 128))
fourier_Li_real_im = fourier_Li_real_im_padded[:, 1:65]
fourier_Li_real_im = fourier_Li_real_im[::2, :]

fourier_H_real = fourier_H_real_im[:, ::2]
fourier_H_im = fourier_H_real_im[:, 1::2]
fourier_H = fourier_H_real + fourier_H_im*1j

# I've exchanged the real and imaginary parts as compared with "MRI_experiments_dTV_2.py"; this is now consistent with
# the fact that the two reconstructions seemed to be misaligned by 180 degrees.
fourier_Li_real = fourier_Li_real_im[:, 1::2]
fourier_Li_im = fourier_Li_real_im[:, ::2]
fourier_Li = fourier_Li_real + fourier_Li_im*1j

my_recon_H = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_H)))
my_recon_Li = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_Li)))

H_image = np.abs(my_recon_H)
Li_image = np.abs(my_recon_Li)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
axes[0].imshow(H_image, cmap=plt.cm.gray)
axes[1].imshow(Li_image, cmap=plt.cm.gray)
fig.tight_layout()

height, width = Li_image.shape
sinfo_height, sinfo_width = H_image.shape

disp_vert = np.zeros(H_image.shape)
disp_horiz = np.zeros(H_image.shape)


image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[sinfo_height, sinfo_width], dtype='float')
coarse_image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height, width], dtype='float')
subsampling_op = ops.Subsampling(image_space, coarse_image_space)
upsampled_Li_image_1 = subsampling_op.adjoint(Li_image).asarray()
upsampled_Li_image_2 = resize(Li_image, (sinfo_height, sinfo_width))

disp_field_space = image_space.tangent_bundle

#can make this much more efficient!!!
for i in range(sinfo_height):
    for j in range(sinfo_width):

        disp_vert[i, j] = (1/sinfo_height)*(i - sinfo_height//2)*(3/2)
        disp_horiz[i, j] = (1/sinfo_width)*(j - sinfo_width//2)*(3/2)

displacement = -(1/2)*np.asarray([disp_vert, disp_horiz])

displ_upsampled_image_1 = defs.linear_deform(image_space.element(upsampled_Li_image_1), disp_field_space.element(displacement))
displ_upsampled_image_2 = defs.linear_deform(image_space.element(upsampled_Li_image_2), disp_field_space.element(displacement))


# doing the same at the coarse resolution
disp_vert = np.zeros(Li_image.shape)
disp_horiz = np.zeros(Li_image.shape)

for i in range(height):
    for j in range(width):

        disp_vert[i, j] = (1/height)*(i - height//2)*(3/2)
        disp_horiz[i, j] = (1/width)*(j - width//2)*(3/2)

displacement = -(1/2)*np.asarray([disp_vert, disp_horiz])
displ_Li_image = defs.linear_deform(coarse_image_space.element(Li_image), coarse_image_space.tangent_bundle.element(displacement))


dTV_recon = True

# super-resolving the downsampled images by dTV
if dTV_recon:

    gamma = 0.995
    strong_cvx = 1e-2
    niter_prox = 20
    niter = 1000

    #alphas = [10.**(i-5) for i in np.arange(10)]
    #etas = [10.**(-i) for i in np.arange(6)]
    alphas = [1.]
    etas = [10**(-5)]

    Yaff = odl.tensor_space(6)

    # Create the forward operator
    forward_op = subsampling_op

    data_odl = coarse_image_space.element(displ_Li_image)
    sinfo = image_space.element(H_image)

    # space of optimised variables
    # X = odl.ProductSpace(coarse_image_space_1, Yaff)
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
    x0 = X.element([subsampling_op.adjoint(data_odl), X[1].zero()])

    f = fctls.DataFitL2Disp(X, data_odl, forward_op)

    dTV_regularised_recons = {}
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:

            reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                                gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                                prox_options=prox_options)

            g = odl.solvers.SeparableSum(reg_im, reg_affine)

            cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                  odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                  odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                  odl.solvers.CallbackShow(step=10))

            L = [1, 1e+2]
            # ud_vars = [0]
            ud_vars = [0, 1]

            # %%
            palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
            palm.run(niter)

            recon = palm.x[0].asarray()
            affine_params = palm.x[1].asarray()





## linesearch for registration
reg_func = fctls.directionalTotalVariationNonnegative(corase_image_space, alpha=1, sinfo=image_space.element(H_image))
n=20

dTV_losses = np.zeros((n, n, n))

for i, phi in enumerate(np.linspace(-0.05, 0.05, num=n)):
    for j, shift_vert in enumerate(np.linspace(-0.1, 0.1, num=n)):
        for k, shift_hor in enumerate(np.linspace(-0.1, 0.1, num=n)):
            cosp = np.cos(phi)
            sinp = np.sin(phi)
            disp_func = [
                lambda x: (cosp-1)*x[0] - sinp*x[1] + shift_vert,
                lambda x: sinp*x[0] + (cosp-1)*x[1] + shift_hor]

            deformed_im = defs.linear_deform(image_space.element(displ_upsampled_image_2),
                                             disp_field_space.element(disp_func))
            dTV_loss = reg_func(deformed_im)

            dTV_losses[i, j, k] = dTV_loss

i_best, j_best, k_best = np.unravel_index(np.argmin(dTV_losses), dTV_losses.shape)
phi = np.linspace(-0.05, 0.05, num=n)[i_best]
shift_vert = np.linspace(-0.1, 0.1, num=n)[j_best]
shift_hor = np.linspace(-0.1, 0.1, num=n)[k_best]

cosp = np.cos(phi)
sinp = np.sin(phi)
disp_func = [
    lambda x: (cosp-1)*x[0] - sinp*x[1] + shift_vert,
    lambda x: sinp*x[0] + (cosp-1)*x[1] + shift_hor]

deformed_im_best = defs.linear_deform(image_space.element(displ_upsampled_image_2), disp_field_space.element(disp_func))
dTV_loss = reg_func(deformed_im_best)

plt.figure()
plt.imshow(displ_upsampled_image_2)

plt.figure()
plt.imshow(deformed_im_best)

plt.figure()
plt.imshow(np.asarray([np.roll(deformed_im_best/np.amax(deformed_im_best), (0,0)), np.zeros(deformed_im_best.shape),
                       H_image/np.amax(H_image)]).transpose((1, 2, 0)), vmin=0, vmax=0.2)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5, 3))
plt.figure()
axes[0].imshow(np.asarray([1.5*np.roll(displ_upsampled_image_2/np.amax(displ_upsampled_image_2), (0,0)), np.zeros(displ_upsampled_image_2.shape),
                       np.zeros(displ_upsampled_image_2.shape)]).transpose((1, 2, 0)))
axes[1].imshow(np.asarray([1.5*np.roll(displ_upsampled_image_2/np.amax(displ_upsampled_image_2), (0,0)), np.zeros(displ_upsampled_image_2.shape),
                       1.5*H_image/np.amax(H_image)]).transpose((1, 2, 0)))
axes[2].imshow(np.asarray([np.zeros(displ_upsampled_image_2.shape), np.zeros(displ_upsampled_image_2.shape),
                       1.5*H_image/np.amax(H_image)]).transpose((1, 2, 0)))
fig.tight_layout()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5, 3))
plt.figure()
axes[0].imshow(np.asarray([1.5*np.roll(upsampled_Li_image_2/np.amax(upsampled_Li_image_2), (0,0)), np.zeros(upsampled_Li_image_2.shape),
                       np.zeros(upsampled_Li_image_2.shape)]).transpose((1, 2, 0)))
axes[1].imshow(np.asarray([1.5*np.roll(upsampled_Li_image_2/np.amax(upsampled_Li_image_2), (0,0)), np.zeros(upsampled_Li_image_2.shape),
                       1.5*H_image/np.amax(H_image)]).transpose((1, 2, 0)))
axes[2].imshow(np.asarray([np.zeros(upsampled_Li_image_2.shape), np.zeros(upsampled_Li_image_2.shape),
                       1.5*H_image/np.amax(H_image)]).transpose((1, 2, 0)))
fig.tight_layout()

# dTV_losses = np.zeros((n, n, n, n, n, n))
#
# t0 = time()
# for i0, a in enumerate(np.linspace(-0.05, 0.05, num=n)):
#     for i1, b in enumerate(np.linspace(-0.05, 0.05, num=n)):
#         for i2, c in enumerate(np.linspace(-0.05, 0.05, num=n)):
#             for i3, d in enumerate(np.linspace(-0.05, 0.05, num=n)):
#                 for i4, e in enumerate(np.linspace(-0.05, 0.05, num=n)):
#                     for i5, f in enumerate(np.linspace(-0.05, 0.05, num=n)):
#
#                         disp_func = [
#                             lambda x: a * x[0] + b * x[1] + e,
#                             lambda x: c * x[0] + d * x[1] + f]  # 0.02 corresponds to 1 pixel
#
#                         deformed_im = defs.linear_deform(image_space.element(upsampled_Li_image_2), disp_field_space.element(disp_func))
#                         dTV_loss = reg_func(deformed_im)
#
#                         dTV_losses[i0, i1, i2, i3, i4, i5] = dTV_loss
#
# t1 = time()
# print(t1-t0)
#
# i0_best, i1_best, i2_best, i3_best, i4_best, i5_best = np.unravel_index(np.argmin(dTV_losses), dTV_losses.shape)
# a = np.linspace(-0.05, 0.05, num=n)[i0_best]
# b = np.linspace(-0.05, 0.05, num=n)[i1_best]
# c = np.linspace(-0.05, 0.05, num=n)[i2_best]
# d = np.linspace(-0.05, 0.05, num=n)[i3_best]
# e = np.linspace(-0.05, 0.05, num=n)[i4_best]
# f = np.linspace(-0.05, 0.05, num=n)[i5_best]
#
# disp_func = [
#     lambda x: a * x[0] + b * x[1] + e,
#     lambda x: c * x[0] + d * x[1] + f]  # 0.02 corresponds to 1 pixel
#
# deformed_im_best = defs.linear_deform(image_space.element(upsampled_Li_image_2), disp_field_space.element(disp_func))
# dTV_loss = reg_func(deformed_im_best)



plt.figure()
plt.imshow(displ_upsampled_image_1, cmap=plt.cm.gray)

plt.figure()
plt.imshow(upsampled_Li_image_2, cmap=plt.cm.gray)

plt.figure()
plt.imshow(displ_upsampled_image_2, cmap=plt.cm.gray)

plt.figure()
plt.imshow(H_image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(Li_image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(Li_image.T[::-1, :], cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.asarray([np.roll(2*displ_upsampled_image_2/np.amax(displ_upsampled_image_1), (0,0)), np.zeros(displ_upsampled_image_1.shape),
                       H_image/np.amax(H_image)]).transpose((1, 2, 0)), vmin=0, vmax=0.2)

plt.figure()
plt.imshow(np.asarray([np.roll(2*displ_upsampled_image_2/np.amax(displ_upsampled_image_1), (-5,0), axis=(0,1)), np.zeros(displ_upsampled_image_1.shape),
                       H_image/np.amax(H_image)]).transpose((1, 2, 0)), vmin=0, vmax=0.2)


# trying to align the reconstructions
subsampling_op = ops.Subsampling(image_space, coarse_image_space)
downsampled_H_image_odl = subsampling_op(image_space.element(H_image))
downsampled_H_image = downsampled_H_image_odl.asarray()

reg_func = fctls.directionalTotalVariationNonnegative(coarse_image_space, alpha=1, sinfo=downsampled_H_image_odl)

phi = -np.pi/4
shift_vert = 0
shift_hor = 0
cosp = np.cos(phi)
sinp = np.sin(phi)
disp_func = [
    lambda x: (cosp-1)*x[0] - sinp*x[1] + shift_vert,
    lambda x: sinp*x[0] + (cosp-1)*x[1] + shift_hor]

rotated_im = defs.linear_deform(coarse_image_space.element(displ_Li_image), coarse_image_space.tangent_bundle.element(disp_func))

plt.figure()
plt.imshow(np.asarray([np.roll(2*rotated_im/np.amax(rotated_im), (0, 0)), np.zeros(rotated_im.shape),
                       downsampled_H_image/np.amax(downsampled_H_image)]).transpose((1, 2, 0)), vmin=0, vmax=0.2)

plt.figure()
plt.imshow(np.asarray([np.roll(2*displ_Li_image/np.amax(displ_Li_image), (0, 0)), np.zeros(displ_Li_image.shape),
                       downsampled_H_image/np.amax(downsampled_H_image)]).transpose((1, 2, 0)), vmin=0, vmax=0.2)

plt.figure()
plt.imshow(displ_Li_image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(rotated_im, cmap=plt.cm.gray)

reg_func(displ_Li_image)

reg_func(rotated_im)