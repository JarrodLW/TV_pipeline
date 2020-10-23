# created 02/10/20
# Aim: process noisy Li data resulting from relatively few averages

import numpy as np
import matplotlib.pyplot as plt
from processing import *
import dTV.myAlgorithms as algs
import matplotlib.pyplot as plt
import os
import odl
import dTV.myFunctionals as fctls
import myOperators as ops

avgs = ['512', '1024', '4096', '8192']
Li_images = []

for avg in avgs:
    # files from Bearshare folder, labelled 7Li_Axial_512averages_1mmslicethickness etc.
    fourier_Li_real_im_padded = np.reshape(np.fromfile('dTV/MRI_data/fid_Li_'+avg+'_averages', dtype=np.int32), (64, 128))
    fourier_Li_real_im = fourier_Li_real_im_padded[:, 1:65]
    fourier_Li_real_im = fourier_Li_real_im[::2, :]

    fourier_Li_real = fourier_Li_real_im[:, 1::2]
    fourier_Li_im = fourier_Li_real_im[:, ::2]
    fourier_Li = fourier_Li_real + fourier_Li_im*1j

    my_recon_Li = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_Li)))
    Li_image = np.abs(my_recon_Li)
    Li_images.append(Li_image)

for Li_image in Li_images:

    plt.figure()
    plt.imshow(Li_image, cmap=plt.cm.gray)

reg_types = ['TV']
reg_params = [10**40]

for reg_type in reg_types:
    model = VariationalRegClass('MRI', reg_type)

    for reg_param in reg_params:

        recons_bernoulli = model.regularised_recons_from_subsampled_data(np.fft.fftshift(fourier_Li), reg_param, niter=10000)

recon_rotated = np.abs(recons_bernoulli)[0].T[:, ::-1]

plt.figure()
plt.imshow(Li_image, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(recon_rotated, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.fft.fftshift(recon_rotated), cmap=plt.cm.gray)

np.sum(np.square(Li_image - recon_rotated))

## Trying dTV script instead

alpha = 10**5
eta = 0.01
gamma = 0.
strong_cvx = 1e-2
niter_prox = 20
niter = 200

Yaff = odl.tensor_space(6)

height, width = fourier_Li.shape
complex_space = odl.uniform_discr(min_pt=[-height//2, -width//2], max_pt=[height//2, width//2],
                                      shape=[height, width], dtype='complex', interp='linear')
image_space = complex_space.real_space ** 2
forward_op = ops.RealFourierTransform(image_space)

data_fft = np.fft.fftshift(fourier_Li)
#data = np.array([fourier_Li_real, fourier_Li_im])
data = np.array([np.real(data_fft), np.imag(data_fft)])
data_odl = forward_op.range.element(data)

X = odl.ProductSpace(image_space, Yaff)

# Set some parameters and the general TV prox options
prox_options = {}
prox_options['name'] = 'FGP'
prox_options['warmstart'] = True
prox_options['p'] = None
prox_options['tol'] = None
prox_options['niter'] = niter_prox

reg_affine = odl.solvers.ZeroFunctional(Yaff)
x0 = X.zero()

f = fctls.DataFitL2Disp(X, data_odl, forward_op)

reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=None,
                                                    gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                    prox_options=prox_options)

g = odl.solvers.SeparableSum(reg_im, reg_affine)

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True))

L = [1, 1e+2]
ud_vars = [0]

# %%
palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
palm.run(niter)

recon = palm.x[0].asarray()

plt.figure()
plt.imshow(Li_image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(recon[0]+1j*recon[1]), cmap=plt.cm.gray)
