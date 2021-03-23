# created 19/03/2021.
# The purpose of this script is to determine the affine registration parameters needed to register the 1H-image

import numpy as np
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json
import dTV.myAlgorithms as algs
import matplotlib.pyplot as plt
import os
import odl
#import dTV.myOperators as ops
import myOperators as ops
from Utils import *
from skimage.measure import block_reduce
from processing import *
from dTV.myOperators import Embedding_Affine
import dTV.myDeform
from skimage.transform import resize

# loading images
dir_H = 'dTV/MRI_15032021/Data_15032021/H_data/'

f_coeffs = np.reshape(np.fromfile(dir_H +str(6)+'/fid', dtype=np.int32), (128, 256))
f_coeffs = f_coeffs[:, 1::2] + 1j*f_coeffs[:, ::2]
recon_high_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs)))

f_coeffs = np.reshape(np.fromfile(dir_H +str(5)+'/fid', dtype=np.int32), (64, 128))
f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs)
recon_low_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs_unpacked)))

Li_recon = np.load('dTV/MRI_15032021/Results_15032021/example_TV_recon_15032021.npy')
Li_recon_complex = Li_recon[0] + 1j*Li_recon[1]

image_H_high_res = np.abs(recon_high_res)/np.sqrt(np.sum(np.square(np.abs(recon_high_res))))
image_H_low_res = np.abs(recon_low_res)/np.sqrt(np.sum(np.square(np.abs(recon_low_res))))
image_Li = np.abs(Li_recon_complex)/np.sqrt(np.sum(np.square(np.abs(Li_recon_complex))))

plt.figure()
plt.imshow(image_H_high_res, cmap=plt.cm.gray)

plt.figure()
plt.imshow(image_H_low_res, cmap=plt.cm.gray)

plt.figure()
plt.imshow(image_Li, cmap=plt.cm.gray)

## estimating affine registration params

height = 32
width = 32
X = odl.uniform_discr([-1, -1], [1, 1], [height, width], dtype='float32')
#x0 = X.element(image_H_high_res)
x0 = X.element(image_H_low_res)
#TV_regularised_16384_upsampled = resize(image_Li, (128, 128))
#x1 = X.element(TV_regularised_16384_upsampled)
x1 = X.element(image_Li)

# Create a product space for displacement field and a shift space
V = X.tangent_bundle
Y = odl.tensor_space(6)
deform_op = dTV.myDeform.LinDeformFixedTempl(x0)

# Optimisation routine
embed = Embedding_Affine(Y, V)
transl_operator = deform_op * embed

datafit = 0.5 * odl.solvers.L2NormSquared(X).translated(x1)
#datafit = fctls.directionalTotalVariationNonnegative(X, alpha=alpha, sinfo=sinfo,
                                                                          #  gamma=gamma, eta=eta, NonNeg=False, strong_convexity=strong_cvx,
                                                                           # prox_options=prox_options)
f = datafit * transl_operator

ls = 1e-2
cb = (odl.solvers.CallbackPrintIteration(step=1, end=', ') &
      odl.solvers.CallbackPrintTiming(step=1, cumulative=True))

v_recon = Y.zero()
odl.solvers.steepest_descent(f, v_recon, line_search=ls, maxiter=10000,
                             tol=1e-7, callback=cb)

transl_operator(v_recon).show(title='estimated')
x0.show(title='original')
x1.show(title='data')

(x1 - transl_operator(v_recon)).show(title='diff data-est')
(x1 - x0).show(title='diff data-orig')

(x0 - transl_operator(v_recon)).show()

resize(x0.asarray(), (32, 32))
print('Estimated defomation field: ', v_recon)

pre_registered_H_image = transl_operator(v_recon).asarray()
np.save('dTV/MRI_15032021/Results_15032021/pre_registered_H_image_low_res.npy', pre_registered_H_image)

plt.imshow(pre_registered_H_image, cmap=plt.cm.gray)

# registration of higher-res guide image
X_high_res = odl.uniform_discr([-1, -1], [1, 1], [128, 128], dtype='float32')

deform_op_high_res = dTV.myDeform.LinDeformFixedTempl(X_high_res.element(image_H_high_res))
embed_high_res = Embedding_Affine(Y, X_high_res.tangent_bundle)
transl_operator_high_res = deform_op_high_res * embed_high_res

transl_operator_high_res(v_recon).show()
pre_registered_H_image_high_res = transl_operator_high_res(v_recon).asarray()

np.save('dTV/MRI_15032021/Results_15032021/pre_registered_H_image_high_res.npy', pre_registered_H_image_high_res)

plt.figure()
plt.imshow(pre_registered_H_image_high_res, cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(pre_registered_H_image_high_res - image_H_high_res), cmap=plt.cm.gray)

plt.figure()
plt.imshow(image_Li, cmap=plt.cm.gray)

plt.figure()
plt.imshow(image_H_high_res, cmap=plt.cm.gray)

# checking that when downsampled it agrees with the low-res pre-registered H image
downsampled = resize(pre_registered_H_image_high_res, (32, 32))

plt.figure()
plt.imshow(downsampled/np.amax(downsampled), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(pre_registered_H_image/np.amax(pre_registered_H_image), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(pre_registered_H_image/np.amax(pre_registered_H_image) - downsampled/np.amax(downsampled)), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(downsampled/np.amax(downsampled) - image_Li/np.amax(image_Li)), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(pre_registered_H_image/np.amax(pre_registered_H_image) - image_Li/np.amax(image_Li)), cmap=plt.cm.gray)
plt.colorbar()

## registration using dTV
alpha = 10**3
eta = 0.01
gamma = 0.995
strong_cvx = 1e-5
niter_prox = 20
niter = 500

sinfo = image_H_low_res

X = odl.uniform_discr([-1, -1], [1, 1], [sinfo.shape[0], sinfo.shape[1]], dtype='float32')
V = X.tangent_bundle
Y = odl.tensor_space(6)
prod_space = odl.ProductSpace(X, Y)

prox_options = {}
prox_options['name'] = 'FGP'
prox_options['warmstart'] = True
prox_options['p'] = None
prox_options['tol'] = None
prox_options['niter'] = niter_prox

data_odl = X.element(image_Li)
reg_im = fctls.directionalTotalVariationNonnegative(X, alpha=alpha, sinfo=sinfo,
                                                                            gamma=gamma, eta=eta, NonNeg=False, strong_convexity=strong_cvx,
                                                                            prox_options=prox_options)

forward_op = odl.operator.default_ops.IdentityOperator(X)
reg_affine = odl.solvers.ZeroFunctional(Y)
g = odl.solvers.SeparableSum(reg_im, reg_affine)
f = fctls.DataFitL2Disp(prod_space, data_odl, forward_op)

L = [1, 1e+2]
ud_vars = [1] # only doing registration updates

p0 = prod_space.element([data_odl, Y.zero()])
palm = algs.PALM(f, g, ud_vars=ud_vars, x=p0.copy(), callback=None, L=L)
palm.run(niter)

recon = palm.x[0].asarray()
affine_params = palm.x[1].asarray()

deform_op_new = dTV.myDeform.LinDeformFixedTempl(data_odl)
embed_new = Embedding_Affine(Y, V)
transl_operator_new = deform_op_new * embed_new

transl_operator_new(affine_params).show(title='estimated')
data_odl.show()
#X.element(sinfo_low_res).show(title='data')

