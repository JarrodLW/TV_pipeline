## created on 07/01/2021.
## estimating the affine parameters needed to register the H image and Li image (the latter using averge of all data)

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

dir_H = 'dTV/7Li_1H_MRI_Data_31112020/1H_Li2SO4/'

image_H = np.reshape(np.fromfile(dir_H+'6/pdata/1/2dseq', dtype=np.uint16), (128, 128))
# plt.figure()
# plt.imshow(np.abs(image_H), cmap=plt.cm.gray)

sinfo_low_res = block_reduce(image_H.T, block_size=(4, 4), func=np.mean)

dir = 'dTV/7Li_1H_MRI_Data_31112020/'

def unpacking_fourier_coeffs(arr):

    fourier_real_im = arr[:, 1:65]
    fourier_real_im = fourier_real_im[::2, :]

    fourier_real = fourier_real_im[:, 1::2]
    fourier_im = fourier_real_im[:, ::2]
    fourier = fourier_real + fourier_im * 1j

    return fourier

f_coeff_list = []
for i in range(2, 34):
    f_coeffs = np.reshape(np.fromfile(dir + 'Li2SO4/'+str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)

f_coeff_16384 = np.average(np.asarray(f_coeff_list), axis=0)
recon_16384 = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(f_coeff_16384)))
image_16384 = np.abs(recon_16384)

TV_regularised_16384 = np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/Results_MRI_dTV/'
                               'example_TV_recon_Li2SO4_16384_avgs_reg_param_1000.npy').T[:, ::-1]

plt.figure()
plt.imshow(sinfo_low_res, cmap=plt.cm.gray)

plt.figure()
plt.imshow(image_16384, cmap=plt.cm.gray)

## estimating affine registration params

#img = sinfo_low_res/np.sqrt(np.sum(np.square(sinfo_low_res)))
img = image_H.T/np.sqrt(np.sum(np.square(image_H.T)))

X = odl.uniform_discr([-1, -1], [1, 1], [img.shape[0], img.shape[1]], dtype='float32')
x1 = X.element(img)
#x0 = X.element(image_16384/np.sqrt(np.sum(np.square(image_16384))))
TV_regularised_16384_upsampled = resize(TV_regularised_16384, (128, 128))
x0 = X.element(TV_regularised_16384_upsampled/np.sqrt(np.sum(np.square(TV_regularised_16384_upsampled))))
x1.show()

# Create a product space for displacement field and a shift space
V = X.tangent_bundle
Y = odl.tensor_space(6)
deform_op = dTV.myDeform.LinDeformFixedTempl(x0)


# deformed (target) image
# phi = np.pi/36
# x1 = deform_op(Embedding_Affine(Y, V)([0.2, -0.15, np.cos(phi)-1, -np.sin(phi), np.sin(phi), np.cos(phi)-1]))


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
odl.solvers.steepest_descent(f, v_recon, line_search=ls, maxiter=5000,
                             tol=1e-7, callback=cb)

transl_operator(v_recon).show(title='estimated')
x0.show()
x1.show(title='data')

(x1 - transl_operator(v_recon)).show(title='diff data-est')

(x1 - x0).show(title='diff data-orig')

print('Estimated defomation field: ', v_recon)

# bruteforce search
n = 50
angle_arr = np.linspace(-1., 1., num=n)*np.pi/20
#scale_arr = np.exp(np.linspace(0., 0.1, num=n))
scale_arr = [1.]
transl_arr = np.linspace(-1., 1., num=n)*0.1

loss_new = 1.
i=0
for a in angle_arr:
    for b in scale_arr:
        for c in transl_arr:
            for d in transl_arr:
                i+=1

                print("iter_" + str(i))

                affine_params = [c, d, b*np.cos(a)-1, -b*np.sin(a), b*np.sin(a), b*np.cos(a)-1]
                loss_old = loss_new
                loss_new = f(affine_params)

                if loss_new < loss_old:
                    a_best, b_best, c_best, d_best = a, b, c, d


affine_params_best = [c_best, d_best, b_best*np.cos(a_best)-1, -b_best*np.sin(a_best), b_best*np.sin(a_best), b_best*np.cos(a_best)-1]

transl_operator(affine_params_best).show(title='estimated')
x0.show()
x1.show(title='data')

## trying with dTV

alpha = 10**3
eta = 0.01
gamma = 0.995
strong_cvx = 1e-5
niter_prox = 20
niter = 500

#sinfo = sinfo_low_res
sinfo = image_H.T

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

data_odl = X.element(TV_regularised_16384_upsampled)
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
# p0 = prod_space.element([forward_op.adjoint(data_odl), v_recon])
#p0 = prod_space.element([forward_op.adjoint(data_odl), Y.element([1., 1., 0., 0., 0., 0.])]) # should get crazy result


# %%
palm = algs.PALM(f, g, ud_vars=ud_vars, x=p0.copy(), callback=None, L=L)
palm.run(niter)

recon = palm.x[0].asarray()
affine_params = palm.x[1].asarray()


deform_op_new = dTV.myDeform.LinDeformFixedTempl(data_odl)
embed_new = Embedding_Affine(Y, V)
transl_operator_new = deform_op_new * embed_new

transl_operator_new(affine_params).show(title='estimated')
data_odl.show()
X.element(sinfo_low_res).show(title='data')


# some plots
np.abs(transl_operator_new(v_recon) - X.element(sinfo_low_res)).show("diff_sinfo_deformed_recon")
np.abs(data_odl - X.element(sinfo_low_res)).show("diff_sinfo_deformed_recon")
np.abs(transl_operator_new(v_recon) - data_odl).show("diff_recon_deformed_recon")

transl_operator_new(v_recon).show()
data_odl.show()
X.element(sinfo_low_res).show()
transl_operator_new(affine_params).show()

deformed_im = transl_operator_new(v_recon).asarray()
deformed_im_high_res = transl_operator(v_recon).asarray()

plt.figure()
plt.imshow(np.asarray([2*data_odl.asarray()/np.amax(data_odl.asarray()), np.zeros((32, 32)), sinfo_low_res/np.amax(sinfo_low_res)]).transpose((1,2,0)))

plt.figure()
plt.imshow(np.asarray([2*deformed_im/np.amax(deformed_im), np.zeros((32, 32)), sinfo_low_res/np.amax(sinfo_low_res)]).transpose((1,2,0)))

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(np.asarray([TV_regularised_16384_upsampled/np.amax(TV_regularised_16384_upsampled), np.zeros((128, 128)),
                    np.zeros((128, 128))]).transpose((1,2,0)))
axs[1].imshow(np.asarray([TV_regularised_16384_upsampled/np.amax(TV_regularised_16384_upsampled), np.zeros((128, 128)),
                          image_H.T/np.amax(image_H.T)]).transpose((1,2,0)))
axs[2].imshow(np.asarray([np.zeros((128, 128)), np.zeros((128, 128)),
                          image_H.T/np.amax(image_H.T)]).transpose((1,2,0)))

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(np.asarray([deformed_im_high_res/np.amax(deformed_im_high_res), np.zeros((128, 128)),
                          np.zeros((128, 128))]).transpose((1,2,0)))
axs[1].imshow(np.asarray([deformed_im_high_res/np.amax(deformed_im_high_res), np.zeros((128, 128)),
                          image_H.T/np.amax(image_H.T)]).transpose((1,2,0)))
axs[2].imshow(np.asarray([np.zeros((128, 128)), np.zeros((128, 128)),
                          image_H.T/np.amax(image_H.T)]).transpose((1,2,0)))

# checkerboard

from skimage.util import compare_images

TV_image_normalised = TV_regularised_16384_upsampled/np.amax(TV_regularised_16384_upsampled)
comp_1 = compare_images(TV_image_normalised, image_H.T/np.amax(image_H.T), method='checkerboard', n_tiles=(16, 16))

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(TV_image_normalised, cmap=plt.cm.gray)
axs[1].imshow(comp, cmap=plt.cm.gray)
axs[2].imshow(image_H.T/np.amax(image_H.T), cmap=plt.cm.gray)

deformed_im_high_res_normalised = deformed_im_high_res/np.amax(deformed_im_high_res)
comp_2 = compare_images(deformed_im_high_res_normalised, image_H.T/np.amax(image_H.T), method='checkerboard', n_tiles=(16, 16))

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(deformed_im_high_res_normalised, cmap=plt.cm.gray)
axs[1].imshow(comp_2, cmap=plt.cm.gray)
axs[2].imshow(image_H.T/np.amax(image_H.T), cmap=plt.cm.gray)
