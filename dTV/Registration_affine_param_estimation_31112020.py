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


# Define the embedding operator from the parameter space to the space
# of deformation vectorfields

class Embedding_Affine(odl.Operator):
    def __init__(self, space_dom, space_range):
        self.space_affine = space_dom
        self.space_vf = space_range
        super(Embedding_Affine, self).__init__(domain=space_dom,
                                               range=space_range, linear=True)

    def _call(self, inp, out):
        shift = inp[0:2]
        matrix = inp[2:6]
        disp_vf = [
            lambda x: matrix[0] * x[0] + matrix[1] * x[1] + shift[0],
            lambda x: matrix[2] * x[0] + matrix[3] * x[1] + shift[1]]
        v = self.space_vf.element(disp_vf)
        out.assign(v)

    @property
    def adjoint(self):
        op = self

        class AuxOp(odl.Operator):
            def __init__(self):
                super(AuxOp, self).__init__(domain=op.range,
                                            range=op.domain, linear=True)

            def _call(self, phi, out):
                phi0 = phi[0]
                phi1 = phi[1]
                space = phi0.space
                aux_func0 = lambda x: x[0]
                aux_func1 = lambda x: x[1]

                x0 = space.element(aux_func0)
                x1 = space.element(aux_func1)

                mom00 = space.inner(x0, phi0)
                mom10 = space.inner(x1, phi0)
                mom01 = space.inner(x0, phi1)
                mom11 = space.inner(x1, phi1)

                mean0 = space.inner(phi0, space.one())
                mean1 = space.inner(phi1, space.one())

                ret = self.range.element([mean0, mean1,
                                          mom00, mom10, mom01, mom11])
                out.assign(ret)

        return AuxOp()


img = sinfo_low_res/np.sqrt(np.sum(np.square(sinfo_low_res)))

X = odl.uniform_discr([-1, -1], [1, 1], [img.shape[0], img.shape[1]],
                      dtype='float32', interp='linear')
x1 = X.element(img)
#x0 = X.element(image_16384/np.sqrt(np.sum(np.square(image_16384))))
x0 = X.element(TV_regularised_16384/np.sqrt(np.sum(np.square(TV_regularised_16384))))
x1.show()

# Create a product space for displacement field and a shift space
V = X.tangent_bundle
Y = odl.tensor_space(6)
deform_op = odl.deform.LinDeformFixedTempl(x0)


# deformed (target) image
# phi = np.pi/36
# x1 = deform_op(Embedding_Affine(Y, V)([0.2, -0.15, np.cos(phi)-1, -np.sin(phi), np.sin(phi), np.cos(phi)-1]))


# Optimisation routine
embed = Embedding_Affine(Y, V)
transl_operator = deform_op * embed

datafit = 0.5 * odl.solvers.L2NormSquared(X).translated(x1)
f = datafit * transl_operator

ls = 1e-1

cb = (odl.solvers.CallbackPrintIteration(step=1, end=', ') &
      odl.solvers.CallbackPrintTiming(step=1, cumulative=True))

v_recon = Y.zero()
odl.solvers.steepest_descent(f, v_recon, line_search=ls, maxiter=10000,
                             tol=1e-12, callback=cb)

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