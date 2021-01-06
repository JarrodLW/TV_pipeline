#import astra
import h5py
import numpy as np
import matplotlib.pyplot as plt
from processing import *
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json

# pre-processing of data
filename = 'dTV/CT_data/Experiment2_XRF.hdf5'
f1 = h5py.File(filename, 'r+')

slice_num = 40
sino_Co = np.array(f1['sino_Co'])
sino_Co_1 = sino_Co[:, :, slice_num]

cumulative_sum = np.zeros(1024)
# histogram of intensities
for i in range(31):
    filename = 'dTV/CT_data/Experiment2_XRD_projection_'+"{:02d}".format(i)+'.hdf5'
    f = h5py.File(filename, 'r+')
    #cumulative_sum += np.sum(f['map'][:, slice_num, :], axis=0)
    cumulative_sum += np.sum(f['map'], axis=(0, 1))

plt.figure()
plt.hist(cumulative_sum, range=(0, np.amax(cumulative_sum)), bins=200)

window_endpoints = (200, 2000)
filter = np.zeros(cumulative_sum.shape[0])
filter[window_endpoints[0] : window_endpoints[1]] = 1

plt.figure()
plt.hist(cumulative_sum, range=window_endpoints, bins=20)

sino_XRD = np.zeros(sino_Co_1.shape)
for i in range(31):
    filename = 'dTV/CT_data/Experiment2_XRD_projection_'+"{:02d}".format(i)+'.hdf5'
    f = h5py.File(filename, 'r+')
    sino_XRD_single_proj = np.dot(f['map'][:, slice_num], filter)

    sino_XRD[:, i] = sino_XRD_single_proj

plt.hist(sino_XRD)

# plt.figure()
# plt.imshow(sino_Co_1, cmap=plt.cm.gray, aspect=0.1)
# plt.colorbar()
#
# plt.figure()
# plt.imshow(sino_XRD, cmap=plt.cm.gray, aspect=0.1, vmin=2.5, vmax=3.2)
# plt.colorbar()
#
# plt.figure()
# plt.hist(sino_Co_1.tolist(), range=(0, 7000), bins=20)
#
# plt.figure()
# plt.hist(sino_XRD.tolist(), range=(0, 18), bins=20)
#
# plt.figure()
# plt.imshow(sino_XRD>2.7, cmap=plt.cm.gray, aspect=0.1)
#
# plt.figure()
# plt.imshow(sino_XRD>5, cmap=plt.cm.gray, aspect=0.1)

# reconstructions using FBP
height=width=230
image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                            shape=[height, width], dtype='float')

a_offset = 0
a_range = np.pi
d_offset = 0
d_width = 40

# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, 31)
# Detector: uniformly sampled
detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, width)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')
FBP = odl.tomo.fbp_op(forward_op)

recon_XRF = FBP(forward_op.range.element(sino_Co_1.T))
recon_XRD = FBP(forward_op.range.element(sino_XRD.T))

plt.figure()
plt.imshow(recon_XRF, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recon_XRD, cmap=plt.cm.gray)

# pdhg using new functional
from odl_implementation_CT_KL import CTKullbackLeibler

max_intens = 1.
beta = 0.05
data = max_intens*np.exp(-beta*forward_op.range.element(sino_XRD.T))
#data_odl = forward_op.range(data)

op_norm = 1.1 * odl.power_method_opnorm(forward_op)
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable
niter = 200

g = CTKullbackLeibler(forward_op.range, prior=data, max_intens=max_intens)

#f = odl.solvers.ZeroFunctional(image_space)
#f = 0.01*odl.solvers.L2NormSquared(image_space)
f = odl.solvers.IndicatorNonnegativity(image_space)
x = image_space.zero()

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                      odl.solvers.CallbackShow(step=10))

odl.solvers.pdhg(x, f, g, forward_op, niter=niter, tau=tau, sigma=sigma, callback=cb)

## TV recon

TV_recon = "True"

sino_XRF_non_backround_vals = np.nonzero(np.where(np.ndarray.flatten(sino_Co_1)>500, np.ndarray.flatten(sino_Co_1), 0))
moment_ratio_XRF = np.std(sino_XRF_non_backround_vals)/np.mean(sino_XRF_non_backround_vals)

percentile = 80
sino_XRD_thresh = (sino_XRD>np.percentile(sino_XRD, percentile))*sino_XRD
sino_XRD_thresh_nonzero_vals = np.nonzero(np.ndarray.flatten(sino_XRD_thresh))
moment_ratio_XRD = np.std(sino_XRD_thresh_nonzero_vals)/np.mean(sino_XRD_thresh_nonzero_vals)

percentile_lower = 80
percentile_higher = 99
#subsampling_arr=((sino_XRD>np.percentile(sino_XRD, percentile_lower))*(sino_XRD<np.percentile(sino_XRD, percentile_higher)) | (sino_Co_1<500))*np.ones(sino_XRD.shape)

subsampling_arr=((sino_XRD>np.percentile(sino_XRD, percentile_lower))*(sino_XRD<np.percentile(sino_XRD, percentile_higher)) + 100*(sino_Co_1<500))*np.ones(sino_XRD.shape)
#subsampling_arr = (sino_Co_1<500)*np.ones(sino_XRD.shape)

#sino_XRF_normalised = sino_Co_1/np.amax(sino_Co_1)
#sino_XRD_normalised = sino_XRD/np.amax(sino_XRD)
#subsampling_arr = np.exp(-np.square(sino_XRD_normalised - sino_XRF_normalised))

## TV-regularised reconstructions
if TV_recon:

    model = VariationalRegClass('CT', 'TV')
    a_offset = 0
    a_range = np.pi
    d_offset = 0
    d_width = 40

    reg_params = [10.**(-1)]

    TV_regularised_recons = {'XRF': {}, 'XRD': {}}

    for reg_param in reg_params:
        recons_XRD_TV = model.regularised_recons_from_subsampled_data((sino_XRD * subsampling_arr).T, reg_param,
                                                                      recon_dims=(230, 230),
                                                                      subsampling_arr=subsampling_arr.T,
                                                                      niter=500, a_offset=a_offset,
                                                                      enforce_positivity=True,
                                                                      a_range=a_range, d_offset=d_offset,
                                                                      d_width=d_width)

        recons_XRF_TV = model.regularised_recons_from_subsampled_data((sino_Co_1*subsampling_arr).T, reg_param, recon_dims=(230, 230), subsampling_arr=subsampling_arr.T,
                                                                      niter=500, a_offset=a_offset, enforce_positivity=True,
                                                                      a_range=a_range, d_offset=d_offset, d_width=d_width)


recon_XRF_TV = recons_XRF_TV[0]
recon_XRD_TV = recons_XRD_TV[0]

plt.figure()
plt.imshow(recon_XRF_TV, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recon_XRD_TV, cmap=plt.cm.gray)

forward_op(image_space.element(recon_XRD_TV)).show()

plt.figure()
plt.imshow(recon_XRF.asarray(), cmap=plt.cm.gray)

plt.figure()
plt.imshow(sino_XRD, cmap=plt.cm.gray, aspect=0.1)

plt.figure()
plt.imshow(sino_Co_1, cmap=plt.cm.gray, aspect=0.1)

plt.figure()
plt.imshow(sino_XRD*subsampling_arr, cmap=plt.cm.gray, aspect=0.1)

plt.figure()
plt.imshow(sino_Co_1, cmap=plt.cm.gray, aspect=0.1)
plt.colorbar()

## dTV-regularised XRD recon with XRF sinfo
dTV_recon = 'True'
if dTV_recon:

    gamma = 0.995
    niter_prox = 20
    niter = 100

    strong_cvxs = [1e-2]

    alphas = [10.**(-1)]
    etas = [10.**(-3)]

    Yaff = odl.tensor_space(6)

    data = sino_XRD.T
    #data = sino_XRD
    height, width = data.shape

    image_space = odl.uniform_discr(min_pt=[-d_width//2, -d_width//2], max_pt=[d_width//2, d_width//2], shape=[230, 230], dtype='float')
    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, height)
    # Detector: uniformly sampled
    detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, width)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    # Create the forward operator
    forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')

    data_odl = forward_op.range.element(data)

    # rotating so that orientations match
    recon_XRF_rotated = recon_XRF_TV.T[:, ::-1]
    sinfo = image_space.element(recon_XRF_rotated)

    # space of optimised variables
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

    dTV_regularised_recons = {}
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:
            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = {}
            for strong_cvx in strong_cvxs:

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

                dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]['strong_cvx=' + '{:.1e}'.format(strong_cvx)] = recon.tolist()


## inpainting of sinograms
sino_space = odl.uniform_discr(min_pt=[0, 0], max_pt=[1, 1], shape=sino_XRD.shape, dtype='float')

if dTV_recon:

    gamma = 0.995
    strong_cvx = 1e-2
    niter_prox = 20
    niter = 100

    alphas = [1.]
    etas = [10**(-5)]

    Yaff = odl.tensor_space(6)

    # Create the forward operator
    forward_op = odl.operator.default_ops.IdentityOperator(sino_space)

    data_odl = sino_space.element(sino_XRD)
    sinfo = sino_space.element(sino_Co_1)

    # space of optimised variables
    X = odl.ProductSpace(sino_space, Yaff)

    reg_affine = odl.solvers.ZeroFunctional(Yaff)
    x0 = X.zero()

    f = fctls.DataFitL2Disp(X, data_odl, forward_op)

    for alpha in alphas:
        for eta in etas:

            reg_im = fctls.directionalTotalVariationNonnegative(sino_space, alpha=alpha, sinfo=sinfo,
                                                                gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                                prox_options=prox_options)

            g = odl.solvers.SeparableSum(reg_im, reg_affine)

            L = [1, 1e+2]
            ud_vars = [0]

            # %%
            palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), L=L, callback=cb)
            palm.run(niter)

            recon = palm.x[0].asarray()

FBP(forward_op.range.element(recon)).show()



# dTV with subsampling operator

dTV_recon = 'True'
if dTV_recon:

    gamma = 0.995
    niter_prox = 20
    niter = 100

    strong_cvxs = [1e-4]

    alphas = [10.**(-1)]
    etas = [10.**(-3)]

    Yaff = odl.tensor_space(6)

    #data = (sino_XRD.T)*(subsampling_arr.T)
    data = forward_op(image_space.element(recon_XRD_TV)).asarray()*(subsampling_arr.T)
    #data = sino_XRD
    height, width = data.shape

    image_space = odl.uniform_discr(min_pt=[-d_width//2, -d_width//2], max_pt=[d_width//2, d_width//2], shape=[230, 230], dtype='float')
    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, height)
    # Detector: uniformly sampled
    detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, width)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    # Create the forward operator
    pre_forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')
    forward_op = pre_forward_op.range.element(subsampling_arr.T)*pre_forward_op

    data_odl = forward_op.range.element(data)

    # rotating so that orientations match
    recon_XRF_rotated = recon_XRF_TV.T[:, ::-1]
    sinfo = image_space.element(recon_XRF_rotated)

    # space of optimised variables
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

    dTV_regularised_recons = {}
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:
            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = {}
            for strong_cvx in strong_cvxs:

                reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=None,
                                                                    gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                                    prox_options=prox_options)

                g = odl.solvers.SeparableSum(reg_im, reg_affine)

                cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                      odl.solvers.CallbackShow(step=5))

                L = [1, 1e+2]
                ud_vars = [0]

                # %%
                palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
                palm.run(niter)

                recon = palm.x[0].asarray()


