#import astra
import h5py
import numpy as np
import matplotlib.pyplot as plt
from processing import *
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json

# pre-processing of data
filename = 'dTV/Experiment1_XRF.hdf5'
f1 = h5py.File(filename, 'r+')

sino_Co = np.array(f1['sino_Co'])
sino_Co_1 = sino_Co[:, :, 0]

filename = 'dTV/Experiment1_XRD.hdf5'
f2 = h5py.File(filename, 'r+')

data_XRD = np.array(f2['sino_XRD'])

## Selecting which recons are to run
FBP_recon = 'True'
TV_recon = 'True'
dTV_recon = 'False'

## pre-processing of data: cumulative sum of hits within specified freq range
data_XRD_0 = data_XRD[:, :, :, 0]
# summing over pixels
plt.figure()
plt.hist(np.sum(data_XRD_0, axis=(0, 1)), range=(0, 4000), bins=200)
plt.show()

# we select the range 450-500, corresponding roughly to the second spectral peak
filter = np.zeros(data_XRD.shape[2])
filter[450:500] = 1
sino_0_XRD = np.dot(data_XRD[:, :, :, 0], filter)

## reconstructions using FBP
if FBP_recon:

    center = 0
    angle_array = 2 * np.pi * np.arange(60) / 60

    recon_XRF_FBP = recon_astra(sino_Co_1.T, center, angles=angle_array, num_iter=200)
    recon_XRD_FBP = recon_astra(sino_0_XRD.T, center, angles=angle_array, num_iter=200)


## TV-regularised reconstructions
if TV_recon:

    model = VariationalRegClass('CT', 'TV')
    a_offset = -np.pi
    a_range = 2*np.pi
    d_offset = 0
    d_width = 40

    reg_params = [10.**(i-5) for i in np.arange(1)]

    TV_regularised_recons = {'XRF': {}, 'XRD': {}}

    for reg_param in reg_params:
        recons_XRF_TV = model.regularised_recons_from_subsampled_data(sino_Co_1.T, reg_param, recon_dims=(561, 561),
                                                                      niter=50, a_offset=a_offset, enforce_positivity=True,
                                                                      a_range=a_range, d_offset=0, d_width=40)[0]

        recons_XRD_TV = model.regularised_recons_from_subsampled_data(sino_0_XRD.T, reg_param, recon_dims=(561, 561),
                                                                      niter=50, a_offset=a_offset, enforce_positivity=True,
                                                                      a_range=a_range, d_offset=0, d_width=40)[0]

        TV_regularised_recons['XRF']['recon_param = '+'{:.1e}'.format(reg_param)] = recons_XRF_TV.tolist()
        TV_regularised_recons['XRD']['recon_param = '+'{:.1e}'.format(reg_param)] = recons_XRD_TV.tolist()


    json.dump(TV_regularised_recons, open('dTV/Results_CT_dTV/TV_regularised_recons.json', 'w'))
    #hf = h5py.File('dTV/Results_CT_dTV/TV_regularised_recons.h5', 'w')


## dTV-regularised XRD recon with XRF sinfo

if dTV_recon:

    gamma = 0.995
    strong_cvx = 1e-2
    niter_prox = 20
    niter = 50

    alphas = [100.]
    etas = [0.01]

    Yaff = odl.tensor_space(6)

    data = sino_0_XRD.T
    height, width = data.shape

    image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[561, 561], dtype='float')
    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, height)
    # Detector: uniformly sampled
    detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, width)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    # Create the forward operator
    forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')

    data_odl = forward_op.range.element(data)

    sinfo = image_space.element(recons_XRF[0])
    #sinfo = recons_XRF[0]

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

            reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
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

            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = recon

    hf = h5py.File('dTV_regularised_recons.h5', 'w')
