#import astra
import h5py
import numpy as np
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
FBP_recon = 'False'
TV_recon = 'False'
masked_TV_recon = True
dTV_recon = 'False'
plotting = 'False'

## pre-processing of data: cumulative sum of hits within specified freq range
data_XRD_0 = data_XRD[:, :, :, 0]
# summing over pixels
plt.figure()
plt.hist(np.sum(data_XRD_0, axis=(0, 1)), range=(0, 4000), bins=200)
plt.title("Spectrum obtained by summing over pixels")
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

    np.save('recon_XRF_FBP', recon_XRF_FBP)
    np.save('recon_XRD_FBP', recon_XRD_FBP)


## TV-regularised reconstructions
if TV_recon:

    model = VariationalRegClass('CT', 'TV')
    a_offset = -np.pi
    a_range = 2*np.pi
    d_offset = 0
    #d_width = 40
    d_width = 2

    #reg_params = [10.**(i-5) for i in np.arange(10)]
    #reg_params = [10**100]
    reg_params = [10**2]

    TV_regularised_recons = {'XRF': {}, 'XRD': {}}

    for reg_param in reg_params:
        recons_XRF_TV = model.regularised_recons_from_subsampled_data(sino_Co_1.T, reg_param, recon_dims=(561, 561),
                                                                      niter=50, a_offset=a_offset, enforce_positivity=True,
                                                                      a_range=a_range, d_offset=0, d_width=d_width)[0]

        recons_XRD_TV = model.regularised_recons_from_subsampled_data(sino_0_XRD.T, reg_param, recon_dims=(561, 561),
                                                                      niter=50, a_offset=a_offset, enforce_positivity=True,
                                                                      a_range=a_range, d_offset=0, d_width=d_width)[0]

        TV_regularised_recons['XRF']['recon_param = '+'{:.1e}'.format(reg_param)] = recons_XRF_TV.tolist()
        TV_regularised_recons['XRD']['recon_param = '+'{:.1e}'.format(reg_param)] = recons_XRD_TV.tolist()


    json.dump(TV_regularised_recons, open('dTV/Results_CT_dTV/TV_regularised_recons.json', 'w'))
    #hf = h5py.File('dTV/Results_CT_dTV/TV_regularised_recons.h5', 'w')


## TV-regularised XRD reconstructions, masking out the spikes

if masked_TV_recon:


    model = VariationalRegClass('CT', 'TV')
    a_offset = -np.pi
    a_range = 2 * np.pi
    d_offset = 0
    d_width = 2

    # figuring out what to mask

    width, height = sino_0_XRD.shape

    image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[561, 561], dtype='float')
    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(a_offset, a_offset + a_range, height)
    # Detector: uniformly sampled
    detector_partition = odl.uniform_partition(d_offset - d_width / 2, d_offset + d_width / 2, width)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')
    FBP = odl.tomo.fbp_op(forward_op)

    data = (sino_0_XRD < 0.1)*sino_0_XRD
    #data = background_minus_buffer[:, :-1]*sino_0_XRD
    #data = background[:, :-1] * sino_0_XRD

    recon_XRD = FBP(forward_op.range.element(data.T))

    # optimisation

    #reg_param = 1. #10. ** 0
    #reg_params = [10. ** (i - 5) for i in np.arange(20)]

    reg_params = [10**-7]

    subsampling_arr =  (sino_0_XRD < 0.18)*np.ones(sino_0_XRD.shape)
    masked_data = 60000*sino_0_XRD*subsampling_arr # I don't remember whether or not I have to do this explicitly.... check!
    # background = sino_Co_1 < 800
    # background_shifted_down = np.roll(background, 5, axis=0)
    # background_shifted_up = np.roll(background, -5, axis=0)
    # background_minus_buffer = background*background_shifted_down*background_shifted_up
    # masked_data = (1-background_minus_buffer)[:, :-1]*masked_data # this is ad-hoc!

    # recons_XRD_TV = model.regularised_recons_from_subsampled_data(masked_data.T, reg_param, recon_dims=(561, 561), subsampling_arr=((1-background_minus_buffer)[:, :-1]*subsampling_arr).T,
    #                                                               niter=50, a_offset=a_offset, enforce_positivity=True,
    #                                                               a_range=a_range, d_offset=0, d_width=40)[0]

    TV_regularised_recons = {}
    exp=0

    for reg_param in reg_params:
        exp += 1
        print("Experiment " + str(exp))

        recons_XRD_TV = model.regularised_recons_from_subsampled_data(masked_data.T, reg_param, recon_dims=(561, 561),
                                                                      subsampling_arr=subsampling_arr.T,
                                                                      niter=50, a_offset=a_offset,
                                                                      enforce_positivity=True,
                                                                      a_range=a_range, d_offset=0, d_width=d_width)[0]

        TV_regularised_recons['recon_param = ' + '{:.1e}'.format(reg_param)] = recons_XRD_TV.tolist()

    json.dump(TV_regularised_recons, open('dTV/Results_CT_dTV/TV_regularised_recons_XRD_with_masking.json', 'w'))

    # recons_XRD_TV = model.regularised_recons_from_subsampled_data(masked_data.T, reg_param, recon_dims=(561, 561),
    #                                                               subsampling_arr=None,
    #                                                               niter=50, a_offset=a_offset, enforce_positivity=True,
    #                                                               a_range=a_range, d_offset=0, d_width=40)[0]
    with open('dTV/Results_CT_dTV/TV_regularised_recons_XRD_with_masking.json') as f:
        d = json.load(f)

    f.close()

    fig, axs = plt.subplots(5, 4, figsize=(8, 6))
    for i, reg_param in enumerate(reg_params):

        recon = np.asarray(d['recon_param = ' + '{:.1e}'.format(reg_param)])
        axs[i // 4, i % 4].imshow(recon, cmap=plt.cm.gray)
        axs[i // 4, i % 4].axis("off")





## dTV-regularised XRD recon with XRF sinfo

if dTV_recon:

    gamma = 0.995
    #strong_cvx = 1e-2
    #strong_cvx = 1e-1
    niter_prox = 20
    #niter = 250
    niter = 100

    #strong_cvxs = [1e1, 1e0, 1e-1, 1e-2, 1e-3]
    strong_cvxs = [1e-5]

    #alphas = [10.**(i-5) for i in np.arange(4, 8)]
    #etas = [10.**(-i) for i in np.arange(5)]
    alphas = [10.**(-2)]
    etas = [10.**(-4)]

    Yaff = odl.tensor_space(6)

    data = sino_0_XRD.T
    #data = sino_Co_1.T
    height, width = data.shape

    image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[561, 561], dtype='float')
    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, height)
    # Detector: uniformly sampled
    detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, width)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    subsampling_arr = (sino_0_XRD < 0.3) * np.ones(sino_0_XRD.shape)
    masked_data = (subsampling_arr.T)*data

    # Create the forward operator
    forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')
    subsampled_forward_op = forward_op.range.element(subsampling_arr.T)*forward_op

    data_odl = forward_op.range.element(masked_data)

    # remembering to rotate so that orientations match
    with open('dTV/Results_CT_dTV/TV_regularised_recons.json') as f:
        d = json.load(f)

    recon_XRF = np.asarray(d['XRF']['recon_param = 1.0e-02']).T[:, ::-1]
    sinfo = image_space.element(recon_XRF)
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

    f = fctls.DataFitL2Disp(X, data_odl, subsampled_forward_op)

    dTV_regularised_recons = {}
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:
            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = {}
            for strong_cvx in strong_cvxs:

                reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                                    gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                                    prox_options=prox_options)

                g = odl.solvers.SeparableSum(reg_im, reg_affine)

                cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                      odl.solvers.CallbackShow()
                      )

                L = [1, 1e+2]
                ud_vars = [0]

                # %%
                palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
                palm.run(niter)

                recon = palm.x[0].asarray()

                dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]['strong_cvx=' + '{:.1e}'.format(strong_cvx)] = recon.tolist()

    json.dump(dTV_regularised_recons, open('dTV/Results_CT_dTV/dTV_regularised_recons_varying_strong_cvx.json', 'w'))



if plotting:

    ## turning arrays into images
    imaging_types = ['XRF', 'XRD']

    for imaging_type in imaging_types:
        for reg_param in reg_params:

            recon = np.asarray(TV_regularised_recons[imaging_type]['recon_param = '+'{:.1e}'.format(reg_param)])

            plt.figure()
            plt.imshow(recon, cmap=plt.cm.gray)
            plt.colorbar()
            plt.axis('off')
            plt.savefig('dTV/Results_CT_dTV/TV_'+imaging_type+'_reg_param_'+'{:.1e}'.format(reg_param)+'.png')
            plt.close()



    with open('dTV/Results_CT_dTV/dTV_regularised_recons_varying_strong_cvx.json') as f:
        d = json.load(f)

    #fig, axs = plt.subplots(4, 5)

    for i, alpha in enumerate(alphas):
        for j, eta in enumerate(etas):
            for k, strong_cvx in enumerate(strong_cvxs):


            #dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]

            axs[i, j].imshow(np.asarray(d['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]).T[::-1, :], cmap=plt.cm.gray)
            axs[i, j].axis("off")

    plt.tight_layout(w_pad=0.1, rect=[0.2, 0, 0.2, 1])

with open('dTV/Results_CT_dTV/dTV_regularised_recons_varying_strong_cvx.json') as f:
    d = json.load(f)

plt.figure()

for i, alpha in enumerate(alphas):
    for j, eta in enumerate(etas):
        for k, strong_cvx in enumerate(strong_cvxs):

            plt.subplot(1,5,k+1)
            plt.imshow(np.asarray(d['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]['strong_cvx=' +'{:.1e}'.format(strong_cvx)]).T[::-1, :], cmap=plt.cm.gray)
            plt.axis("off")