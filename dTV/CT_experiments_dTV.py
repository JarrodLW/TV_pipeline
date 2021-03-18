#import astra
import h5py
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from processing import *
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json
from time import time

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


sino_Co_1_normalised = sino_Co_1/np.sqrt(np.sum(np.square(sino_Co_1)))
sino_0_XRD_normalised = sino_0_XRD/np.sqrt(np.sum(np.square(sino_0_XRD)))

## TV-regularised reconstructions
if TV_recon:

    model = VariationalRegClass('CT', 'TV')
    a_offset = -np.pi
    a_range = 2*np.pi
    d_offset = 0
    d_width = 40
    #d_width = 2

    #reg_params = [10.**(i-5) for i in np.arange(10)]
    #reg_params = [10**100]
    reg_params = [10**(-5)]

    TV_regularised_recons = {'XRF': {}, 'XRD': {}}

    for reg_param in reg_params:
        recons_XRF_TV = model.regularised_recons_from_subsampled_data(sino_Co_1_normalised.T, reg_param, recon_dims=(561, 561),
                                                                      niter=50, a_offset=a_offset, enforce_positivity=True,
                                                                      a_range=a_range, d_offset=0, d_width=d_width)[0]

        recons_XRD_TV = model.regularised_recons_from_subsampled_data(sino_0_XRD_normalised.T, reg_param, recon_dims=(561, 561),
                                                                      niter=100, a_offset=a_offset, enforce_positivity=True,
                                                                      a_range=a_range, d_offset=0, d_width=d_width)[0]

        TV_regularised_recons['XRF']['recon_param = '+'{:.1e}'.format(reg_param)] = recons_XRF_TV.tolist()
        TV_regularised_recons['XRD']['recon_param = '+'{:.1e}'.format(reg_param)] = recons_XRD_TV.tolist()


    json.dump(TV_regularised_recons, open('dTV/Results_CT_dTV/TV_regularised_recons.json', 'w'))
    #hf = h5py.File('dTV/Results_CT_dTV/TV_regularised_recons.h5', 'w')


## TV-regularised XRD reconstructions, masking out the spikes

# generating a mask that picks out only the "interior" of the sinogram

from scipy import ndimage

arr = ndimage.binary_fill_holes((sino_0_XRD<0.06)*sino_0_XRD).astype(int)

arr_lower_half = arr[arr.shape[0]//2:, :]
arr_upper_half = arr[:arr.shape[0]//2, :]

mask_exterior_signal = np.zeros(arr.shape)

for k in range(arr.shape[1]):

    I = np.nonzero(arr_lower_half[:, k])[0][0] + arr.shape[0]//2
    J = np.nonzero(arr_upper_half[:, k])[0][-1]

    mask_exterior_signal[J:I, k] = 1


if masked_TV_recon:

    model = VariationalRegClass('CT', 'TV')
    a_offset = -np.pi
    a_range = 2 * np.pi
    #a_range = np.pi
    d_offset = 0
    #d_width = 2
    d_width=40

    # figuring out what to mask

    # width, height = sino_0_XRD.shape
    sino_0_XRD_subsampled = sino_0_XRD[::10, :]
    width, height = sino_0_XRD_subsampled.shape

    image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[561, 561], dtype='float')
    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(a_offset, a_offset + a_range, height)
    # Detector: uniformly sampled
    detector_partition = odl.uniform_partition(d_offset - d_width / 2, d_offset + d_width / 2, width)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')
    FBP = odl.tomo.fbp_op(forward_op)

    #data = (sino_0_XRD < 0.18)*sino_0_XRD
    data = (sino_0_XRD_subsampled < 0.18) * sino_0_XRD_subsampled

    #data = background_minus_buffer[:, :-1]*sino_0_XRD
    #data = background[:, :-1] * sino_0_XRD

    recon_XRD = FBP(forward_op.range.element(data.T))

    # optimisation

    #reg_param = 1. #10. ** 0
    #reg_params = [10. ** (i - 5) for i in np.arange(20)]

    reg_params = [10**(-2.5)]

    #subsampling_arr = (sino_0_XRD < 0.18) * np.ones(sino_0_XRD.shape)
    spike_subsampling_arr = (sino_0_XRD_subsampled < 0.18)*np.ones(sino_0_XRD_subsampled.shape)
    #subsampling_arr =  (sino_0_XRD < 0.18)*np.ones(sino_0_XRD.shape)*mask_exterior_signal
    #subsampling_arr = (sino_0_XRD > 0.06)*(sino_0_XRD < 0.18) * np.ones(sino_0_XRD.shape)
    #masked_data = sino_0_XRD_normalised*subsampling_arr # I don't remember whether or not I have to do this explicitly.... check!
    #masked_data = sino_0_XRD_normalised*subsampling_arr
    #background = synth_data.T < 1.5
    # background = sino_0_XRD < 0.06
    # background_shifted_down = np.roll(background, 0, axis=0)
    # background_shifted_up = np.roll(background, 0, axis=0)
    # background_minus_buffer = background*background_shifted_down*background_shifted_up
    # masked_data_new = (1-background_minus_buffer)*masked_data # this is ad-hoc!
    #
    # new_subsampling_arr = (1-background_minus_buffer)*subsampling_arr

    # recons_XRD_TV = model.regularised_recons_from_subsampled_data(masked_data.T, reg_param, recon_dims=(561, 561), subsampling_arr=((1-background_minus_buffer)[:, :-1]*subsampling_arr).T,
    #                                                               niter=50, a_offset=a_offset, enforce_positivity=True,
    #                                                               a_range=a_range, d_offset=0, d_width=40)[0]

    TV_regularised_recons = {}
    exp=0

    for reg_param in reg_params:
        exp += 1
        print("Experiment " + str(exp))

        recons_XRD_TV = model.regularised_recons_from_subsampled_data(data.T, reg_param, recon_dims=(width, width),
                                                                      subsampling_arr=spike_subsampling_arr.T,
                                                                      niter=1000, a_offset=a_offset,
                                                                      enforce_positivity=True,
                                                                      a_range=a_range, d_offset=0, d_width=d_width)[0]

        model = VariationalRegClass('STEM', 'TV')
        recon = model.regularised_recons_from_subsampled_data(recons_XRD_TV, 0.0001, recon_dims=(561, 561), niter=200,
                                                                      enforce_positivity=True)

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
    #niter_prox = 1
    #niter = 250
    niter = 50

    #strong_cvxs = [1e1, 1e0, 1e-1, 1e-2, 1e-3]
    strong_cvxs = [1e-5]

    alphas = np.logspace(-5, -3, num=10)
    etas = [1. , 0.1, 0.01, 0.001, 0.0001, 0.00001]

    Yaff = odl.tensor_space(6)

    data = sino_0_XRD_normalised.T
    #data = sino_Co_1.T
    height, width = data.shape

    image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[561, 561], dtype='float')
    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, height)
    # Detector: uniformly sampled
    detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, width)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    subsampling_arr = (sino_0_XRD < 0.25) * np.ones(sino_0_XRD.shape)
    #every_other_angle_mask = np.zeros(sino_0_XRD.shape)
    #every_other_angle_mask[:, ::2] = 1
    #subsampling_arr = every_other_angle_mask*subsampling_arr
    masked_data = (subsampling_arr.T)*data

    # temp
    # background = synth_data.T < 1.5
    # background_shifted_down = np.roll(background, 0, axis=0)
    # background_shifted_up = np.roll(background, 0, axis=0)
    # background_minus_buffer = background * background_shifted_down * background_shifted_up
    # masked_data_new = ((1 - background_minus_buffer)[:, :-1]).T * masked_data  # this is ad-hoc!
    #
    # new_subsampling_arr = (1 - background_minus_buffer)[:, :-1] * subsampling_arr

    ##

    # Create the forward operator
    forward_op_CT = odl.tomo.RayTransform(image_space, geometry, impl='skimage')
    subsampled_forward_op = forward_op_CT.range.element(subsampling_arr.T)*forward_op_CT

    #data_odl = forward_op.range.element(masked_data)
    data_odl = forward_op_CT.range.element(masked_data)

    #remembering to rotate so that orientations match
    # with open('dTV/Results_CT_dTV/TV_regularised_recons.json') as f:
    #     d = json.load(f)
    #
    # f.close()
    #
    # recon_XRF = np.asarray(d['XRF']['recon_param = 1.0e-02']).T[:, ::-1]
    #sinfo = recons_XRF[0]
    sinfo = image_space.element(pre_registered_recon_XRF.T[:, ::-1])

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
    #f = fctls.DataFitL2Disp(X, data_odl, subsampled_forward_op)

    dTV_regularised_recons = {}
    exp=0
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:
            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = {}
            for strong_cvx in strong_cvxs:
                dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]['strong_cvx=' + '{:.1e}'.format(strong_cvx)] = {}
                start = time()

                print("Experiment "+str(exp))
                exp+=1

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
                palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
                palm.run(niter)

                recon = palm.x[0].asarray()
                fidelity = f(palm.x)

                print("Data fidelity: "+str(fidelity))

                end = time()
                diff = end - start
                print("Done in "+str(diff))

                dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)][
                    'strong_cvx=' + '{:.1e}'.format(strong_cvx)]['recon'] = recon.tolist()
                dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)][
                    'strong_cvx=' + '{:.1e}'.format(strong_cvx)]['fidelity'] = fidelity

    json.dump(dTV_regularised_recons, open('dTV/Results_CT_dTV/dTV_regularised_recons_varying_alpha_eta.json', 'w'))

with open('dTV/Results_CT_dTV/dTV_regularised_recons_varying_alpha_eta.json') as f:
    d = json.load(f)

f.close()

fig, axs = plt.subplots(10, 6)

for i, alpha in enumerate(alphas):
    for j, eta in enumerate(etas):
        for k, strong_cvx in enumerate(strong_cvxs):

            recon = np.asarray(d['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)][
                    'strong_cvx=' + '{:.1e}'.format(strong_cvx)]['recon'])

            axs[i, j].imshow(recon.T[::-1, :], cmap=plt.cm.gray)
            axs[i, j].axis("off")

plt.tight_layout(w_pad=0.1, rect=[0.2, 0, 0.2, 1])

for j, eta in enumerate(etas):
    fidelities = []
    for i, alpha in enumerate(alphas):


        fidelity = np.asarray(d['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)][
                'strong_cvx=' + '{:.1e}'.format(1e-5)]['fidelity'])

        # fidelity was defined as half of squared L2 norm
        fidelities.append(np.sqrt(2*fidelity)/np.sqrt(np.sum(np.square(masked_data))))

    plt.plot(alphas, fidelities, label="eta: "+'{:.1e}'.format(eta))#, color="C"+str(k%10))
    plt.legend()


np.save('/Users/jlw31/Desktop/dTV_example_pre_registered_raw_data.npy', recon.T[::-1, :])

# FBP reconstructions

FBP = odl.tomo.fbp_op(forward_op_CT)
FBP_XRD = FBP(forward_op_CT.range.element(sino_0_XRD.T))
FBP_XRD_background = FBP(forward_op_CT.range.element(((sino_0_XRD<0.08)*sino_0_XRD).T))

# SIRT reconstruction code - doesn't currently run on local machine

# import astra
#
# subsampling_arr = (sino_0_XRD < 0.18) * np.ones(sino_0_XRD.shape)
#
# proj_geom = astra.create_proj_geom('parallel', 1, 561, np.linspace(0, 2*np.pi, 59))
# vol_geom = astra.create_vol_geom(561, 561)
# proj_id = astra.creators.create_projector('line', proj_geom, vol_geom)
#
# sino_id = astra.data2d.create('-sino', proj_geom, sino_0_XRD.T)
# rec_id = astra.data2d.create('-vol', vol_geom)
# cfg = astra.astra_dict('SIRT');
# #cfg.ProjectorId = proj_id;
# #cfg.ProjectionDataId = sino_id;
# #cfg.ReconstructionDataId = rec_id;
# cfg['ProjectionDataId'] = sino_id
# cfg['ProjectorId'] = proj_id # new code
# cfg['ReconstructionDataId'] = rec_id
# #cfg.option.MinConstraint = 0;
# #cfg.option.MaxConstraint = 561;
# #cfg.option.SinogramMaskId = subsampling_arr
# cfg.option['MinConstraint'] = 0;
# cfg['MaxConstraint'] = 561;
# cfg['SinogramMaskId'] = subsampling_arr
# #sirt_id = astra.astra_mex_algorithm('create', cfg);
# #astra.astra_mex_algorithm('iterate', sirt_id, 100);
# alg_id = astra.algorithm.create(cfg)
# astra.algorithm.run(alg_id, 100)
# rec = astra.data2d.get(rec_id)
#V = astra.astra_mex_data2d('get', rec_id);
#imshow(V, [])

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


# registration stuff

from scipy import io
from skimage.util import compare_images

with open('dTV/Results_CT_dTV/TV_regularised_recons.json') as f:
    d = json.load(f)

f.close()

example_XRD_TV_recon = np.load('dTV/Results_CT_dTV/example_XRD_recon_TV_with_masking.npy')

f.close()

recon_XRF = np.asarray(d['XRF']['recon_param = 1.0e-02'])

pre_registered_recon_XRF = io.loadmat('/Users/jlw31/Desktop/XRF_example_TV_recon_registered.mat')['movingRegistered']

plt.figure()
plt.imshow(example_XRD_TV_recon, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recon_XRF, cmap=plt.cm.gray)

plt.figure()
plt.imshow(pre_registered_recon_XRF, cmap=plt.cm.gray)

comp_1 = compare_images(example_XRD_TV_recon/np.amax(example_XRD_TV_recon), recon_XRF/np.amax(recon_XRF), method='checkerboard', n_tiles=(16, 16))
comp_2 = compare_images(example_XRD_TV_recon/np.amax(example_XRD_TV_recon), pre_registered_recon_XRF/np.amax(pre_registered_recon_XRF), method='checkerboard', n_tiles=(16, 16))

plt.figure()
plt.imshow(comp_1, cmap=plt.cm.gray)

plt.figure()
plt.imshow(comp_2, cmap=plt.cm.gray)


fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(np.asarray([example_XRD_TV_recon/np.amax(example_XRD_TV_recon), np.zeros((561, 561)),
                          np.zeros((561, 561))]).transpose((1,2,0)))
axs[1].imshow(np.asarray([example_XRD_TV_recon/np.amax(example_XRD_TV_recon), np.zeros((561, 561)),
                          recon_XRF/np.amax(recon_XRF)]).transpose((1,2,0)))
axs[2].imshow(np.asarray([np.zeros((561, 561)), np.zeros((561, 561)),
                          recon_XRF/np.amax(recon_XRF)]).transpose((1,2,0)))

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].imshow(np.asarray([example_XRD_TV_recon/np.amax(example_XRD_TV_recon), np.zeros((561, 561)),
                          np.zeros((561, 561))]).transpose((1,2,0)))
axs[1].imshow(np.asarray([example_XRD_TV_recon/np.amax(example_XRD_TV_recon), np.zeros((561, 561)),
                          pre_registered_recon_XRF/np.amax(pre_registered_recon_XRF)]).transpose((1,2,0)))
axs[2].imshow(np.asarray([np.zeros((561, 561)), np.zeros((561, 561)),
                          pre_registered_recon_XRF/np.amax(pre_registered_recon_XRF)]).transpose((1,2,0)))


synth_data = forward_op_CT(pre_registered_recon_XRF.T[:,::-1]).asarray()

plt.figure()
plt.imshow(synth_data.T, cmap=plt.cm.gray, aspect=0.1, vmax=2)
plt.colorbar()

plt.figure()
plt.imshow(sino_Co_1, cmap=plt.cm.gray, aspect=0.1, vmax=2000)
plt.colorbar()

plt.figure()
plt.imshow((sino_0_XRD<0.06)*sino_0_XRD, cmap=plt.cm.gray, aspect=0.1)#, vmin=0.06, vmax=0.1)
plt.colorbar()


# dTV-assisted recon but at level of images

gamma = 0.995
#strong_cvx = 1e-2
#strong_cvx = 1e-1
niter_prox = 20
#niter = 250
niter = 100
strong_cvx = 1e-5

#alphas = [10.**(i-5) for i in np.arange(4, 8)]
#etas = [10.**(-i) for i in np.arange(5)]
#alphas = [10.**(-1)]
#etas = [10.**(-3)]

alphas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
etas = [0.1, 0.01, 0.001, 0.0001]

Yaff = odl.tensor_space(6)

data = example_XRD_TV_recon/np.amax(example_XRD_TV_recon)*10
#data = (FBP_XRD.asarray()).T[::-1, :]
height, width = data.shape

image_space = odl.uniform_discr(min_pt=[0, 0], max_pt=[1, 1], shape=[561, 561], dtype='float')

forward_op = odl.operator.default_ops.IdentityOperator(image_space)

data_odl = forward_op.range.element(data)
sinfo = image_space.element(pre_registered_recon_XRF)

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
#x = image_space.zero()

f = fctls.DataFitL2Disp(X, data_odl, forward_op)
#f = fctls.DataFitL2Disp(image_space, data_odl, forward_op)
#f = odl.solvers.L2NormSquared(image_space).translated(data_odl)
exp=0


dTV_regularised_recons = {}
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:

            print("Experiment "+str(exp))
            exp+=1

            t0 = time()

            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = {}

            reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                                gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                                prox_options=prox_options)

            g = odl.solvers.SeparableSum(reg_im, reg_affine)
            #g = odl.solvers.SeparableSum(reg_im)

            cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                  odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                  odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                  odl.solvers.CallbackShow()
                  )

            L = [1, 1e+2]
            ud_vars = [0]

            # %%
            palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
            palm.run(niter)

            # op_norm = 1.1 * odl.power_method_opnorm(forward_op)
            # tau = 1.0 / op_norm  # Step size for the primal variable
            # sigma = 1.0 / op_norm
            #
            # odl.solvers.pdhg(x, f, g, forward_op, niter=niter, tau=tau, sigma=sigma)

            recon = palm.x[0].asarray()
            #recon = x[0].asarray()

            dt = time() - t0
            print('done in %.2fs.' % dt)

            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = recon.tolist()

json.dump(dTV_regularised_recons, open('dTV/Results_CT_dTV/dTV_at_image_level.json', 'w'))

with open('dTV/Results_CT_dTV/dTV_at_image_level.json') as f:
    d = json.load(f)

f.close()

fig, axs = plt.subplots(5, 4, figsize=(8, 10))
for i, alpha in enumerate(alphas[:5]):
    for j, eta in enumerate(etas):
        recon = np.asarray(dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)])
        axs[i, j].imshow(recon, cmap=plt.cm.gray)
        axs[i, j].axis("off")

plt.tight_layout()