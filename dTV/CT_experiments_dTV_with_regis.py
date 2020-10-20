# created on 13/10, based on "CT_experiments_dTV.py". Aim: to see how well the datasets are registered and
# to see if simultaneous recon-registration performs better

#import astra
import h5py
import numpy as np
import matplotlib.pyplot as plt
from processing import *
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json
import imageio

# pre-processing of data
filename = 'dTV/Experiment1_XRD.hdf5'
f2 = h5py.File(filename, 'r+')
data_XRD = np.array(f2['sino_XRD'])

## pre-processing of data: cumulative sum of hits within specified freq range
data_XRD_0 = data_XRD[:, :, :, 0]

# we select the range 450-500, corresponding roughly to the second spectral peak
filter = np.zeros(data_XRD.shape[2])
filter[450:500] = 1
sino_0_XRD = np.dot(data_XRD[:, :, :, 0], filter)

# grabbing example recons
# with open('dTV/Results_CT_dTV/TV_regularised_recons.json') as f:
#     d = json.load(f)
#
# recon_XRF = np.asarray(d['XRF']['recon_param = 1.0e-02'])
# np.save('dTV/Results_CT_dTV/example_TV_recon_XRF.npy', recon_XRF)
#
# with open('dTV/Results_CT_dTV/dTV_regularised_recons_l1_pdhg.json') as f:
#     d = json.load(f)
#
# recon_XRD = np.asarray(d['alpha=1.0e-01']['eta=1.0e+04']).T[::-1, :]
# np.save('dTV/Results_CT_dTV/example_dTV_l1_recon_XRD.npy', recon_XRD)

recon_XRF = np.load('dTV/Results_CT_dTV/example_TV_recon_XRF.npy')
recon_XRD = np.load('dTV/Results_CT_dTV/example_dTV_l1_recon_XRD.npy')

image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=list(recon_XRD.shape), dtype='float')
reg_func_fine = fctls.directionalTotalVariationNonnegative(image_space, alpha=1, sinfo=recon_XRF)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(np.asarray([recon_XRF/np.amax(recon_XRF), np.zeros(recon_XRF.shape),
                       np.zeros(recon_XRF.shape)]).transpose((1, 2, 0)))
ax2.imshow(np.asarray([recon_XRF/np.amax(recon_XRF), np.zeros(recon_XRF.shape),
                       0.6*recon_XRD/np.amax(recon_XRD)]).transpose((1, 2, 0)))
ax3.imshow(np.asarray([np.zeros(recon_XRD.shape),
                       np.zeros(recon_XRD.shape), recon_XRD/np.amax(recon_XRD)]).transpose((1, 2, 0)))