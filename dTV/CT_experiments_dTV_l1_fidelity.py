# created on 5/10/2020, based on "CT_experiments_dTV.py" - this is a bit hacked together!

import h5py
import numpy as np
import matplotlib.pyplot as plt
#import astra
from processing import *
#import dTV.myFunctionals as fctls
#import dTV.myAlgorithms as algs
import json

#filename = 'dTV/Experiment1_XRD.hdf5'
filename = 'Experiment1_XRD.hdf5'
f2 = h5py.File(filename, 'r+')

data_XRD = np.array(f2['sino_XRD'])

## pre-processing of data: cumulative sum of hits within specified freq range
data_XRD_0 = data_XRD[:, :, :, 0]
# summing over pixels
plt.figure()
plt.hist(np.sum(data_XRD_0, axis=(0, 1)), range=(0, 4000), bins=200)
plt.title("Spectrum obtained by summing over pixels")
#plt.show()

# we select the range 450-500, corresponding roughly to the second spectral peak
filter = np.zeros(data_XRD.shape[2])
filter[450:500] = 1
sino_0_XRD = np.dot(data_XRD[:, :, :, 0], filter)

FBP_recon = "True"

sino_0_XRD_thresh = sino_0_XRD*(sino_0_XRD<0.2) + 0.2*np.ones(sino_0_XRD.shape)*(sino_0_XRD>=0.2)

if FBP_recon:

    center = 0
    angle_array = 2*np.pi * np.arange(59) / 59

    recon_XRD_thresh_FBP = recon_astra(sino_0_XRD_thresh.T, center, method='FBP', ratio=None, angles=angle_array, num_iter=200)
    recon_XRD_FBP = recon_astra(sino_0_XRD.T, center, method='FBP', ratio=None, angles=angle_array, num_iter=200)

    np.save('recon_XRD_thresh_FBP', recon_XRD_thresh_FBP)
    np.save('recon_XRD_FBP', recon_XRD_FBP)


recon_XRD_thresh_FBP = np.load('dTV/Results_CT_dTV/recon_XRD_thresh_FBP.npy')
recon_XRD_FBP = np.load('dTV/Results_CT_dTV/recon_XRD_FBP.npy')



## dTV-regularised XRD recon with XRF sinfo
a_offset = -np.pi
a_range = 2*np.pi
d_offset = 0
d_width = 40

Yaff = odl.tensor_space(6)

data = sino_0_XRD.T
height, width = data.shape

X = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[561, 561], dtype='float')
# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, height)
# Detector: uniformly sampled
detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, width)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create the forward operator
K = odl.tomo.RayTransform(X, geometry, impl='skimage')

data_odl = K.range.element(data)

# remembering to rotate so that orientations match
with open('dTV/Results_CT_dTV/TV_regularised_recons.json') as f:
    d = json.load(f)

recon_XRF = np.asarray(d['XRF']['recon_param = 1.0e-02']).T[:, ::-1]
sinfo = X.element(recon_XRF)

norm_K = K.norm(estimate=True)
Y = K.range

#data_fit = 0.5 * odl.solvers.L2NormSquared(Y).translated(data)
#data_fit = odl.solvers.functional.default_functionals.Huber(Y, gamma=0.2).translated(data)
data_fit = odl.solvers.L1Norm(Y).translated(data)

grad = odl.Gradient(X)
sinfo_grad = grad(sinfo)

grad_space = grad.range
norm = odl.PointwiseNorm(grad_space, 2)

norm_sinfo_grad = norm(sinfo_grad)

alphas = [10.**(i-5) for i in np.arange(10)]
etas = [10.**(i-5) for i in np.arange(10)]

dTV_regularised_recons = {}
experiment_num = 0
for alpha in alphas:
    dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
    for eta in etas:

        experiment_num += 1
        print("Experiment " + str(experiment_num))

        norm_eta_sinfo_grad = np.sqrt(norm_sinfo_grad ** 2 +
                                      eta ** 2)

        xi = grad_space.element([g / norm_eta_sinfo_grad for g in sinfo_grad])

        Id = odl.operator.IdentityOperator(grad_space)
        xiT = odl.PointwiseInner(grad_space, xi)

        xixiT = odl.BroadcastOperator(*[x*xiT for x in xi])

        gamma = 0.995
        D = (Id - gamma * xixiT) * grad

        norm_D = D.norm(estimate=True)

        c = float(norm_K) / float(norm_D)
        D *= c
        norm_D *= c
        L1 = (alpha / c) * odl.solvers.GroupL1Norm(D.range)

        A = odl.BroadcastOperator(K, D)
        #strong_convexity = 0*odl.solvers.L2NormSquared(X)
        f = odl.solvers.SeparableSum(data_fit, L1)
        #g = box_fun
        g = odl.solvers.IndicatorNonnegativity(X)
        #g = 0.01*odl.solvers.L2NormSquared(X)
        #g = odl.solvers.SeparableSum(0.01*odl.solvers.L2NormSquared(X), odl.solvers.IndicatorNonnegativity(X))

        # output function to be used with the iterations
        cb = (odl.solvers.CallbackPrintIteration(end=', ') &
              odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
              odl.solvers.CallbackPrintTiming(fmt='total={:.3f} s', cumulative=True))

        x = X.zero()  # initialise variable

        rho = .99
        norm_A = A.norm(estimate=True)
        sigma = rho / norm_A
        tau = rho / norm_A

        niter = 100
        #%%
        odl.solvers.pdhg(x, g, f, A, niter, tau=tau, sigma=sigma, callback=cb)
        recon = x.asarray()

        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = recon.tolist()

json.dump(dTV_regularised_recons, open('dTV/Results_CT_dTV/dTV_regularised_recons_l1_pdhg.json', 'w'))

with open('dTV/Results_CT_dTV/dTV_regularised_recons_l1_pdhg.json') as f:
    d = json.load(f)

fig, axs = plt.subplots(10, 5)
for i, alpha in enumerate(alphas):
    for j, eta in enumerate(etas[:5][::-1]):

        axs[i, j].imshow(np.asarray(d['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]).T[::-1, :],
                         cmap=plt.cm.gray)
        axs[i, j].axis("off")

plt.tight_layout(w_pad=0.1, rect=[0.2, 0, 0.2, 1])
