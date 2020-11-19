import odl
from odl.solvers.functional.functional import Functional
from odl.operator import Operator
import numpy as np
from odl_implementation_CT_KL import CTKullbackLeibler
from processing import *
import json

height=width=100
image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                            shape=[height, width], dtype='float')

a_offset = 0
a_range = 2*np.pi
d_offset = 0
d_width = 40

# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, height)
# Detector: uniformly sampled
detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, width)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')

op_norm = 1.1 * odl.power_method_opnorm(forward_op)
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable
niter = 1000

phantom = 0.3*odl.phantom.transmission.shepp_logan(image_space, modified=True)
max_intens = 10.
#max_intens = np.ones(phantom.shape)
#max_intens = np.abs(np.random.normal(size=phantom.shape))
max_intens_2 = 1000.
synth_data = max_intens*np.exp(-forward_op(phantom))
synth_data_2 = max_intens_2*np.exp(-forward_op(phantom))

eps = 1e-5
noisy_synth_data = odl.phantom.poisson_noise(synth_data, seed=16)
noisy_synth_data_2 = odl.phantom.poisson_noise(synth_data_2, seed=16)
log_data = -np.log(np.maximum(noisy_synth_data, eps)/max_intens)
log_data_2 = -np.log(np.maximum(noisy_synth_data_2, eps)/max_intens_2)

log_data.show()

# fbp
FBP = odl.tomo.fbp_op(forward_op)
recon_fbp = FBP(-np.log(synth_data/max_intens))
recon_fbp.show()

FBP = odl.tomo.fbp_op(forward_op)
recon_fbp = FBP(forward_op.range.element(np.maximum(log_data.asarray(), 0)))
recon_fbp.show()

FBP = odl.tomo.fbp_op(forward_op)
recon_fbp = FBP(forward_op.range.element(log_data.asarray()))
recon_fbp.show()

FBP = odl.tomo.fbp_op(forward_op)
recon_fbp = FBP(forward_op.range.element(log_data_2.asarray()))
recon_fbp.show()

# using new functional
g = CTKullbackLeibler(forward_op.range, prior=noisy_synth_data, max_intens=max_intens)

#f = odl.solvers.ZeroFunctional(image_space)
#f = 0.1*odl.solvers.L2NormSquared(image_space)
f = odl.solvers.IndicatorNonnegativity(image_space)
x = image_space.zero()

odl.solvers.pdhg(x, f, g, forward_op, niter=niter, tau=tau, sigma=sigma)

# using new functional + TV
G = odl.Gradient(image_space)
op = odl.BroadcastOperator(forward_op, G)
f = odl.solvers.IndicatorNonnegativity(image_space)

reg_params = np.linspace(0.05, 1., num=20).tolist()
regularised_recons = {}
exp = 0

for reg_param in reg_params:

    print("CTKL experiment"+str(exp))

    g = odl.solvers.SeparableSum(CTKullbackLeibler(forward_op.range, prior=noisy_synth_data, max_intens=max_intens), reg_param * odl.solvers.GroupL1Norm(G.range))
    x = image_space.zero()

    odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma)
    regularised_recons['reg_param=' + '{:.1e}'.format(reg_param)] = x.asarray().tolist()

    exp+=1

json.dump(regularised_recons, open('dTV/Results_CT_dTV/TV_CTKL_phantom.json', 'w'))

# TV with L2 data fidelity
model = VariationalRegClass('CT', 'TV')

reg_params = np.linspace(1., 10., num=20).tolist()
regularised_recons = {}
exp = 0

for reg_param in reg_params:

    print("L2 experiment" + str(exp))

    recons = model.regularised_recons_from_subsampled_data(log_data.asarray(), reg_param, recon_dims=phantom.shape,
                                                                          niter=1000, a_offset=a_offset, enforce_positivity=True,
                                                                          a_range=a_range, d_offset=d_offset, d_width=d_width)

    regularised_recons['reg_param=' + '{:.1e}'.format(reg_param)] = recons[0].tolist()

    exp+=1

json.dump(regularised_recons, open('dTV/Results_CT_dTV/TV_L2_phantom.json', 'w'))


with open('dTV/Results_CT_dTV/TV_CTKL_phantom.json') as f:
    d = json.load(f)

fig, axs = plt.subplots(4, 5, figsize=(5, 4))
reg_params = np.linspace(0.05, 1., num=20).tolist()

for i, reg_param in enumerate(reg_params):

    recon = np.asarray(d['reg_param=' + '{:.1e}'.format(reg_param)]).astype('float64')
    axs[i//5, i % 5].imshow(recon.T[::-1 ,:], cmap=plt.cm.gray)
    axs[i//5, i % 5].axis("off")

with open('dTV/Results_CT_dTV/TV_L2_phantom.json') as f:
    d = json.load(f)

fig, axs = plt.subplots(4, 5, figsize=(5, 4))
reg_params = np.linspace(1., 10., num=20).tolist()

for i, reg_param in enumerate(reg_params):

    recon = np.asarray(d['reg_param=' + '{:.1e}'.format(reg_param)]).astype('float64')
    axs[i//5, i % 5].imshow(recon, cmap=plt.cm.gray)
    axs[i//5, i % 5].axis("off")

log_data.show()
synth_data.show()
noisy_synth_data.show()