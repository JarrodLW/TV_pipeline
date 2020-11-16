import odl
from odl.solvers.functional.functional import Functional
from odl.operator import Operator
import numpy as np
from odl_implementation_CT_KL import CTKullbackLeibler


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
max_intens = 1.
max_intens_2 = 1000.
synth_data = max_intens*np.exp(-forward_op(phantom))
synth_data_2 = max_intens_2*np.exp(-forward_op(phantom))

noisy_synth_data = odl.phantom.poisson_noise(synth_data, seed=16)
noisy_synth_data_2 = odl.phantom.poisson_noise(synth_data_2, seed=16)
log_data = -np.log(noisy_synth_data/max_intens)
log_data_2 = -np.log(noisy_synth_data_2/max_intens_2)

log_data.show()

g = CTKullbackLeibler(forward_op.range, prior=synth_data, max_intens=max_intens)

#f = odl.solvers.ZeroFunctional(image_space)
#f = 0.1*odl.solvers.L2NormSquared(image_space)
f = odl.solvers.IndicatorNonnegativity(image_space)
x = image_space.zero()

odl.solvers.pdhg(x, f, g, forward_op, niter=niter, tau=tau, sigma=sigma)

FBP = odl.tomo.fbp_op(forward_op)
recon_fbp = FBP(log_data)
recon_fbp.show()

log_data.show()

synth_data.show()
noisy_synth_data.show()