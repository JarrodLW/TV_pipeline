import odl
import dTV.myFunctionals as fctls
import numpy as np
import matplotlib.pyplot as plt
from dTV.myDeform.linearized import linear_deform

height=width=100
image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                            shape=[height, width], dtype='float')

# the image we're fitting to
#data = 2*odl.phantom.transmission.shepp_logan(image_space, modified=True)
data = 4*odl.phantom.geometric.smooth_cuboid(image_space)
#data = image_space.zero()

Yaff = odl.tensor_space(6)

# the point at which we'll compute the gradient
image = odl.phantom.transmission.shepp_logan(image_space, modified=True)
#aff_0 = Yaff.element([0.01, 0.05, 0.02, -0.01, 0.02, -0.05])
aff_0 = Yaff.zero()
base_point = [image, aff_0]

X = odl.ProductSpace(image_space, Yaff)

f = fctls.DataFitL2Disp(X, data, odl.operator.default_ops.IdentityOperator(image_space))
#base_point = [image, Yaff.zero()]
inner_prod = f.derivative(base_point)

noise_templ_image = odl.phantom.noise.white_noise(image_space, seed=10)
noise_templ_image /= np.sqrt(np.sum(np.square(noise_templ_image)))
noise_templ_aff = 0.001*odl.phantom.noise.white_noise(Yaff, seed=10)
noise_templ_image_norm_squared = np.sum(np.square(noise_templ_image.asarray()))
noise_templ_aff_norm_squared = np.sum(np.square(noise_templ_aff.asarray()))

pert_factors = np.logspace(-5, 0, num=30)
diffs_0 = []
diffs_1 = []
diffs_2 = []
diffs_3 = []

for pert_factor in pert_factors:
    image_pert = pert_factor*noise_templ_image
    aff_pert = pert_factor*noise_templ_aff
    perturbed_image = image + image_pert
    perturbed_aff = aff_0 + aff_pert
    diff_0 = f([perturbed_image, aff_0]) - f(base_point)
    diff_1 = f([perturbed_image, aff_0]) - f(base_point) - inner_prod([image_pert, Yaff.zero()])
    diff_2 = f([image, perturbed_aff]) - f(base_point)
    diff_3 = f([image, perturbed_aff]) - f(base_point) - inner_prod([image_space.zero(), aff_pert])

    diffs_0.append(diff_0)
    diffs_1.append(diff_1)
    diffs_2.append(diff_2)
    diffs_3.append(diff_3)

# we get a roughly linear plot with slope 2, as expected:
# plt.scatter(np.log(np.sqrt(noise_templ_image_norm_squared)*np.asarray(pert_factors)), np.log(diffs_1))
#
# plt.scatter(np.log(np.sqrt(noise_templ_aff_norm_squared)*np.asarray(pert_factors)), np.log(diffs_2))
#
# plt.figure()
# plt.scatter(np.sqrt(noise_templ_image_norm_squared)*np.asarray(pert_factors), diffs_0, label='zeroth order')
# plt.scatter(np.sqrt(noise_templ_image_norm_squared)*np.asarray(pert_factors), diffs_1, label='first order')
# plt.legend()
# plt.show()

plt.figure()
plt.scatter(np.log(np.sqrt(noise_templ_image_norm_squared)*np.asarray(pert_factors)), np.log(np.abs(diffs_0)), label='zeroth order')
plt.scatter(np.log(np.sqrt(noise_templ_image_norm_squared)*np.asarray(pert_factors)), np.log(np.abs(diffs_1)), label='first order')
plt.legend()
plt.show()

plt.figure()
plt.scatter(np.log(np.sqrt(noise_templ_image_norm_squared)*np.asarray(pert_factors)), np.log(np.abs(diffs_2)), label='zeroth order')
plt.scatter(np.log(np.sqrt(noise_templ_image_norm_squared)*np.asarray(pert_factors)), np.log(np.abs(diffs_3)), label='first order')
plt.legend()
plt.show()


