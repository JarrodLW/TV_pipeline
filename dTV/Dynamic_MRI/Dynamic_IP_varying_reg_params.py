# Created 8/04/2021. In this script we treat space and time on an equal footing by using 3-dimensional spacetime domain.
# The payoff is that the RealFourierTransform implementation had to be adjusted to work on stacks of data
# ---see Real_Fourier_op_stacked.py

import odl
from Utils import *
from dTV.Dynamic_MRI.Real_Fourier_op_stacked import RealFourierTransformStacked
import matplotlib.pyplot as plt
from myOperators import RealFourierTransform

clean_first_and_last_images = True

im_array = np.load('dTV/Dynamic_MRI/dynamic_phantom_images.npy')
frame_num, height, width = im_array.shape

complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                              shape=[height, width], dtype='complex')
image_space = complex_space.real_space ** 2
fourier_transf = RealFourierTransform(image_space)

data_stack = np.zeros((frame_num, height, width), dtype='complex')
naive_recons = np.zeros((frame_num, height, width), dtype='complex')

for i in range(frame_num):
    complex_im = image_space.element([im_array[i], image_space[1].zero()])
    synth_data = fourier_transf(complex_im).asarray()
    synth_data = synth_data[0] + 1j * synth_data[1]

    if clean_first_and_last_images:
        if i == 0:
            noise = 0.01 * np.random.normal(loc=0.0, scale=0.5, size=[height, width]) \
                    + 0.01 * 1j * np.random.normal(loc=0.0, scale=0.5, size=[height, width])

        elif i == frame_num-1:
            noise = 0.01 * np.random.normal(loc=0.0, scale=0.5, size=[height, width]) \
                    + 0.01 * 1j * np.random.normal(loc=0.0, scale=0.5, size=[height, width])

        else:
            noise = 0.1 * np.random.normal(loc=0.0, scale=0.5, size=[height, width]) \
                    + 0.1 * 1j * np.random.normal(loc=0.0, scale=0.5, size=[height, width])

    else:
        noise = 0.1 * np.random.normal(loc=0.0, scale=0.5, size=[height, width]) \
                + 0.1 * 1j * np.random.normal(loc=0.0, scale=0.5, size=[height, width])

    synth_data_with_noise = synth_data + noise
    data_stack[i] = synth_data_with_noise
    naive_recons[i] = np.fft.ifftshift(np.fft.ifft2(synth_data_with_noise))

averaged_fourier_data = np.average(data_stack, axis=0)
naive_recon_fully_averaged = np.fft.ifftshift(np.fft.ifft2(averaged_fourier_data))

# real-valued data
data_real = [np.real(data_stack), np.imag(data_stack)]

# defining domain of inverse problem
complex_space = odl.uniform_discr(min_pt=[-1., -1., -1.], max_pt=[1., 1., 1.], shape=data_stack.shape, dtype='complex')
recon_space = complex_space.real_space ** 2

# constructing gradient operators, spatial and temporal
if clean_first_and_last_images:
    alphas = [0.00001] + [0.004]*(frame_num-2) + [0.00001]
else:
    alphas = [0.004]*frame_num

weight_arr = np.ones(im_array.shape)

for i in range(frame_num):
    weight_arr[i] *= alphas[i]

G_of_real = odl.Gradient(recon_space[0]) * odl.ComponentProjection(recon_space, 0)
G_of_imag = odl.Gradient(recon_space[1]) * odl.ComponentProjection(recon_space, 1)
temporal_G_of_real = odl.ComponentProjection(G_of_real.range, 0)*G_of_real
spatial_G_of_real_1 = odl.ComponentProjection(G_of_real.range, 1)*G_of_real
spatial_G_of_real_2 = odl.ComponentProjection(G_of_real.range, 2)*G_of_real
temporal_G_of_imag = odl.ComponentProjection(G_of_imag.range, 0)*G_of_imag
spatial_G_of_imag_1 = odl.ComponentProjection(G_of_imag.range, 1)*G_of_imag
spatial_G_of_imag_2 = odl.ComponentProjection(G_of_imag.range, 2)*G_of_imag

# stacking the spatial gradients
spatial_grad_codomain = complex_space.real_space ** 4
embedding_real_1 = odl.ComponentProjection(spatial_grad_codomain, 0).adjoint
embedding_real_2 = odl.ComponentProjection(spatial_grad_codomain, 1).adjoint
embedding_imag_1 = odl.ComponentProjection(spatial_grad_codomain, 2).adjoint
embedding_imag_2 = odl.ComponentProjection(spatial_grad_codomain, 3).adjoint

spatial_G_stacked = embedding_real_1*spatial_G_of_real_1 + embedding_real_2*spatial_G_of_real_2 + \
                    embedding_imag_1*spatial_G_of_imag_1 + embedding_imag_2*spatial_G_of_imag_2
weighted_spatial_G_stacked = spatial_G_stacked*spatial_G_stacked.domain.element([weight_arr, weight_arr])
spatial_reg = odl.solvers.GroupL1Norm(spatial_grad_codomain)

# stacking the temporal gradients
gamma = 0.01
#gamma = 0.1

temporal_grad_codomain = complex_space.real_space ** 2
embedding_real = odl.ComponentProjection(temporal_grad_codomain, 0).adjoint
embedding_imag = odl.ComponentProjection(temporal_grad_codomain, 1).adjoint

temporal_G_stacked = embedding_real*temporal_G_of_real + embedding_imag*temporal_G_of_imag
temporal_reg = gamma*odl.solvers.GroupL1Norm(temporal_grad_codomain)

# constructing the forward operator
f_op = RealFourierTransformStacked(recon_space)
op = odl.BroadcastOperator(f_op, weighted_spatial_G_stacked, temporal_G_stacked)
op_norm = 1.1 * odl.power_method_opnorm(op)
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable
niter = 5
#niter = 100

data_odl = f_op.range.element(data_real)
data_fit = odl.solvers.L2Norm(f_op.range).translated(data_odl)
g = odl.solvers.SeparableSum(data_fit, spatial_reg, temporal_reg)
f = odl.solvers.ZeroFunctional(op.domain)
x = op.domain.zero()

odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma)
recon_slice_0 = x.asarray()[0, 0] + 1j*x.asarray()[1, 0]
recon_slice_1 = x.asarray()[0, frame_num//4] + 1j*x.asarray()[1, frame_num//3]
recon_slice_2 = x.asarray()[0, 2*frame_num//4] + 1j*x.asarray()[1, 2*frame_num//4]
recon_slice_3 = x.asarray()[0, 3*frame_num//4] + 1j*x.asarray()[1, 3*frame_num//4]
recon_slice_4 = x.asarray()[0, -1] + 1j*x.asarray()[1, -1]
#recon_slice_4 = x.asarray()[0, -2] + 1j*x.asarray()[1, -1]

f, axs = plt.subplots(3, 5)
axs[0, 0].imshow(np.abs(im_array[0]).T[::-1], cmap=plt.cm.gray)
axs[0, 0].axis("off")
axs[0, 1].imshow(np.abs(im_array[frame_num//4]).T[::-1], cmap=plt.cm.gray)
axs[0, 1].axis("off")
axs[0, 2].imshow(np.abs(im_array[2*frame_num//4]).T[::-1], cmap=plt.cm.gray)
axs[0, 2].axis("off")
axs[0, 3].imshow(np.abs(im_array[3*frame_num//4]).T[::-1], cmap=plt.cm.gray)
axs[0, 3].axis("off")
axs[0, 4].imshow(np.abs(im_array[-1]).T[::-1], cmap=plt.cm.gray)
axs[0, 4].axis("off")
# axs[0, 4].imshow(im_array[-2].T[::-1], cmap=plt.cm.gray)
# axs[0, 4].axis("off")
axs[1, 0].imshow(np.abs(recon_slice_0).T[::-1], cmap=plt.cm.gray)
axs[1, 0].axis("off")
axs[1, 1].imshow(np.abs(recon_slice_1).T[::-1], cmap=plt.cm.gray)
axs[1, 1].axis("off")
axs[1, 2].imshow(np.abs(recon_slice_2).T[::-1], cmap=plt.cm.gray)
axs[1, 2].axis("off")
axs[1, 3].imshow(np.abs(recon_slice_3).T[::-1], cmap=plt.cm.gray)
axs[1, 3].axis("off")
axs[1, 4].imshow(np.abs(recon_slice_4).T[::-1], cmap=plt.cm.gray)
axs[1, 4].axis("off")
# axs[1, 4].imshow(np.abs(recon_slice_4).T[::-1], cmap=plt.cm.gray)
# axs[1, 4].axis("off")
axs[2, 0].imshow(np.abs(naive_recons[0]).T[::-1], cmap=plt.cm.gray)
axs[2, 0].axis("off")
axs[2, 1].imshow(np.abs(naive_recons[frame_num//4]).T[::-1], cmap=plt.cm.gray)
axs[2, 1].axis("off")
axs[2, 2].imshow(np.abs(naive_recons[2*frame_num//4]).T[::-1], cmap=plt.cm.gray)
axs[2, 2].axis("off")
axs[2, 3].imshow(np.abs(naive_recons[3*frame_num//4]).T[::-1], cmap=plt.cm.gray)
axs[2, 3].axis("off")
axs[2, 4].imshow(np.abs(naive_recons[-1]).T[::-1], cmap=plt.cm.gray)
axs[2, 4].axis("off")
# axs[2, 4].imshow(np.abs(naive_recons[-2]).T[::-1], cmap=plt.cm.gray)
# axs[2, 4].axis("off")
f.tight_layout(w_pad=0.4, h_pad=0.1)

