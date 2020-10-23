# Here we use dTV to super-resolve an image, working at the level of the images

import numpy as np
import matplotlib.pyplot as plt
import json
import dTV.myOperators as ops
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
from processing import *
from Utils import *
import dTV.myDeform.linearized as defs

# high resolution H image
fourier_H_real_im = np.reshape(np.fromfile('dTV/MRI_data/fid_H', dtype=np.int32), (128, 256))
fourier_H_real = fourier_H_real_im[:, ::2]
fourier_H_im = fourier_H_real_im[:, 1::2]
fourier_H = fourier_H_real + fourier_H_im*1j
my_recon_H = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_H)))
image = np.abs(my_recon_H)

height, width = image.shape
image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height, width], dtype='float')
image_odl = image_space.element(image)

# downsampling
coarse_image_space_1 = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height//2, width//2], dtype='float')
coarse_image_space_2 = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height//4, width//4], dtype='float')
subsampling_op_1 = ops.Subsampling(image_space, coarse_image_space_1)
subsampling_op_2 = ops.Subsampling(image_space, coarse_image_space_2)
subsampling_op_3 = ops.Subsampling(coarse_image_space_1, coarse_image_space_2)
downsampled_image_1_odl = subsampling_op_1(image_odl)
downsampled_image_2_odl = subsampling_op_2(image_odl)
downsampled_image_1 = downsampled_image_1_odl.asarray()
downsampled_image_2 = downsampled_image_2_odl.asarray()

# a displaced version of the high-res H image
disp_horiz = 0.05*np.ones((height, width))
displacement = np.asarray([disp_horiz, np.zeros((height, width))])
displ_image = defs.linear_deform(image_odl, image_space.tangent_bundle.element(displacement))
displ_image_odl = image_space.element(displ_image)
displ_image_odl.show()

# the given raw data for for lower-resolution image
f = open('dTV/MRI_data/1H_lowRes_imaginaryRaw_noZeros', 'r')
fourier_data_im = np.genfromtxt(f, delimiter=' ').T
f = open('dTV/MRI_data/1H_lowRes_realRaw_noZeros', 'r')
fourier_data_real = np.genfromtxt(f, delimiter=' ').T

fourier_data = (fourier_data_real + fourier_data_im*1j)
recon_from_unpacked_data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_data)))
low_res_image = np.abs(recon_from_unpacked_data)
low_res_image_odl = coarse_image_space_2.element(low_res_image)

TV_recon = False
dTV_recon = True

# super-resolving the downsampled images by dTV
if dTV_recon:

    gamma = 0.995
    strong_cvx = 1e-2
    niter_prox = 20
    niter = 1000

    alphas = [10.**(i-5) for i in np.arange(10)]
    etas = [10.**(-i) for i in np.arange(6)]
    #alphas = [1.]
    #etas = [10**(-5)]

    Yaff = odl.tensor_space(6)

    # Create the forward operator
    forward_op = subsampling_op_2

    data_odl = low_res_image_odl
    #data_odl = downsampled_image_2_odl
    #sinfo = displ_image_odl
    sinfo = image_odl

    # space of optimised variables
    # X = odl.ProductSpace(coarse_image_space_1, Yaff)
    X = odl.ProductSpace(image_space, Yaff)

    # Set some parameters and the general TV prox options
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = niter_prox

    reg_affine = odl.solvers.ZeroFunctional(Yaff)
    # x0 = X.zero()
    x0 = X.element([subsampling_op_2.adjoint(data_odl), X[1].zero()])


    f = fctls.DataFitL2Disp(X, data_odl, forward_op)

    dTV_regularised_recons = {}
    exp = 0
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:

            reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                                gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                                prox_options=prox_options)

            g = odl.solvers.SeparableSum(reg_im, reg_affine)

            # cb = (odl.solvers.CallbackPrintIteration(end=', ') &
            #       odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
            #       odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
            #       odl.solvers.CallbackShow(step=10))

            L = [1, 1e+2]
            # ud_vars = [0]
            ud_vars = [0, 1]

            print('experiment '+str(exp))
            # %%
            palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
            palm.run(niter)

            recon = palm.x[0].asarray()
            affine_params = palm.x[1].asarray()

            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = recon.tolist()

            exp+=1

    json.dump(dTV_regularised_recons, open('dTV/Results_MRI_dTV/dTV_regularised_SR_32_to_128_with_regis.json', 'w'))

exit()

# plotting

plt.figure()
plt.imshow(image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(downsampled_image_2, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recon, cmap=plt.cm.gray)


with open('dTV/Results_MRI_dTV/dTV_regularised_SR_32_to_128_with_regis.json') as f:
    d = json.load(f)

fig, axs = plt.subplots(10, 6)

#l2_norm_sinfo = np.sum(image**2)

for j, eta in enumerate(etas):
    ssim_vals = []
    for i, alpha in enumerate(alphas):
        # dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]

        recon = np.asarray(d['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)])

        axs[i, j].imshow(recon.T[::-1, :], cmap=plt.cm.gray)
        axs[i, j].axis("off")


plt.figure()

for j, eta in enumerate(etas):
    ssim_vals = []
    for i, alpha in enumerate(alphas[:-2]):
        # dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]

        recon = np.asarray(d['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)])

        axs[i, j].imshow(recon.T[::-1, :], cmap=plt.cm.gray)
        axs[i, j].axis("off")

        rescaled_recon = recon/np.sqrt(np.sum(recon**2))

        _, _, ssim_val = recon_error(rescaled_recon, image/np.sqrt(l2_norm_sinfo))
        ssim_vals.append(ssim_val)

    plt.plot(alphas[:-2], ssim_vals, label=r"$\eta$="+'{:.1e}'.format(eta))

plt.xscale("log")
plt.legend()
plt.xlabel(r"$\alpha$", fontsize=14)
plt.ylabel("SSIM", fontsize=14)
plt.title("SSIM values as "+r"$\eta, \alpha$"+ " are varied")


recon = np.asarray(d['alpha=' + '{:.1e}'.format(10.**0)]['eta=' + '{:.1e}'.format(10.**(-5))])

plt.figure()
plt.imshow(recon, cmap=plt.cm.gray)
plt.colorbar()

## line search to improve registration of low-res and high-res H images (translations only)
# upsampling of the low-res H image
h_coarse, w_coarse = downsampled_image_2.shape
h_fine, w_fine = image.shape
image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[h_fine, w_fine], dtype='float')
coarse_image_space_2 = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[h_coarse, w_coarse], dtype='float')

from skimage.transform import resize
upsampled_image = resize(low_res_image, (h_fine, w_fine))
upsampled_image_odl = image_space.element(upsampled_image)
#subsampling_op = ops.Subsampling(image_space, coarse_image_space_2)
#upsampled_image_odl = subsampling_op.adjoint(low_res_image_odl)

# defining dTV functionals
reg_func_coarse = fctls.directionalTotalVariationNonnegative(coarse_image_space_2, alpha=1, sinfo=downsampled_image_2)
reg_func_fine = fctls.directionalTotalVariationNonnegative(image_space, alpha=1, sinfo=image)

# linesearch
n = 100
h_coarse, w_coarse = low_res_image.shape
h_fine, w_fine = image.shape
dTV_losses_coarse = np.zeros((n, n))
dTV_losses_fine = np.zeros((n, n))

for i, a in enumerate(np.linspace(-0.05, 0.05, num=n)):
    for j, b in enumerate(np.linspace(-0.05, 0.05, num=n)):

        disp_horiz_coarse = a*np.ones((h_coarse, w_coarse))
        disp_vert_coarse = b*np.ones((h_coarse, w_coarse))
        displacement_coarse = np.asarray([disp_horiz_coarse, disp_vert_coarse])
        displ_image_coarse = defs.linear_deform(low_res_image_odl, coarse_image_space_2.tangent_bundle.element(displacement_coarse))

        dTV_losses_coarse[i, j] = reg_func_coarse(displ_image_coarse)

        disp_horiz_fine = a * np.ones((h_fine, w_fine))
        disp_vert_fine = b * np.ones((h_fine, w_fine))
        displacement_fine = np.asarray([disp_horiz_fine, disp_vert_fine])
        displ_image_fine = defs.linear_deform(upsampled_image_odl, image_space.tangent_bundle.element(displacement_fine))

        dTV_losses_fine[i, j] = reg_func_fine(displ_image_fine)

ind_vert_coarse, ind_horiz_coarse = np.unravel_index(np.argmin(dTV_losses_coarse), dTV_losses_coarse.shape)
ind_vert_fine, ind_horiz_fine = np.unravel_index(np.argmin(dTV_losses_fine), dTV_losses_fine.shape)

# corresponding best-aligned images
disp_horiz_coarse = np.linspace(-0.05, 0.05, num=n)[ind_horiz_coarse]*np.ones((h_coarse, w_coarse))
disp_vert_coarse = np.linspace(-0.05, 0.05, num=n)[ind_vert_coarse]*np.ones((h_coarse, w_coarse))
displacement_coarse = np.asarray([disp_horiz_coarse, disp_vert_coarse])
displ_image_coarse_best = defs.linear_deform(low_res_image_odl, coarse_image_space_2.tangent_bundle.element(displacement_coarse))

disp_horiz_fine = np.linspace(-0.05, 0.05, num=n)[ind_horiz_fine]*np.ones((h_fine, w_fine))
disp_vert_fine = np.linspace(-0.05, 0.05, num=n)[ind_vert_fine]*np.ones((h_fine, w_fine))
displacement_fine = np.asarray([disp_horiz_fine, disp_vert_fine])
displ_image_fine_best = defs.linear_deform(upsampled_image_odl, image_space.tangent_bundle.element(displacement_fine))


fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(5, 3))
axes[0, 0].imshow(np.asarray([2*downsampled_image_2/np.amax(downsampled_image_2), np.zeros(downsampled_image_2.shape),
                       np.zeros(downsampled_image_2.shape)]).transpose((1, 2, 0)))
axes[0, 1].imshow(np.asarray([2*downsampled_image_2/np.amax(downsampled_image_2), np.zeros(downsampled_image_2.shape),
                       low_res_image/np.amax(low_res_image)]).transpose((1, 2, 0)))
axes[0, 2].imshow(np.asarray([np.zeros(downsampled_image_2.shape), np.zeros(downsampled_image_2.shape),
                       low_res_image/np.amax(low_res_image)]).transpose((1, 2, 0)))

axes[1, 0].imshow(np.asarray([2*downsampled_image_2/np.amax(downsampled_image_2), np.zeros(downsampled_image_2.shape),
                              np.zeros(downsampled_image_2.shape)]).transpose((1, 2, 0)))
axes[1, 1].imshow(np.asarray([2*downsampled_image_2/np.amax(downsampled_image_2), np.zeros(downsampled_image_2.shape),
                       displ_image_coarse_best/np.amax(displ_image_coarse_best)]).transpose((1, 2, 0)))
axes[1, 2].imshow(np.asarray([np.zeros(downsampled_image_2.shape), np.zeros(downsampled_image_2.shape),
                       displ_image_coarse_best/np.amax(displ_image_coarse_best)]).transpose((1, 2, 0)))
axes[2, 0].imshow(np.asarray([2*image/np.amax(image), np.zeros(image.shape),
                       np.zeros(image.shape)]).transpose((1, 2, 0)))
axes[2, 1].imshow(np.asarray([2*image/np.amax(image), np.zeros(image.shape),
                       upsampled_image/np.amax(upsampled_image)]).transpose((1, 2, 0)))
axes[2, 2].imshow(np.asarray([np.zeros(image.shape), np.zeros(image.shape),
                       upsampled_image/np.amax(upsampled_image)]).transpose((1, 2, 0)))
axes[3, 0].imshow(np.asarray([2*image/np.amax(image), np.zeros(image.shape),
                       np.zeros(image.shape)]).transpose((1, 2, 0)))
axes[3, 1].imshow(np.asarray([2*image/np.amax(image), np.zeros(image.shape),
                       displ_image_fine_best/np.amax(displ_image_fine_best)]).transpose((1, 2, 0)))
axes[3, 2].imshow(np.asarray([np.zeros(image.shape), np.zeros(image.shape),
                       displ_image_fine_best/np.amax(displ_image_fine_best)]).transpose((1, 2, 0)))

fig.tight_layout()

### fourier upsampling of low-res image
height = 128
width = 128
data_height = 32
data_width = 32
padded_fourier_data = np.zeros((128, 128), dtype=complex)
padded_fourier_data[height // 2 - data_height // 2: height // 2 + data_height // 2,
        width // 2 - data_width // 2: width // 2 + data_width // 2] = fourier_data

recon_fourier_upsampled = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(padded_fourier_data)))
