import odl
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#barbara = np.asarray(imageio.imread('STEM_experiments/barbara.jpg'), dtype='float')
im = np.asarray(Image.open('STEM_experiments/9565_NMC811_after_second_delithiation.tif'), dtype=float)
barbara = im[1000:1750, 750:1500]
height, width = barbara.shape

image_space = odl.uniform_discr(min_pt=[0., 0.], max_pt=[float(height), float(width)],
                                            shape=[height, width], dtype='float')

data = image_space.element(barbara)
#noisy_data = data + odl.phantom.noise.white_noise(image_space, stddev=25)
noisy_data = data

# TV version

grad = odl.Gradient(image_space)
tangent_space = grad.range
div = odl.Divergence(tangent_space)

data_fidelity = odl.solvers.L2NormSquared(image_space).translated(noisy_data)
TV_semi_norm = odl.solvers.GroupL1Norm(tangent_space) # bad name!!!

#wavelet_transf = odl.trafos.wavelet.WaveletTransform(image_space, 'haar')
reg_param_1 = 100.
reg_param_2 = 0.05
#reg_param_3 = 0.
f = odl.solvers.SeparableSum(odl.solvers.ZeroFunctional(image_space), reg_param_2*odl.solvers.L2Norm(tangent_space))
#g = odl.solvers.SeparableSum(data_fidelity, reg_param_1*TV_semi_norm, reg_param_3*reg_param_2*odl.solvers.L1Norm(wavelet_transf.range))
g = odl.solvers.SeparableSum(data_fidelity, reg_param_1*TV_semi_norm)
# L = odl.operator.pspace_ops.ProductSpaceOperator([[odl.IdentityOperator(image_space), div],
#                                                   [grad, odl.ZeroOperator(tangent_space)],
#                                                   [0, wavelet_transf*div]])
L = odl.operator.pspace_ops.ProductSpaceOperator([[odl.IdentityOperator(image_space), div],
                                                  [grad, odl.ZeroOperator(tangent_space)]])

op_norm = 1.1 * odl.power_method_opnorm(L)
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm
niter = 175

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                  odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                  odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                  odl.solvers.CallbackShow(step=5))

# minimises f(x)+g(Lx)
x = L.domain.zero()
odl.solvers.pdhg(x, f, g, L, niter=niter, tau=tau, callback=cb, sigma=sigma)

geometric_part = x[0]
texture_part = div(x[1])

data.show()
noisy_data.show()
geometric_part.show()
texture_part.show()
(geometric_part+texture_part).show()


#######################################
# TGV version
grad = odl.Gradient(image_space)
tangent_space = grad.range
div = odl.Divergence(tangent_space)

data_fidelity = odl.solvers.L2NormSquared(image_space).translated(noisy_data)

reg_param_1 = 2000.
reg_param_2 = 1.
reg_param_3 = 0.1

Dx = odl.PartialDerivative(image_space, 0, method='backward', pad_mode='symmetric')
Dy = odl.PartialDerivative(image_space, 1, method='backward', pad_mode='symmetric')

# Create symmetrized operator and weighted space.
E = odl.operator.ProductSpaceOperator(
    [[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx], [0.5 * Dy, 0.5 * Dx]])
W = E.range

domain = odl.ProductSpace(image_space, tangent_space, tangent_space)

op_0 = odl.IdentityOperator(image_space)*odl.ComponentProjection(domain, 0) \
       + odl.ZeroOperator(tangent_space, range=image_space)*odl.ComponentProjection(domain, 1) \
       + div*odl.ComponentProjection(domain, 2)

op_1 = grad*odl.ComponentProjection(domain, 0) \
       - odl.IdentityOperator(tangent_space)*odl.ComponentProjection(domain, 1) \
       + odl.ZeroOperator(tangent_space)*odl.ComponentProjection(domain, 2)

op_2 = odl.ZeroOperator(image_space, range=W)*odl.ComponentProjection(domain, 0) \
       + E*odl.ComponentProjection(domain, 1) \
       + odl.ZeroOperator(tangent_space, range=W)*odl.ComponentProjection(domain, 2)

L = odl.BroadcastOperator(op_0, op_1, op_2)

# L = odl.operator.pspace_ops.ProductSpaceOperator([[odl.IdentityOperator(image_space), odl.ZeroOperator(tangent_space, range=image_space), div],
#                                                   [grad, -odl.IdentityOperator(tangent_space), odl.ZeroOperator(tangent_space)],
#                                                   [odl.ZeroOperator(image_space, range=W), E, odl.ZeroOperator(tangent_space, range=W)]])

f = odl.solvers.SeparableSum(odl.solvers.ZeroFunctional(image_space), odl.solvers.ZeroFunctional(tangent_space),
                             reg_param_3*odl.solvers.L2Norm(tangent_space))
g = odl.solvers.SeparableSum(data_fidelity, reg_param_1*odl.solvers.GroupL1Norm(tangent_space),
                             reg_param_1*reg_param_2*odl.solvers.GroupL1Norm(W))

op_norm = 1.1 * odl.power_method_opnorm(L)
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm
niter = 300

x = L.domain.zero()
odl.solvers.pdhg(x, f, g, L, niter=niter, tau=tau, callback=cb, sigma=sigma)

geometric_part = x[0]
texture_part = div(x[2])


# removing noise using wavelet denoising
from skimage.restoration import (denoise_wavelet, estimate_sigma)

noisy_background_patch = im[1700:, 1700:]

#sigma_est = estimate_sigma(texture_part.asarray(), multichannel=True, average_sigmas=True)
sigma_est = estimate_sigma(noisy_background_patch, multichannel=False, average_sigmas=False)

im_visushrink = denoise_wavelet(texture_part.asarray(), multichannel=False, convert2ycbcr=False,
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est, rescale_sigma=True)

denoised_texture_part = image_space.element(im_visushrink)

# filtered_texture_part = wavelet_transf.inverse(wavelet_transf(texture_part)*(np.abs(wavelet_transf(texture_part).asarray())>40.))
# texture_part.show()
# filtered_texture_part.show()

noisy_data.show()
denoised_texture_part.show()
(geometric_part+denoised_texture_part).show()

fourier_transf = odl.trafos.fourier.FourierTransform(image_space)
fourier = fourier_transf(texture_part).asarray()

plt.imshow(np.log(np.abs(fourier)), cmap=plt.cm.gray)
plt.colorbar()

# shearlet denoising
import pyshearlab

X = texture_part.asarray()
sigma=30
scales=4
thresholdingFactor=3

shearletSystem = pyshearlab.SLgetShearletSystem2D(0, X.shape[0], X.shape[1], scales)
coeffs = pyshearlab.SLsheardec2D(X, shearletSystem)

oldCoeffs = coeffs.copy()
weights = np.ones(coeffs.shape)

for j in range(len(shearletSystem["RMS"])):
    weights[:, :, j] = shearletSystem["RMS"][j] * np.ones((X.shape[0], X.shape[1]))

coeffs = np.real(coeffs)
zero_indices = np.abs(coeffs) / (thresholdingFactor * weights * sigma) < 1
coeffs[zero_indices] = 0

Xrec = pyshearlab.SLshearrec2D(coeffs, shearletSystem)

plt.figure()
plt.imshow(X, cmap=plt.cm.gray)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(Xrec, cmap=plt.cm.gray)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.abs(Xrec), cmap=plt.cm.gray)
plt.colorbar()
plt.show()


# automating retrieval plane separation distance
x[1].show()
#gradients = grad(denoised_texture_part)
gradients = grad(data)
gradients[0].show()
gradients[1].show()
np.abs(gradients[0]).show()

denoised_texture_part_shifted = denoised_texture_part - np.amin(denoised_texture_part)

denoised_texture_part_shifted.show()

#np.exp(gradients[0]).show()

x, y = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
u = gradients[0].asarray()[::(data.shape[0]//50), ::(data.shape[1]//50)]
v = gradients[1].asarray()[::(data.shape[0]//50), ::(data.shape[1]//50)]
# u_normalised = u/np.sqrt(u**2+v**2)
# v_normalised = v/np.sqrt(u**2+v**2)
# plt.quiver(x, y, u_normalised, v_normalised)
plt.quiver(x, y, u, v)
plt.show()

# rough
wavelet_transf = odl.trafos.wavelet.WaveletTransform(image_space, 'haar')

stripes = np.zeros((height, width))
stripes[::2, :] = 1
stripes_normalised = image_space.element(stripes/np.sqrt(np.sum(np.square(stripes))))

white_noise = odl.phantom.noise.white_noise(image_space, stddev=10, seed=108)
white_noise_normalised = white_noise/np.sqrt(np.sum(np.square(white_noise)))

np.sum(np.abs(wavelet_transf(stripes_normalised).asarray()))
np.sum(np.abs(wavelet_transf(white_noise_normalised)))


# rough 2
from skimage.restoration import (denoise_wavelet, estimate_sigma)

wavelet_denoised_texture = denoise_wavelet(texture_part.asarray(), multichannel=False, convert2ycbcr=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)

wavelet_denoised_texture_odl = image_space.element(wavelet_denoised_texture)

wavelet_denoised_texture_odl.show()
texture_part.show()

wavelet_denoised_im = denoise_wavelet(noisy_data.asarray(), multichannel=False, convert2ycbcr=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)

wavelet_denoised_im_odl = image_space.element(wavelet_denoised_im)

data.show()
noisy_data.show()
(geometric_part+wavelet_denoised_texture_odl).show()

wavelet_denoised_im_odl.show()

comparison = odl.solvers.L2NormSquared(image_space).translated(data)

comparison(wavelet_denoised_im_odl)
comparison(geometric_part+wavelet_denoised_texture_odl)



# Fourier stuff

plane_patch_orig = barbara[150:300, 150:300]
plane_patch_text = denoised_texture_part.asarray()[150:300, 150:300]

fourier_plane_patch_orig = np.fft.fft2(plane_patch_orig)
fourier_plane_patch_orig_vis = np.fft.fftshift(np.abs(fourier_plane_patch_orig))

fourier_plane_patch_text = np.fft.fft2(plane_patch_text)
fourier_plane_patch_text_vis = np.fft.fftshift(np.abs(fourier_plane_patch_text))

filtered_plane_patch_orig = np.fft.ifft2(fourier_plane_patch_orig*(np.abs(fourier_plane_patch_orig)>0.1*1e6))
filtered_plane_patch_text = np.fft.ifft2(fourier_plane_patch_text*(np.abs(fourier_plane_patch_text)>0.1*1e6))

plt.figure()
plt.imshow(fourier_plane_patch_orig_vis[60:90, 60:90], cmap=plt.cm.gray, vmax=150000)
plt.axis("off")
plt.colorbar()

plt.figure()
plt.imshow(fourier_plane_patch_text_vis[60:90, 60:90], cmap=plt.cm.gray)
plt.axis("off")
plt.colorbar()

plt.figure()
plt.imshow(np.fft.fftshift((np.abs(fourier_plane_patch_orig)>0.1*1e6)), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.real(filtered_plane_patch_orig), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(fourier_plane_patch_text_vis, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.fft.fftshift((np.abs(fourier_plane_patch_text)>0.1*1e6)), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.real(filtered_plane_patch_text), cmap=plt.cm.gray)

######
salt_patch_orig = barbara[375:525, 150:300]
salt_patch_text = denoised_texture_part.asarray()[375:525, 150:300]

fourier_salt_patch_orig = np.fft.fft2(salt_patch_orig)
fourier_salt_patch_orig_vis = np.fft.fftshift(np.abs(fourier_salt_patch_orig))

fourier_salt_patch_text = np.fft.fft2(salt_patch_text)
fourier_salt_patch_text_vis = np.fft.fftshift(np.abs(fourier_salt_patch_text))

filtered_salt_patch_orig = np.fft.ifft2(fourier_salt_patch_orig*(np.abs(fourier_salt_patch_orig)>0.01*1e6))
filtered_salt_patch_text = np.fft.ifft2(fourier_salt_patch_text*(np.abs(fourier_salt_patch_text)>0.01*1e6))

plt.figure()
plt.imshow(fourier_salt_patch_orig_vis[60:90, 60:90], cmap=plt.cm.gray, vmax=4*1e4)
plt.axis("off")
plt.colorbar()

plt.figure()
plt.imshow(fourier_salt_patch_text_vis[60:90, 60:90], cmap=plt.cm.gray)
plt.axis("off")
plt.colorbar()

plt.figure()
plt.imshow(np.fft.fftshift(np.abs(fourier_salt_patch_orig)*(np.abs(fourier_salt_patch_orig)>0.01*1e6)), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.real(filtered_salt_patch_orig), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.fft.fftshift(np.abs(fourier_salt_patch_text)*(np.abs(fourier_salt_patch_text)>0.01*1e6)), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.real(filtered_salt_patch_text), cmap=plt.cm.gray)

# playing around with Fourier transformation

checkerboard = np.zeros((20, 20))
stripes = np.zeros((20, 20))
checkerboard[1::2, ::2]=1
checkerboard[::2, 1::2]=1
checkerboard -= 1/2
stripes[::2, :]=1
stripes -= 1/2

f_coeffs_checkerboard = np.fft.fftshift(np.fft.fft2(checkerboard))
f_coeffs_stripes = np.fft.fftshift(np.fft.fft2(stripes))

plt.figure()
plt.imshow(checkerboard, cmap=plt.cm.gray)

plt.figure()
plt.imshow(stripes, cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(f_coeffs_checkerboard), vmax=1, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(f_coeffs_stripes), cmap=plt.cm.gray)
plt.colorbar()


# plt.figure()
# plt.imshow(np.abs(np.real(f_coeffs_checkerboard)), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(np.abs(np.imag(f_coeffs_checkerboard)), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(np.abs(np.real(f_coeffs_stripes)), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(np.abs(np.imag(f_coeffs_stripes)), cmap=plt.cm.gray)

# shearlet transform of salt/plane
scales=2

plane_patch_text = denoised_texture_part.asarray()[150:300, 150:300]
salt_patch_text = denoised_texture_part.asarray()[375:525, 150:300]

#plane_patch_text = denoised_texture_part.asarray()[150:200, 150:200]
#salt_patch_text = denoised_texture_part.asarray()[375:425, 150:200]

shearletSystem = pyshearlab.SLgetShearletSystem2D(0, plane_patch_text.shape[0], plane_patch_text.shape[1], scales)

coeffs_plane = pyshearlab.SLsheardec2D(plane_patch_text, shearletSystem)
coeffs_salt = pyshearlab.SLsheardec2D(salt_patch_text, shearletSystem)

fourier_salt_patch_text = np.fft.fft2(salt_patch_text)
fourier_plane_patch_text = np.fft.fft2(plane_patch_text)
#fourier_salt_patch_text_vis = np.fft.fftshift(np.abs(fourier_salt_patch_text))

threshold=30000

plt.figure()
plt.imshow(np.abs(np.fft.fftshift(fourier_salt_patch_text)), vmin=5000, cmap=plt.cm.gray)
plt.axis("off")
plt.colorbar()

plt.figure()
plt.imshow(np.abs(np.fft.fftshift(fourier_salt_patch_text))[50:100, 50:100], vmin=threshold, cmap=plt.cm.gray)
plt.axis("off")
plt.colorbar()


f_plane_filtered=fourier_plane_patch_text*(np.abs(fourier_plane_patch_text)>threshold)
f_plane_filtered_recon = np.fft.ifft2(f_plane_filtered)

f_salt_filtered=fourier_salt_patch_text*(np.abs(fourier_salt_patch_text)>threshold)
f_salt_filtered_recon = np.fft.ifft2(f_salt_filtered)

plt.figure()
plt.imshow(np.real(f_plane_filtered_recon), cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
plt.imshow(np.real(f_salt_filtered_recon), cmap=plt.cm.gray)
plt.axis("off")

plt.figure()
plt.imshow(plane_patch_text, cmap=plt.cm.gray)

plt.figure()
plt.imshow(salt_patch_text, cmap=plt.cm.gray)



