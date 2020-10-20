# created 10/09/2020
# experiments with 1H data  (low and high res) via inverse problem...


import h5py
import numpy as np
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json
import dTV.myAlgorithms as algs
import matplotlib.pyplot as plt
import os
import odl
import dTV.myOperators as ops
from Utils import *
from skimage.measure import block_reduce

fourier_H_real_im = np.reshape(np.fromfile('dTV/Results_MRI_dTV/fid_H', dtype=np.int32), (128, 256))

# for some reason the fid_7 data comes padded: all even rows are zero and the first and 65-128th columns are zeros
# once the zeros are removed, you get a 32x64 array that can then be unpacked into real-im parts as before
fourier_Li_real_im_padded = np.reshape(np.fromfile('dTV/Results_MRI_dTV/fid_Li_actual', dtype=np.int32), (64, 128))
fourier_Li_real_im = fourier_Li_real_im_padded[:, 1:65]
fourier_Li_real_im = fourier_Li_real_im[::2, :]

# same for the low res H data
fourier_H_low_res_real_im_padded = np.reshape(np.fromfile('dTV/Results_MRI_dTV/fid_H_low_res', dtype=np.int32), (64, 128))
fourier_H_low_res_real_im = fourier_H_low_res_real_im_padded[:, 1:65]
fourier_H_low_res_real_im = fourier_H_low_res_real_im[::2, :]

fourier_H_real = fourier_H_real_im[:, ::2]
fourier_H_im = fourier_H_real_im[:, 1::2]
fourier_H = fourier_H_real + fourier_H_im*1j

fourier_Li_real = fourier_Li_real_im[:, ::2]
fourier_Li_im = fourier_Li_real_im[:, 1::2]
fourier_Li = fourier_Li_real + fourier_Li_im*1j

fourier_H_low_res_real = fourier_H_low_res_real_im[:, ::2]
fourier_H_low_res_im = fourier_H_low_res_real_im[:, 1::2]
fourier_H_low_res = fourier_H_low_res_real + fourier_H_low_res_im*1j

my_recon_H = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_H)))
my_recon_Li = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_Li)))
my_recon_H_low_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_H_low_res)))
#my_recon_H_low_res = np.fft.fftshift(np.fft.ifft2(fourier_H_low_res))

# for some reason, the recons aren't automatically aligned. Here I rotate both so that they align, but also to match the
# orientation of Pooja and Claire's recons
my_recon_H_rotated = my_recon_H.T[::-1, :]
my_recon_Li_rotated = my_recon_Li.T[:, ::-1]
my_recon_H_low_res_rotated = my_recon_H_low_res.T[:, ::-1]

### working instead with the unpacked raw data provided
f = open('dTV/Results_MRI_dTV/1H_lowRes_imaginaryRaw_noZeros', 'r')
fourier_data_im = np.genfromtxt(f, delimiter=' ').T
f = open('dTV/Results_MRI_dTV/1H_lowRes_realRaw_noZeros', 'r')
fourier_data_real = np.genfromtxt(f, delimiter=' ').T

fourier_data = (fourier_data_real + fourier_data_im*1j)
recon_from_unpacked_data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_data)))

plt.figure()
plt.imshow(np.abs(my_recon_H_rotated), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.real(my_recon_H_rotated), cmap=plt.cm.gray)
plt.axis("off")
plt.colorbar()

plt.figure()
plt.imshow(np.imag(my_recon_H_rotated), cmap=plt.cm.gray)
plt.axis("off")
plt.colorbar()


plt.figure()
plt.imshow(np.abs(my_recon_Li_rotated), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(my_recon_H_low_res_rotated), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.real(my_recon_H_low_res_rotated), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.imag(my_recon_H_low_res_rotated), cmap=plt.cm.gray)
plt.colorbar()

gamma = 0.995
strong_cvx = 1e-1
niter_prox = 20
niter = 150

#alphas = [10**(i) for i in np.arange(7)]
#etas = [10**(-i-1) for i in range(6)]

#alphas = [10. ** (i - 5) for i in np.arange(10)]
#etas = [10. ** (-i) for i in np.arange(6)]

alphas = [0.]
etas = [1.]

Yaff = odl.tensor_space(6)

#sinfo_high_res = np.abs(my_recon_H).T[::-1, :]).T[::-1, :]
sinfo_high_res = np.abs(my_recon_H)
sinfo_med_res = block_reduce(sinfo_high_res, block_size=(2, 2), func=np.mean)
sinfo_low_res = block_reduce(sinfo_high_res, block_size=(4, 4), func=np.mean)


sinfos = {}
sinfos['high_res'] = sinfo_high_res
#sinfos['med_res'] = sinfo_med_res
#sinfos['low_res'] = sinfo_low_res


for dict_key in sinfos.keys():

    sinfo = sinfos[dict_key]

    height, width = sinfo.shape
    complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                          shape=[height, width], dtype='complex', interp='linear')
    image_space = complex_space.real_space ** 2

    X = odl.ProductSpace(image_space, Yaff)

    # adding noise to the data

    #fourier_data_real += 4000 * odl.phantom.white_noise(image_space[0]).asarray()
    #fourier_data_im += 4000 * odl.phantom.white_noise(image_space[0]).asarray()
    #naive_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_data_real + fourier_data_im*1j)))

    # defining the forward op - I should do the subsampling in a more efficient way
    fourier_transf = ops.RealFourierTransform(image_space)
    data_height, data_width = fourier_data.shape

    subsampling_arr = np.zeros((height, width))
    subsampling_arr[height//2 - data_height//2: height//2 + data_height//2, width//2 - data_width//2: width//2 + data_width//2] = 1
    subsampling_arr_doubled = np.array([subsampling_arr, subsampling_arr])

    forward_op = fourier_transf.range.element(subsampling_arr_doubled) * fourier_transf

    sinfo = complex_space.real_space.element(sinfo)
    padded_fourier_data_real = np.zeros((height, width))
    padded_fourier_data_real[height//2 - data_height//2: height//2 + data_height//2,
        width//2 - data_width//2: width//2 + data_width//2] = fourier_data_real

    padded_fourier_data_im = np.zeros((height, width))
    padded_fourier_data_im[height // 2 - data_height // 2: height // 2 + data_height // 2,
        width // 2 - data_width // 2: width // 2 + data_width // 2] = fourier_data_im

    data_odl = forward_op.range.element([padded_fourier_data_real, padded_fourier_data_im])

    # Set some parameters and the general TV prox options
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = niter_prox

    reg_affine = odl.solvers.ZeroFunctional(Yaff)
    x0 = X.zero()

    f = fctls.DataFitL2Disp(X, data_odl, forward_op)

    dTV_regularised_recons = {}
    for alpha in alphas:
        dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)] = {}
        for eta in etas:
            reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                                            gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                                            prox_options=prox_options)

            g = odl.solvers.SeparableSum(reg_im, reg_affine)

            cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                  odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                  odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True))

            L = [1, 1e+2]
            ud_vars = [0]

            # %%
            palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
            print('Experiment '+'alpha='+str(alpha)+' eta='+str(eta))

            palm.run(niter)

            recon = palm.x[0].asarray()

            dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] = recon.tolist()

    json.dump(dTV_regularised_recons, open('dTV/Results_MRI_dTV/dTV_regularised_recons_MRI.json', 'w'))


# plotting

with open('dTV/Results_MRI_dTV/dTV_regularised_recons_MRI.json') as f:
    d = json.load(f)

fig, axs = plt.subplots(10, 6)

for i, alpha in enumerate(alphas):
    for j, eta in enumerate(etas):
        # dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]

        recon = np.asarray(d['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)])
        image = np.abs(recon[0] + recon[1]*1j).T[:, ::-1]

        axs[i, j].imshow(image, cmap=plt.cm.gray)
        axs[i, j].axis("off")

#plt.tight_layout(w_pad=0.1, rect=[0.2, 0, 0.2, 1])

#plt.figure()
#plt.imshow(np.abs(recon[0]+recon[1]*1j), cmap=plt.cm.gray)

#plt.figure()
#plt.imshow(sinfo, cmap=plt.cm.gray)


d2 = dTV_regularised_recons['alpha=0.0e+00']
recon_as_list = d2['eta=1.0e-03']
recon = np.asarray(recon_as_list)
recon = recon[0] + recon[1]*1j

plt.figure()
plt.imshow(sinfo, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(recon_from_unpacked_data), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(naive_recon), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(recon), cmap=plt.cm.gray)

# checking registration
plt.figure()
plt.imshow(np.asarray([2*sinfo_low_res/np.amax(sinfo_low_res), np.zeros(sinfo_low_res_rotated.shape),
                       np.abs(recon_from_unpacked_data)/np.amax(np.abs(recon_from_unpacked_data))]).transpose((1, 2, 0)))

image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[128, 128], dtype='float')
coarse_image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[32, 32], dtype='float')
subsampling_op = ops.Subsampling(image_space, coarse_image_space)
upsampled_image = subsampling_op.adjoint(np.abs(recon_from_unpacked_data)).asarray()

plt.figure()
plt.imshow(np.asarray([4*sinfo_high_res/np.amax(sinfo_high_res), np.zeros(sinfo_high_res.shape),
                       upsampled_image/np.amax(upsampled_image)]).transpose((1, 2, 0)))

# are low-res H image and downsampled high-res H image aligned?

height = 128
width = 128
cross_gradients = np.zeros((height, width))
for i in range(-5, 6):
    for j in range(-5, 6):
        #cross_grad = fctls.cross_gradient(np.roll(sinfo_low_res, (i, j)), np.abs(recon_from_unpacked_data))
        cross_grad = fctls.cross_gradient(np.roll(sinfo_high_res, (i, j)), upsampled_image)
        cross_gradients[i, j] = cross_grad

ind = np.unravel_index(np.argmin(cross_gradients), cross_gradients.shape)

#best_aligned = np.roll(sinfo_low_res, ind)
best_aligned = np.roll(sinfo_high_res, ind)

plt.figure()
plt.imshow(np.log(np.abs(fourier_data)), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.log(np.abs(fourier_H)), cmap=plt.cm.gray)