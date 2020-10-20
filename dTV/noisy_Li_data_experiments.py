# created 02/10/20
# Aim: process noisy Li data resulting from relatively few averages

import numpy as np
import matplotlib.pyplot as plt
from processing import *

avgs = ['512', '1024', '4096', '8192']
Li_images = []

for avg in avgs:
    # files from Bearshare folder, labelled 7Li_Axial_512averages_1mmslicethickness etc.
    fourier_Li_real_im_padded = np.reshape(np.fromfile('dTV/Results_MRI_dTV/fid_Li_'+avg+'_averages', dtype=np.int32), (64, 128))
    fourier_Li_real_im = fourier_Li_real_im_padded[:, 1:65]
    fourier_Li_real_im = fourier_Li_real_im[::2, :]

    fourier_Li_real = fourier_Li_real_im[:, 1::2]
    fourier_Li_im = fourier_Li_real_im[:, ::2]
    fourier_Li = fourier_Li_real + fourier_Li_im*1j

    my_recon_Li = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_Li)))
    Li_image = np.abs(my_recon_Li)
    Li_images.append(Li_image)

for Li_image in Li_images:

    plt.figure()
    plt.imshow(Li_image, cmap=plt.cm.gray)



reg_types = ['TV']
reg_params = [1000.]

for reg_type in reg_types:
    model = VariationalRegClass('MRI', reg_type)

    for reg_param in reg_params:

        recons_bernoulli = model.regularised_recons_from_subsampled_data(fourier_Li, reg_param, niter=2000)

recon_rotated = np.abs(recons_bernoulli)[0].T[:, ::-1]

plt.figure()
plt.imshow(Li_image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recon_rotated, cmap=plt.cm.gray)

np.sum(np.square(Li_image - recon_rotated))
