from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from processing import *

im_0 = Image.open('STEM_experiments/9534_NMC811_initial state.tif')
im_1 = Image.open('STEM_experiments/9545_NMC811_after_first_delithiation.tif')
im_2 = Image.open('STEM_experiments/9555_NMC811_after_first_lithiation.tif')
im_3 = Image.open('STEM_experiments/9565_NMC811_after_second_delithiation.tif')
#im.show()

im_0_arr = np.asarray(im_0, dtype=float)
im_1_arr = np.asarray(im_1, dtype=float)
im_2_arr = np.asarray(im_2, dtype=float)
im_3_arr = np.asarray(im_3, dtype=float)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
ax0.imshow(im_0_arr, cmap=plt.cm.gray)
ax0.axis("off")
ax1.imshow(im_1_arr, cmap=plt.cm.gray)
ax1.axis("off")
ax2.imshow(im_2_arr, cmap=plt.cm.gray)
ax2.axis("off")
ax3.imshow(im_3_arr, cmap=plt.cm.gray)
ax3.axis("off")

model = VariationalRegClass('STEM', 'TV')
reg_param = 100.
recons = model.regularised_recons_from_subsampled_data(im_3_arr, reg_param, niter=100)
recon = recons[0].T[:, ::-1]

plt.figure()
plt.imshow(im_3_arr, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recon, cmap=plt.cm.gray)

from skimage.restoration import (denoise_wavelet, estimate_sigma)

im_bayes = denoise_wavelet(im_3_arr, multichannel=False, convert2ycbcr=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)

sigma_est = estimate_sigma(im_3_arr)

im_visushrink = denoise_wavelet(im_3_arr, multichannel=False, convert2ycbcr=False,
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est, rescale_sigma=True)

im_visushrink2 = denoise_wavelet(im_3_arr, multichannel=False, convert2ycbcr=False,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est/2, rescale_sigma=True)
im_visushrink4 = denoise_wavelet(im_3_arr, multichannel=False, convert2ycbcr=False,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est/4, rescale_sigma=True)

plt.figure()
plt.imshow(im_3_arr, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(im_bayes, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(im_visushrink, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(im_visushrink2, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(im_visushrink4, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(im_3_arr - im_bayes, cmap=plt.cm.gray)
plt.colorbar()


