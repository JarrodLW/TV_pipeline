import numpy as np
from scipy import stats
import json
import matplotlib.pyplot as plt
import scipy as sp
import itertools

dir = 'dTV/7Li_1H_MRI_Data_31112020/'

def unpacking_fourier_coeffs(arr):

    fourier_real_im = arr[:, 1:65]
    fourier_real_im = fourier_real_im[::2, :]

    fourier_real = fourier_real_im[:, 1::2]
    fourier_im = fourier_real_im[:, ::2]
    fourier = fourier_real + fourier_im * 1j

    return fourier

f_coeff_list = []
recon_list = []

for i in range(2, 36):
    f_coeffs = np.reshape(np.fromfile(dir + 'Li2SO4/'+str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)
    recon = 32*np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs_unpacked))) # inserting factor of 32 to ensure transform is unitary
    recon_list.append(recon)

plt.imshow(np.abs(recon_list[0]), cmap=plt.cm.gray)

recon_arr = np.asarray(recon_list)
# grabbing the pixels around the border of each reconstruction
block_1 = np.reshape(recon_arr[:, :8, :24], (34, 8*24))
block_2 = np.reshape(recon_arr[:, 8:, :8], (34, 8*24))
block_3 = np.reshape(recon_arr[:, 24:, 8:], (34, 8*24))
block_4 = np.reshape(recon_arr[:, :24, 24:], (34, 8*24))

border_pixels = np.concatenate((block_1, block_2, block_3, block_4), axis=1)
border_pixels_real_part = np.real(border_pixels)
border_pixels_imag_part = np.imag(border_pixels)

# means and standard deviations for 512 data
mean_for_each_pixel_real = np.mean(border_pixels_real_part, axis=0)
mean_real = np.mean(border_pixels_real_part)

mean_for_each_pixel_imag = np.mean(border_pixels_imag_part, axis=0)
mean_imag = np.mean(border_pixels_imag_part)

std_for_each_pixel_real = np.std(border_pixels_real_part, axis=0)
std_real = np.std(border_pixels_real_part)

std_for_each_imag_real = np.std(border_pixels_imag_part, axis=0)
std_imag = np.std(border_pixels_imag_part)

# histograms
plt.figure()
plt.hist(np.ndarray.flatten(border_pixels_real_part), bins=100)

plt.figure()
plt.hist(np.ndarray.flatten(border_pixels_imag_part), bins=100)

# standard deviations for varying numbers of averages
border_pixels_real_part_paired = (1/2)*(border_pixels_real_part[:-2, :][::2, :] + border_pixels_real_part[:-2, :][1::2, :])
border_pixels_real_part_paired_2 = (1/2)*(border_pixels_real_part_paired[::2, :] + border_pixels_real_part_paired[1::2, :])
border_pixels_real_part_paired_3 = (1/2)*(border_pixels_real_part_paired_2[::2, :] + border_pixels_real_part_paired_2[1::2, :])

np.std(border_pixels_real_part_paired)
np.std(border_pixels_real_part_paired_2)
np.std(border_pixels_real_part_paired_3)

np.std(border_pixels_real_part_paired, axis=0)[:20]
np.std(border_pixels_real_part_paired_2, axis=0)[:20]
np.std(border_pixels_real_part_paired_3, axis=0)[:20]

# covariances of pixels
cov_matrix = np.cov(border_pixels_real_part.T)

plt.imshow(cov_matrix, cmap=plt.cm.gray)
plt.colorbar()

## Morozov
allowed_data_res_real = 32*std_real
allowed_data_res_imag = 32*std_imag

## Hypothesis testing
# testing that each pixel has zero mean (assuming iid samples, normal, unknown variance) - student-t
pop_mean = np.zeros(border_pixels_real_part.shape[1])
p_vals_real = sp.stats.ttest_1samp(border_pixels_real_part, pop_mean, axis=0)[1]
p_vals_imag = sp.stats.ttest_1samp(border_pixels_imag_part, pop_mean, axis=0)[1]

# testing that collected measurements (all pixels) have zero mean (assuming iid samples, normal, unknown variance) - student-t
pop_mean = 0
p_val_real = sp.stats.ttest_1samp(np.ndarray.flatten(border_pixels_real_part), pop_mean)[1]
p_val_imag = sp.stats.ttest_1samp(np.ndarray.flatten(border_pixels_imag_part), pop_mean)[1]

# testing normality of each pixel, unknown mean and variance (assuming iid samples) - Shapiro-Wilks
p_vals_real = []
p_vals_imag = []
for i in range(border_pixels_real_part.shape[1]):
    p_vals_real.append(stats.shapiro(border_pixels_real_part[:, i])[1])
    p_vals_imag.append(stats.shapiro(border_pixels_imag_part[:, i])[1])

# testing normality of each pixel - D'Agostino-Pearson
p_vals_real = []
p_vals_imag = []
for i in range(border_pixels_real_part.shape[1]):
    p_vals_real.append(stats.normaltest(border_pixels_real_part[:, i])[1])
    p_vals_imag.append(stats.normaltest(border_pixels_imag_part[:, i])[1])

# testing that pairs of pixels have same means (assuming iid samples, same variance) - student-t (2 sample)
# does this assume normality?
random_pixel_nums = np.random.choice(np.arange(750), size = 100)
p_vals_real = []
p_vals_imag = []

for (i, j) in list(itertools.combinations(random_pixel_nums, 2)):
    p_val_real = sp.stats.ttest_ind(border_pixels_real_part[:, i], border_pixels_real_part[:, j])[1]
    p_val_imag = sp.stats.ttest_ind(border_pixels_real_part[:, i], border_pixels_imag_part[:, j])[1]
    p_vals_real.append(p_val_real)
    p_vals_imag.append(p_val_imag)

np.amin(p_vals_real)
np.amin(p_vals_imag) # how to interpret this?

# testing that pairs of pixels have same means (assuming iid samples, not assuming same variance) - Welch
# does this assume normality?
random_pixel_nums = np.random.choice(np.arange(750), size=100)
p_vals_real = []
p_vals_imag = []

for (i, j) in list(itertools.combinations(random_pixel_nums, 2)):
    p_val_real = sp.stats.ttest_ind(border_pixels_real_part[:, i], border_pixels_real_part[:, j], equal_var=False)[1]
    p_val_imag = sp.stats.ttest_ind(border_pixels_real_part[:, i], border_pixels_imag_part[:, j], equal_var=False)[1]
    p_vals_real.append(p_val_real)
    p_vals_imag.append(p_val_imag)

np.amin(p_vals_real)
np.amin(p_vals_imag)  # how to interpret this?

# testing equivalence of means of real and imaginary parts, thinking of pixels as iid - student-t, 2 sample
# does this assume normality?
sp.stats.ttest_ind(np.ndarray.flatten(border_pixels_real_part), np.ndarray.flatten(border_pixels_imag_part))

# testing equivalence of variances of real and imaginary parts, thinking of pixels as iid normal - Bartlett
sp.stats.bartlett(np.ndarray.flatten(border_pixels_real_part), np.ndarray.flatten(border_pixels_imag_part))

# testing equivalence of variances of real and imaginary parts, thinking of pixels as iid, allowing for non-nomrality - Levene
sp.stats.levene(np.ndarray.flatten(border_pixels_real_part), np.ndarray.flatten(border_pixels_imag_part))

# testing pairwise independence of pixels



### some statistics for Li_LS dataset

f_coeff_list = []
recon_list = []

for i in range(2, 36):
    f_coeffs = np.reshape(np.fromfile(dir + 'Li_LS/'+str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)
    recon = 32*np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs_unpacked))) # inserting factor of 32 to ensure transform is unitary
    recon_list.append(recon)

recon_arr = np.asarray(recon_list)
# grabbing the pixels around the border of each reconstruction
block_1 = np.reshape(recon_arr[:, :8, :24], (34, 8*24))
block_2 = np.reshape(recon_arr[:, 8:, :8], (34, 8*24))
block_3 = np.reshape(recon_arr[:, 24:, 8:], (34, 8*24))
block_4 = np.reshape(recon_arr[:, :24, 24:], (34, 8*24))

border_pixels = np.concatenate((block_1, block_2, block_3, block_4), axis=1)
border_pixels_real_part = np.real(border_pixels)
border_pixels_imag_part = np.imag(border_pixels)

32*np.sqrt(np.std(border_pixels_real_part)**2 + np.std(border_pixels_imag_part)**2)

32*np.std((1/2)*(border_pixels_imag_part[:17, :] + border_pixels_imag_part[17:, :]))


