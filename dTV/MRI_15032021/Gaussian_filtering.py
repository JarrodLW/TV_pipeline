import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res.mat')
image_H_high_res = mat['movingRegisteredRigid']

plt.figure()
plt.imshow(image_H_high_res, cmap=plt.cm.gray)
plt.colorbar()

filtered_image_1 = sp.ndimage.filters.gaussian_filter(image_H_high_res, 0.5)
filtered_image_2 = sp.ndimage.filters.gaussian_filter(image_H_high_res, 0.65)

plt.figure()
plt.imshow(image_H_high_res, cmap=plt.cm.gray)

plt.figure()
plt.imshow(filtered_image_1, cmap=plt.cm.gray)

plt.figure()
plt.imshow(filtered_image_2, cmap=plt.cm.gray)

np.save('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res_filtered.npy', filtered_image_2)

# filtering of Fourier reconstructions

from Utils import *
import odl
import myOperators as ops

variances = np.linspace(0, 2, num=21)
#filtered_recons

dir_Li = 'dTV/MRI_15032021/Data_24052021/Li_data/'
avgs = ['1024', '2048']
Li_range = range(8, 40)
low_res_data_width = 40

f_coeff_list = []

for i in Li_range:
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), (80, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, low_res_data_width)
    f_coeff_list.append(f_coeffs_unpacked)

# putting all the data into a single array, first index corresponding to number of averages, second the sample number,
# third and fourth indexing the data itself
f_coeff_arr = np.asarray(f_coeff_list)
f_coeff_arr_combined = np.zeros((len(avgs), 32, low_res_data_width, low_res_data_width), dtype='complex')

for avg_ind in range(len(avgs)):

    num = 2**avg_ind

    for i in range(num):
        data_arr = np.roll(f_coeff_arr, i, axis=0)
        for ele in range(len(f_coeff_list)//num):
            f_coeff_arr_combined[avg_ind, ele+i*len(f_coeff_list)//num, :, :] = np.sum(data_arr[num*ele:num*(ele+1)], axis=0)/num

complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                      shape=[40, 40], dtype='complex', interp='linear')
image_space = complex_space.real_space ** 2

# defining the forward op - I should do the subsampling in a more efficient way
fourier_transf = ops.RealFourierTransform(image_space)

TV_fully_averaged = np.load("dTV/MRI_15032021/Results_24052021/example_TV_recon_with_PDHG_on_32768.npy")
TV_fully_averaged_image = np.abs(TV_fully_averaged[0] + 1j*TV_fully_averaged[1])

for avg_ind in range(len(avgs)):
    avg = avgs[avg_ind]
    bias_variance_vals = np.zeros((2, len(variances)))
    for j, var in enumerate(variances):
        recon_filtered_images = np.zeros((32, 40, 40))
        for measurement in range(32):

            recon = fourier_transf.inverse(fourier_transf.range.element(
                [np.fft.fftshift(np.real(f_coeff_arr_combined[avg_ind, measurement, :, :])),
                                           np.fft.fftshift(np.imag(f_coeff_arr_combined[avg_ind, measurement, :, :]))])).asarray()

            recon_real_part = recon[0]
            recon_imag_part = recon[1]

            recon_real_part_filtered = sp.ndimage.filters.gaussian_filter(recon_real_part, var)
            recon_imag_part_filtered = sp.ndimage.filters.gaussian_filter(recon_imag_part, var)

            recon_filtered = recon_real_part_filtered + 1j*recon_imag_part_filtered
            recon_filtered_image = np.abs(recon_filtered)

            recon_filtered_images[measurement, :, :] = recon_filtered_image

        average_recon_image = np.average(recon_filtered_images, axis=0)
        variance = np.average(np.sum((recon_filtered_images - average_recon_image)**2, axis=(1, 2)))
        bias = np.sqrt(np.sum(np.square(average_recon_image - TV_fully_averaged_image)))

        bias_variance_vals[0, j] = bias
        bias_variance_vals[1, j] = variance

    np.save('dTV/MRI_15032021/Results_24052021/bias_variance_for_filtered_Fourier_'+str(avg)+'.npy', bias_variance_vals)

plt.figure()
plt.imshow(np.abs(recon_real_part+1j*recon_imag_part), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(recon_filtered), cmap=plt.cm.gray)

