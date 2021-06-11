import numpy as np
import json
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform
from Utils import *

avgs = ['512', '1024', '2048', '4096', '8192']
output_dims = [int(32), int(64)]

date = '24052021'
#date = '15032021'

dir_Li = 'dTV/MRI_15032021/Data_' + date + '/Li_data/'
dir_H = 'dTV/MRI_15032021/Data_' + date + '/H_data/'

if date=='15032021':
    H_index_low_res = 5
    H_index_high_res = 6
    low_res_shape = (64, 128)
    Li_range = range(3, 35)
    low_res_data_width = 32

elif date=='24052021':
    H_index_low_res = 29
    H_index_high_res = 32
    low_res_shape = (80, 128)
    Li_range = range(8, 40)
    low_res_data_width = 40

## 1H reconstructions

# low-res
#f_coeffs = np.reshape(np.fromfile(dir_H +str(H_index_low_res)+'/fid', dtype=np.int32), (64, 128))
f_coeffs = np.reshape(np.fromfile(dir_H +str(H_index_low_res)+'/fid', dtype=np.int32), low_res_shape)
f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, low_res_data_width)
recon_low_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs_unpacked)))

plt.figure()
plt.imshow(np.abs(recon_low_res), cmap=plt.cm.gray)
plt.colorbar()

# high-res
f_coeffs = np.reshape(np.fromfile(dir_H +str(H_index_high_res)+'/fid', dtype=np.int32), (128, 256))
f_coeffs = f_coeffs[:, 1::2] + 1j*f_coeffs[:, ::2]
recon_high_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs)))

plt.figure()
plt.imshow(np.abs(recon_high_res), cmap=plt.cm.gray)
plt.colorbar()

## 7Li reconstruction from long run

if date=='24052021':

    f_coeffs = np.reshape(np.fromfile(dir_Li +'4/fid', dtype=np.int32), (80, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, low_res_data_width)
    fourier_long_run = np.fft.fftshift(f_coeffs_unpacked)/32
    recon_long_run = np.fft.fftshift(np.fft.ifft2(fourier_long_run))

    plt.figure()
    plt.imshow(np.abs(recon_long_run), cmap=plt.cm.gray)
    plt.colorbar()

## 7Li reconstructions
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

# generating the reconstructions

fully_averaged = np.average(f_coeff_arr, axis=0)
fully_averaged_shifted = np.fft.fftshift(fully_averaged)

recon_fully_averaged = np.fft.fftshift(np.fft.ifft2(fully_averaged_shifted))
stdev_fully_averaged = np.sqrt(np.sum(np.square(np.abs(recon_fully_averaged[:8, :]))))
stdev_long_run = np.sqrt(np.sum(np.square(np.abs(recon_long_run[:8, :]))))

if date=='24052021': # averaging with the data from the full run

    fully_averaged_shifted_2 = (fully_averaged_shifted + (stdev_fully_averaged/stdev_long_run)*fourier_long_run)/2

np.save('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_24052021/32768_data.npy', fully_averaged_shifted_2)

recon_fully_averaged_2 = np.fft.fftshift(np.fft.ifft2(fully_averaged_shifted_2))

stdev_fully_averaged_2 = np.sqrt(np.sum(np.square(np.abs(recon_fully_averaged_2[:8, :]))))

plt.figure()
plt.imshow(np.abs(recon_long_run), cmap=plt.cm.gray, vmax=160)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(recon_fully_averaged), cmap=plt.cm.gray, vmax=160)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(recon_fully_averaged_2), cmap=plt.cm.gray, vmax=160)
plt.colorbar()



recon_arr = np.zeros((len(avgs), 32, low_res_data_width, low_res_data_width), dtype='complex')
upsampled_size = 2*low_res_data_width
recon_arr_upsampled = np.zeros((len(avgs), 32, upsampled_size, upsampled_size), dtype='complex')

for avg_ind in range(len(avgs)):
    for i in range(32):

        f_data = f_coeff_arr_combined[avg_ind, i, :, :]
        f_data_shifted = np.fft.fftshift(f_data)
        recon = np.fft.fftshift(np.fft.ifft2(f_data_shifted))

        recon_arr[avg_ind, i, :, :] = recon

        f_data_padded = np.zeros((upsampled_size, upsampled_size), dtype='complex')
        f_data_padded[upsampled_size // 2 - low_res_data_width//2:upsampled_size // 2 + low_res_data_width//2,
        upsampled_size // 2 - low_res_data_width//2:upsampled_size // 2 + low_res_data_width//2] = f_data
        f_data_padded_shifted = np.fft.fftshift(f_data_padded)

        recon_upsampled = np.fft.fftshift(np.fft.ifft2(f_data_padded_shifted))
        recon_arr_upsampled[avg_ind, i, :, :] = recon_upsampled

# plotting all reconstructions

for avg_ind in range(len(avgs)):

    fig, axs = plt.subplots(8, 4, figsize=(4, 10))
    for i in range(32):

        image = np.abs(recon_arr[avg_ind, i])

        axs[i // 4, i % 4].imshow(image, cmap=plt.cm.gray)
        axs[i // 4, i % 4].axis("off")

for avg_ind in range(len(avgs)):

    fig, axs = plt.subplots(8, 4, figsize=(4, 10))
    for i in range(32):

        image = np.abs(recon_arr_upsampled[avg_ind, i])

        axs[i // 4, i % 4].imshow(image, cmap=plt.cm.gray)
        axs[i // 4, i % 4].axis("off")

# plotting a subset of reconstructions in a single array

fig, axs = plt.subplots(8, 5, figsize=(5, 8))
for k, avg in enumerate(avgs):

    for i in range(8):

        image = np.abs(recon_arr[k, i])

        axs[i, k].imshow(image, cmap=plt.cm.gray)
        axs[i, k].axis("off")

fig.tight_layout(w_pad=0.4, h_pad=0.4)

fig, axs = plt.subplots(8, 5, figsize=(5, 8))
for k, avg in enumerate(avgs):

    for i in range(8):

        image = np.abs(recon_arr_upsampled[k, i])

        axs[i, k].imshow(image, cmap=plt.cm.gray)
        axs[i, k].axis("off")

fig.tight_layout(w_pad=0.4, h_pad=0.4)

# stdev images

stdev_images = np.std(f_coeff_arr_combined, axis=1, ddof=1)
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for k, ax in enumerate(axs.flat):

    pcm = ax.imshow(stdev_images[k], cmap=plt.cm.gray, vmin=np.amin(stdev_images), vmax=np.amax(stdev_images))
    ax.axis("off")

fig.colorbar(pcm, ax=axs, shrink=0.5)

stdev_images = np.std(recon_arr, axis=1, ddof=1)
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for k, ax in enumerate(axs.flat):

    pcm = ax.imshow(stdev_images[k], cmap=plt.cm.gray, vmin=np.amin(stdev_images), vmax=np.amax(stdev_images))
    ax.axis("off")

fig.colorbar(pcm, ax=axs, shrink=0.5)

stdev_images = np.std(recon_arr_upsampled, axis=1, ddof=1)
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for k, ax in enumerate(axs.flat):

    pcm = ax.imshow(stdev_images[k], cmap=plt.cm.gray, vmin=np.amin(stdev_images), vmax=np.amax(stdev_images))
    ax.axis("off")

fig.colorbar(pcm, ax=axs, shrink=0.5)

# stdev plots
recon_arr_512 = recon_arr[0]
stdev_arr_512 = np.std(np.abs(recon_arr_512), axis=0)
np.sqrt(np.sum(np.square(stdev_arr_512)))

f_coeffs_arr_512 = f_coeff_arr_combined[0]
f_coeffs_stdev_arr_512 = np.std(np.abs(f_coeffs_arr_512), axis=0)
np.sqrt(np.sum(np.square(f_coeffs_stdev_arr_512)))

# Morozov discrepancies
diff_512 = f_coeff_arr_combined[0] - np.average(f_coeff_arr_combined[0], axis=0)
Morozov_512 = np.sqrt(np.sum(np.square(np.abs(diff_512))))/np.sqrt(32)

diff_1024 = f_coeff_arr_combined[1] - np.average(f_coeff_arr_combined[1], axis=0)
Morozov_1024 = np.sqrt(np.sum(np.square(np.abs(diff_1024))))/np.sqrt(32)

diff_2048 = f_coeff_arr_combined[2] - np.average(f_coeff_arr_combined[2], axis=0)
Morozov_2048 = np.sqrt(np.sum(np.square(np.abs(diff_2048))))/np.sqrt(32)

diff_4096 = f_coeff_arr_combined[3] - np.average(f_coeff_arr_combined[3], axis=0)
Morozov_4096 = np.sqrt(np.sum(np.square(np.abs(diff_4096))))/np.sqrt(32)

diff_8192 = f_coeff_arr_combined[4] - np.average(f_coeff_arr_combined[4], axis=0)
Morozov_8192 = np.sqrt(np.sum(np.square(np.abs(diff_8192))))/np.sqrt(32)

# comparing magnitude of Fourier recons using numpy and using RealFourierTransform
f_data = np.average(f_coeff_arr_combined[0, :, :, :], axis=0)
rec_all_averaged = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_data)))

height, width = rec_all_averaged.shape
complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                  shape=[height, width], dtype='complex')
image_space = complex_space.real_space ** 2
forward_op = RealFourierTransform(image_space)

f_data_shifted = np.fft.fftshift(f_data)
rec_odl = forward_op.inverse(forward_op.range.element([np.real(f_data_shifted), np.imag(f_data_shifted)]))

plt.figure()
plt.imshow(np.abs(rec_all_averaged), cmap=plt.cm.gray)
plt.colorbar()
plt.axis("off")

plt.figure()
plt.imshow(np.abs(rec_odl.asarray()[0]+1j*rec_odl.asarray()[1]), cmap=plt.cm.gray)
plt.colorbar()

l2_np = np.sqrt(np.sum(np.square(np.abs(rec_all_averaged))))
l2_odl = np.sqrt(np.sum(np.square(np.abs(rec_odl.asarray()[0]+1j*rec_odl.asarray()[1]))))

# SSIM computation

GT_TV_data = np.load('dTV/MRI_15032021/Results_15032021/example_TV_recon_15032021_synth_data.npy')
GT_TV_image = np.load('dTV/MRI_15032021/Results_15032021/example_TV_recon_15032021.npy')
GT_TV_image = np.abs(GT_TV_image[0] + 1j*GT_TV_image[1])
GT_TV_image_normalised = GT_TV_image/np.sqrt(np.sum(np.square(GT_TV_image)))

fully_averaged_image = np.abs(recon_fully_averaged)
fully_averaged_image_normalised = fully_averaged_image/np.sqrt(np.sum(np.square(fully_averaged_image)))

SSIM_fully_averaged_GT_arr = np.zeros((5, 32))
SSIM_proxy_GT_arr = np.zeros((5, 32))

for k, avg in enumerate(avgs):

    for j in range(32):
        recon = np.abs(recon_arr[k, j, :, :])
        recon_normalised = recon/np.sqrt(np.sum(np.square(recon)))
        SSIM_1 = recon_error(recon_normalised, fully_averaged_image_normalised)[2]
        SSIM_2 = recon_error(recon_normalised, GT_TV_image_normalised)[2]
        SSIM_fully_averaged_GT_arr[k, j] = SSIM_1
        SSIM_proxy_GT_arr[k, j] = SSIM_2

# averaging
upsampled_size=32

f_data_padded = np.zeros((upsampled_size, upsampled_size), dtype='complex')
f_data_padded[upsampled_size // 2 - low_res_data_width//2:upsampled_size // 2 + low_res_data_width//2,
upsampled_size // 2 - low_res_data_width//2:upsampled_size // 2 + low_res_data_width//2] = fully_averaged
f_data_padded_shifted = np.fft.fftshift(f_data_padded)

recon_upsampled = np.fft.fftshift(np.fft.ifft2(f_data_padded_shifted))

plt.imshow(np.abs(recon_upsampled), cmap=plt.cm.gray)
