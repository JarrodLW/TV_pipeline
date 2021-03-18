import numpy as np
import json
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform
from Utils import *

avgs = ['512', '1024', '2048', '4096', '8192']
output_dims = [int(32), int(64)]

dir_Li = 'dTV/MRI_15032021/Data_15032021/Li_data/'
dir_H = 'dTV/MRI_15032021/Data_15032021/H_data/'

## 1H reconstructions

# low-res
f_coeffs = np.reshape(np.fromfile(dir_H +str(5)+'/fid', dtype=np.int32), (64, 128))
f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs)
recon_low_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs_unpacked)))

plt.figure()
plt.imshow(np.abs(recon_low_res), cmap=plt.cm.gray)
plt.colorbar()

# high-res
f_coeffs = np.reshape(np.fromfile(dir_H +str(6)+'/fid', dtype=np.int32), (128, 256))
f_coeffs = f_coeffs[:, 1::2] + 1j*f_coeffs[:, ::2]
recon_high_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs)))

plt.figure()
plt.imshow(np.abs(recon_high_res), cmap=plt.cm.gray)
plt.colorbar()

## 7Li reconstructions
f_coeff_list = []

for i in range(3, 35):
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)

# putting all the data into a single array, first index corresponding to number of averages, second the sample number,
# third and fourth indexing the data itself
f_coeff_arr = np.asarray(f_coeff_list)
f_coeff_arr_combined = np.zeros((len(avgs), 32, 32, 32), dtype='complex')

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

plt.figure()
plt.imshow(np.abs(recon_fully_averaged), cmap=plt.cm.gray)
plt.colorbar()

recon_arr = np.zeros((len(avgs), 32, 32, 32), dtype='complex')
recon_arr_upsampled = np.zeros((len(avgs), 32, 64, 64), dtype='complex')

for avg_ind in range(len(avgs)):
    for i in range(32):

        f_data = f_coeff_arr_combined[avg_ind, i, :, :]
        f_data_shifted = np.fft.fftshift(f_data)
        recon = np.fft.fftshift(np.fft.ifft2(f_data_shifted))

        recon_arr[avg_ind, i, :, :] = recon

        f_data_padded = np.zeros((64, 64), dtype='complex')
        f_data_padded[64 // 2 - 16:64 // 2 + 16, 64// 2 - 16:64 // 2 + 16] = f_data
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

