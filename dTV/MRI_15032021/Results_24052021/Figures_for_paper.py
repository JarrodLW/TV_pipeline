import numpy as np
import matplotlib.pyplot as plt
from Utils import *
import scipy as sp

dir_H = 'dTV/MRI_15032021/Data_24052021/H_data/'
dir_Li = 'dTV/MRI_15032021/Data_24052021/Li_data/'

n = 2048
f_coeff_list = []

for i in range(8, 40):
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), (80, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, 40)
    f_coeff_list.append(f_coeffs_unpacked)

f_coeff_arr = np.asarray(f_coeff_list)
if n !=512:
    f_coeff_list_grouped = []
    num = n//512
    for i in range(num):
        data_arr = np.roll(f_coeff_arr, i, axis=0)
        for ele in range(len(f_coeff_list)//num):
            f_coeff_list_grouped.append(np.sum(data_arr[num*ele:num*(ele+1)], axis=0)/num)

    f_coeff_list = f_coeff_list_grouped

# 512 average Fourier
recon_512 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeff_arr[0])))

# 2048 average Fourier
recon_2048 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeff_list[0])))

# fully averaged (32768) recon
data = np.load('dTV/MRI_15032021/Results_24052021/32768_data.npy')
recon_32768 = np.fft.fftshift(np.fft.ifft2(data))

# Filtering of 32768 recon
# recon_32768_filtered_real = sp.ndimage.filters.gaussian_filter(np.real(recon_32768), 0.8)
# recon_32768_filtered_imag = sp.ndimage.filters.gaussian_filter(np.imag(recon_32768), 0.8)
#
# plt.figure()
# plt.imshow(np.abs(recon_32768), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(np.abs(recon_32768_filtered_real+1j*recon_32768_filtered_imag), cmap=plt.cm.gray)

# High res H
f_coeffs = np.reshape(np.fromfile(dir_H +str(32)+'/fid', dtype=np.int32), (128, 256))
f_coeffs = f_coeffs[:, 1::2] + 1j*f_coeffs[:, ::2]
recon_high_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs)))

#max = np.amax(np.asarray([np.abs(recon_512), np.abs(recon_2048), np.abs(recon_32768)]))

f, axarr = plt.subplots(1, 4, figsize=(12, 4))
im0 = axarr[0].imshow(np.abs(recon_high_res), cmap=plt.cm.gray)
axarr[0].axis("off")
im1 = axarr[1].imshow(np.abs(recon_512), cmap=plt.cm.gray)
axarr[1].axis("off")
im2 = axarr[2].imshow(np.abs(recon_2048), cmap=plt.cm.gray)
axarr[2].axis("off")
im3 = axarr[3].imshow(np.abs(recon_32768), cmap=plt.cm.gray)
axarr[3].axis("off")
f.colorbar(im0, ax=axarr[0], shrink=0.5)
f.colorbar(im1, ax=axarr[1], shrink=0.5)
f.colorbar(im2, ax=axarr[2], shrink=0.5)
f.colorbar(im3, ax=axarr[3], shrink=0.5)

