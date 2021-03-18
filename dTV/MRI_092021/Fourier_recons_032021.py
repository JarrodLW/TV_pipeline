import numpy as np
import json
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform

avgs = ['512', '1024', '2048', '4096', '8192']
output_dims = [int(32), int(64)]

dir_Li = 'dTV/MRI_092021/Data_09032021/Li_data/'
dir_H = 'dTV/MRI_092021/Data_09032021/H_data/'

# I've changed the unpacking function from previous experiments - data seems to be in different form
def unpacking_fourier_coeffs(arr):

    fourier_real_im = arr[:, :64]
    fourier_real_im = fourier_real_im[::2, :]

    fourier_real = fourier_real_im[:, 1::2]
    fourier_im = fourier_real_im[:, ::2]
    fourier = fourier_real + fourier_im * 1j

    return fourier

## 1H reconstructions

# low-res

f_coeffs = np.reshape(np.fromfile(dir_H +str(11)+'/fid', dtype=np.int32), (64, 128))
f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
recon_low_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs_unpacked)))

plt.figure()
plt.imshow(np.abs(recon_low_res), cmap=plt.cm.gray, vmax=200)
plt.colorbar()

# high-res

#f_coeffs = np.reshape(np.fromfile(dir_H +str(10)+'/fid', dtype=np.int64), (128, 128))
f_coeffs = np.reshape(np.fromfile(dir_H +str(10)+'/fid', dtype=np.int32), (128, 256))
f_coeffs = f_coeffs[:, 1::2] + 1j*f_coeffs[:, ::2]
#f_coeffs = np.reshape(np.fromfile(dir_H +str(10)+'/fid', dtype=np.int16), (256, 256))
recon_high_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs)))

plt.figure()
plt.imshow(np.abs(recon_high_res), cmap=plt.cm.gray, vmax=20)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.abs(recon_high_res)[:, 30:], cmap=plt.cm.gray)
plt.colorbar()


#f_coeffs = np.reshape(np.fromfile(dir_H +'/fid', dtype=np.int32), (128, 256))
f_coeffs = np.reshape(np.fromfile(dir_H +'/fid', dtype=np.int64), (128, 128))
#f_coeffs_real = f_coeffs[:, :128]
#f_coeffs_imag = f_coeffs[:, 128:]
#f_coeffs = f_coeffs_real + 1j*f_coeffs_imag
#f_coeffs = np.reshape(np.fromfile(dir_H +'/fid', dtype=np.int), (128, 128))
recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs)))

plt.imshow(np.abs(recon), cmap=plt.cm.gray)

# 32x32 test

f_coeffs = np.reshape(np.fromfile('dTV/MRI_092021/fid', dtype=np.int32), (64, 128))
f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)

plt.imshow(np.abs(f_coeffs_unpacked), cmap=plt.cm.gray)
recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs_unpacked)))

plt.imshow(np.abs(recon), cmap=plt.cm.gray)

## 7Li reconstructions

f_coeff_list = []

for i in range(3, 35):
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
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
plt.imshow(np.abs(recon_fully_averaged), cmap=plt.cm.gray, vmax=100)
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


fig, axs = plt.subplots(8, 4)
for i in range(32):

    axs[i//4, i % 4].imshow(np.abs(np.real(recon_arr[0, i])), cmap=plt.cm.gray)
    axs[i // 4, i % 4].axis("off")

plt.show()

fig, axs = plt.subplots(8,4)
for i in range(32):

    axs[i//4, i % 4].imshow(np.abs(np.imag(recon_arr[0, i])), cmap=plt.cm.gray)
    axs[i // 4, i % 4].axis("off")

plt.show()


recon_from_real=np.fft.fftshift(np.fft.ifft2(np.real(fully_averaged_shifted)))
recon_from_imag=np.fft.fftshift(np.fft.ifft2(np.imag(fully_averaged_shifted)))

plt.imshow(np.abs(recon_from_real))
plt.imshow(np.abs(recon_from_imag))
