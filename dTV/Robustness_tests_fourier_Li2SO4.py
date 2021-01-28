import numpy as np
import json
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform

avgs = ['512', '1024', '2048', '4096', '8192']
output_dims = [int(32), int(64)]

dir = 'dTV/7Li_1H_MRI_Data_31112020/'

def unpacking_fourier_coeffs(arr):

    fourier_real_im = arr[:, 1:65]
    fourier_real_im = fourier_real_im[::2, :]

    fourier_real = fourier_real_im[:, 1::2]
    fourier_im = fourier_real_im[:, ::2]
    fourier = fourier_real + fourier_im * 1j

    return fourier

f_coeff_list = []

for i in range(2, 34):
    f_coeffs = np.reshape(np.fromfile(dir + 'Li2SO4/'+str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)

# putting all the data into a single array, first index corresponding to numebr of averages, second the sample number,
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

#recon_dict = {}
recon_arr = np.zeros((len(avgs), 32, 32, 32), dtype='complex')
recon_arr_upsampled = np.zeros((len(avgs), 32, 64, 64), dtype='complex')
recon_arr_cropped_fourier = np.zeros((len(avgs), 32, 28, 28), dtype='complex')

for avg_ind in range(len(avgs)):
    #recon_dict['avgs='+avgs[avg_ind]] = {}
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

        f_data_cropped = f_coeff_arr_combined[avg_ind, i, 3:-1, 3:-1]
        f_data_cropped_shifted = np.fft.fftshift(f_data_cropped)
        recon = np.fft.fftshift(np.fft.ifft2(f_data_cropped_shifted))

        recon_arr_cropped_fourier[avg_ind, i, :, :] = recon

        #recon_dict['avgs=' + avgs[avg_ind]]['measurement=' + str(i)] =

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

stdev_images = np.std(recon_arr_cropped_fourier, axis=1, ddof=1)
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for k, ax in enumerate(axs.flat):

    pcm = ax.imshow(stdev_images[k], cmap=plt.cm.gray, vmin=np.amin(stdev_images), vmax=np.amax(stdev_images))
    ax.axis("off")

fig.colorbar(pcm, ax=axs, shrink=0.5)

# recon from averaged data

rec_all_averaged = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.average(f_coeff_arr_combined[0, :, :, :], axis=0))))

plt.imshow(np.abs(rec_all_averaged), cmap=plt.cm.gray)
plt.colorbar()
plt.axis("off")

# stdev plots
recon_arr_512 = recon_arr[0]
stdev_arr_512 = np.std(np.abs(recon_arr_512), axis=0)
np.sqrt(np.sum(np.square(stdev_arr_512)))

f_coeffs_arr_512 = f_coeff_arr_combined[0]
f_coeffs_stdev_arr_512 = np.std(np.abs(f_coeffs_arr_512), axis=0)
np.sqrt(np.sum(np.square(f_coeffs_stdev_arr_512)))



# data fidelity for TV-regularised recon from 16384 averages


height=32
width=32
complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.], shape=[height, width], dtype='complex')
image_space = complex_space.real_space ** 2
forward_op = RealFourierTransform(image_space)

f_data_fully_averaged = np.average(f_coeff_arr_combined[0, :, :, :], axis=0)
rec = np.load('dTV/Results_MRI_dTV/example_TV_recon_Li2SO4_16384_avgs_reg_param_1000.npy')
rec_complex = rec[0] + 1j*rec[1]
rec_rotated = rec_complex.T[:, ::-1]

synthetic_data = forward_op(image_space.element([np.real(rec_rotated), np.imag(rec_rotated)]))
synth_data_arr = synthetic_data.asarray()[0] + 1j*synthetic_data.asarray()[1]
f_data_fully_averaged_shifted = np.fft.ifftshift(f_data_fully_averaged)

plt.figure()
plt.imshow(np.abs(f_data_fully_averaged_shifted),cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(synth_data_arr), cmap=plt.cm.gray)
plt.colorbar()

l2_norm = odl.solvers.L2Norm(forward_op.range)

l2_norm(forward_op.range.element([np.real(f_data_fully_averaged_shifted), np.imag(f_data_fully_averaged_shifted)]) - synthetic_data)
np.sqrt(np.sum(np.square(np.abs(f_data_fully_averaged_shifted - synth_data_arr))))



plt.figure()
plt.imshow(np.roll(np.abs(f_data_fully_averaged), -2, axis=1), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.fft.fftshift(np.abs(synth_data_arr)), cmap=plt.cm.gray)
plt.colorbar()

rec_from_rolled = np.fft.fftshift(np.fft.ifft2(f_data_fully_averaged_shifted_rolled))

np.sqrt(np.sum(np.square(np.real(rec_from_rolled))))
np.sqrt(np.sum(np.square(np.imag(rec_from_rolled))))


