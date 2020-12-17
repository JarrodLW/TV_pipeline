import numpy as np
import json
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform

plot_TV_results = False
plot_dTV_results = True
discrepancy_plots = False
dTV_discrepancy_plots = False

avgs = ['512', '1024', '2048', '4096', '8192']
#avgs = ['512']
reg_params = np.logspace(np.log10(2e3), np.log10(1e5), num=20)
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

for avg_ind in range(len(avgs)):
    #recon_dict['avgs='+avgs[avg_ind]] = {}
    for i in range(32):

        f_data = f_coeff_arr_combined[avg_ind, i, :, :]
        f_data_shifted = np.fft.fftshift(f_data)
        recon = np.fft.fftshift(np.fft.ifft2(f_data_shifted))

        recon_arr[avg_ind, i, :, :] = recon

        #recon_dict['avgs=' + avgs[avg_ind]]['measurement=' + str(i)] =

stdev_images = np.std(np.abs(recon_arr), axis=1)

plt.figure()
plt.hist(np.ndarray.flatten(stdev_images[4, :, :]), bins=40)

np.sqrt(np.sum(np.square(stdev_images), axis=(1,2)))

np.std(recon_arr, axis=1)

rec_example = recon_arr[4, 20, :, :]

plt.imshow(np.abs(rec_example), cmap=plt.cm.gray)

rec_all_averaged = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(np.average(f_coeff_arr_combined[0, :, :, :], axis=0))))
