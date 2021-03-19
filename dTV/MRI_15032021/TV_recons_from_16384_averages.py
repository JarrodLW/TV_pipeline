# Created: 17/03/2021.

import numpy as np
import matplotlib.pyplot as plt
from processing import *
import json
import matplotlib.pyplot as plt
import os
import odl
import myOperators as ops
from Utils import *

dir_Li = 'dTV/MRI_15032021/Data_15032021/Li_data/'
f_coeff_list = []

for i in range(3, 35):
    f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), (64, 128))
    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs)
    f_coeff_list.append(f_coeffs_unpacked)

reg_params = np.logspace(2., np.log10(5*10**3), num=15)
output_dims = [int(32)]
Li_fourier = np.average(np.asarray(f_coeff_list), axis=0)

naive_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier)))

plt.figure()
plt.imshow(np.abs(naive_recon), cmap=plt.cm.gray)

run_exp = True
#plot_results = True

recon_arr = np.zeros((15, 2, 32, 32))
#reg_params = [1000.]

if run_exp:

    regularised_recons = {}
    exp = 0

    model = VariationalRegClass('MRI', 'TV')
    for k, reg_param in enumerate(reg_params):
        for output_dim in output_dims:

            print("Experiment_" + str(exp))
            exp+=1

            data = np.zeros((output_dim, output_dim), dtype='complex')
            data[output_dim//2 - 16 :output_dim//2 + 16, output_dim//2 - 16 :output_dim//2 + 16] = Li_fourier
            data = np.fft.fftshift(data)
            subsampling_matrix = np.zeros((output_dim, output_dim))
            subsampling_matrix[output_dim//2 - 16 :output_dim//2 + 16, output_dim//2 - 16 :output_dim//2 + 16] = 1
            subsampling_matrix = np.fft.fftshift(subsampling_matrix)

            recons = model.regularised_recons_from_subsampled_data(data, reg_param, subsampling_arr=subsampling_matrix, niter=5000)
            recon_arr[k, 0, :, :] = np.real(recons[0])
            recon_arr[k, 1, :, :] = np.imag(recons[0])

np.save('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_15032021/TV_reg_recons_16384.npy', recon_arr)

recon_arr_16384 = np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_15032021/TV_reg_recons_16384.npy')

fig, axs = plt.subplots(5, 3, figsize=(6, 10))

for k in range(15):

    axs[k // 3, k % 3].imshow(np.abs(recon_arr_16384[k, :, :]), cmap=plt.cm.gray)
    axs[k // 3, k % 3].axis("off")

## example reconstruction, to be used as GT proxy
example_rec = recon_arr_16384[8, 0] + 1j*recon_arr_16384[8, 1]

plt.imshow(np.abs(example_rec), cmap=plt.cm.gray)

# synthetic K-space data
height, width = example_rec.shape
complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                  shape=[height, width], dtype='complex')
image_space = complex_space.real_space ** 2
forward_op = RealFourierTransform(image_space)

rec_odl = image_space.element([np.real(example_rec), np.imag(example_rec)])
synth_data = forward_op(rec_odl).asarray()
synth_data_complex = synth_data[0] + 1j*synth_data[1]

np.save('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_15032021/example_TV_recon_15032021.npy', synth_data)

plt.figure()
plt.imshow(np.abs(synth_data_complex), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(Li_fourier), cmap=plt.cm.gray)
plt.colorbar()
