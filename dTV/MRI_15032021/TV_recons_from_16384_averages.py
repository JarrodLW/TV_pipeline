# Created: 17/03/2021.

import numpy as np
import matplotlib.pyplot as plt
from processing import *
import json
import matplotlib.pyplot as plt
import os
import odl
import myOperators as ops
import libpysal
import esda
from Utils import *
import datetime as dt

date = '24052021'
#date = '15032021'

#TV_reg_type = 'real_imag_TV'
TV_reg_type = 'complex_TV'

dir_Li = 'dTV/MRI_15032021/Data_' + date + '/Li_data/'

if date=='15032021':
    low_res_shape = (64, 128)
    Li_range = range(3, 35)
    low_res_data_width = 32

elif date=='24052021':
    low_res_shape = (80, 128)
    Li_range = range(8, 40)
    low_res_data_width = 40

# f_coeff_list = []
#
# for i in Li_range:
#     f_coeffs = np.reshape(np.fromfile(dir_Li +str(i)+'/fid', dtype=np.int32), low_res_shape)
#     f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, low_res_data_width)
#     f_coeff_list.append(f_coeffs_unpacked)
#
# reg_params = np.logspace(2., np.log10(5*10**3), num=15)
# output_dims = [int(40)]
# Li_fourier = np.average(np.asarray(f_coeff_list), axis=0)
#
# naive_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier)))

Li_fourier = np.fft.fftshift(np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_24052021/32768_data.npy'))
naive_recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Li_fourier)))

plt.figure()
plt.imshow(np.abs(naive_recon), cmap=plt.cm.gray)

run_exp = True
#plot_results = True

#recon_arr = np.zeros((15, 2, low_res_data_width, low_res_data_width))
reg_params = np.logspace(2., np.log10(5*10**3), num=15)
#reg_params = [1000]
output_dims = [int(40), int(80), int(128)]
#output_dims = [int(40)]
#reg_params = [1000.]

if run_exp:

    regularised_recons = {}
    exp = 0

    model = VariationalRegClass('MRI', 'TV', TV_reg_type='complex_TV')

    for output_dim in output_dims:
        regularised_recons['output_size=' + str(output_dim)] = {}

        for k, reg_param in enumerate(reg_params):
            regularised_recons['output_size=' + str(output_dim)]['lambda=' + '{:.1e}'.format(reg_param)] = {}

            print("Experiment_" + str(exp))
            exp+=1

            data = np.zeros((output_dim, output_dim), dtype='complex')
            data[output_dim// 2 - low_res_data_width // 2:output_dim // 2 + low_res_data_width // 2,
            output_dim // 2 - low_res_data_width // 2:output_dim // 2 + low_res_data_width // 2]  = Li_fourier
            data = np.fft.fftshift(data)
            subsampling_matrix = np.zeros((output_dim, output_dim))
            subsampling_matrix[output_dim//2 - low_res_data_width//2 :output_dim//2 + low_res_data_width//2,
            output_dim//2 - low_res_data_width//2 :output_dim//2 + low_res_data_width//2] = 1
            subsampling_matrix = np.fft.fftshift(subsampling_matrix)

            recons = model.regularised_recons_from_subsampled_data(data, reg_param, subsampling_arr=subsampling_matrix, niter=5000)
            # recon_arr[k, 0, :, :] = np.real(recons[0])
            # recon_arr[k, 1, :, :] = np.imag(recons[0])

            regularised_recons['output_size=' + str(output_dim)]['lambda=' + '{:.1e}'.format(reg_param)][
                'recon'] = [
                np.real(np.real(recons[0])).tolist(),
                np.imag(np.imag(recons[0])).tolist()]

if TV_reg_type=='real_imag_TV':
    np.save('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_'+date+'/TV_reg_recons_32768.npy', recon_arr)

elif TV_reg_type=='complex_TV':
    # np.save('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_' + date + '/TV_complex_reg_recons_32768.npy',
    #         recon_arr)

    outputfile = '/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_' + date + '/TV_complex_reg_recons_32768.json'

    print("About to write to datafile: " + outputfile + " at " + dt.datetime.now().isoformat())
    json.dump(regularised_recons, open(outputfile, 'w'))
    print("Written outputfile at " + dt.datetime.now().isoformat())

recon_arr_16384 = np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_24052021/TV_complex_reg_recons_16384.npy')
#recon_arr_16384 = np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_24052021/TV_reg_recons_16384.npy')

height, width = (40, 40)
complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                  shape=[height, width], dtype='complex')
image_space = complex_space.real_space ** 2
forward_op = RealFourierTransform(image_space)

recs = np.zeros((15, 40, 40), dtype='complex')
residuals = np.zeros((15, 40, 40), dtype='complex')

for k in range(15):

    rec = recon_arr_16384[k, 0, :, :] + 1j*recon_arr_16384[k, 1, :, :]
    recs[k] = rec

    rec_odl = image_space.element([recon_arr_16384[k, 0, :, :], recon_arr_16384[k, 1, :, :]])
    synth_data = forward_op(rec_odl).asarray()
    synth_data_complex = np.fft.fftshift(synth_data[0] + 1j * synth_data[1])
    residual = synth_data_complex - Li_fourier

    residuals[k] = residual

vmax_rec = np.amax(np.abs(recs))
vmax_residuals = np.amax(np.abs(residuals))

# Morans I
w = libpysal.weights.lat2W(output_dim, output_dim)

fig, axs = plt.subplots(6, 5, figsize=(6, 8))

for k in range(15):

    morans_I = esda.Moran(np.abs(recs[k]), w).I
    print(morans_I)

    axs[2*(k // 5), k % 5].imshow(np.abs(recs[k]), cmap=plt.cm.gray, vmax=vmax_rec)
    axs[2*(k // 5), k % 5].axis("off")
    axs[2*(k // 5), k % 5].set_title('I: '+'{:.1e}'.format(morans_I), fontsize=10)
    axs[1+2*(k // 5), k % 5].imshow(np.abs(residuals[k]), cmap=plt.cm.gray, vmax=vmax_residuals)
    axs[1+2*(k // 5), k % 5].axis("off")


## example reconstruction, to be used as GT proxy
#example_rec = recon_arr_16384[8, 0] + 1j*recon_arr_16384[8, 1] # for 15032021 data
example_rec = recon_arr_16384[11, 0] + 1j*recon_arr_16384[11, 1] # for 24052021_data

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

np.save('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_24052021/example_TV_recon_24052021_synth_data.npy', synth_data)
np.save('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_24052021/example_TV_recon_24052021.npy', np.asarray([np.real(example_rec), np.imag(example_rec)]))

plt.figure()
plt.imshow(np.abs(synth_data_complex), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(Li_fourier), cmap=plt.cm.gray)
plt.colorbar()

l2_norm = odl.solvers.L2Norm(forward_op.range)
l2_norm(synth_data)
