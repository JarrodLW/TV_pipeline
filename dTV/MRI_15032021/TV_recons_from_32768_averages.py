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

run_expt = True
plot = False

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

#plot_results = True

#recon_arr = np.zeros((15, 2, low_res_data_width, low_res_data_width))
reg_params = np.logspace(2., np.log10(5*10**3), num=15)
reg_params = [0.]
#output_dims = [int(40), int(80), int(120)]
output_dims = [int(120)]


filename = '/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_' + date + '/TV_complex_reg_recons_32768.json'

if os.path.isfile(filename):

    print("About to read previous datafile: "+filename+" at "+dt.datetime.now().isoformat())
    with open(filename, 'r') as f:
        regularised_recons = json.load(f)
    print("Loaded previous datafile at "+dt.datetime.now().isoformat())

    f.close()

else:
    print("Could not find: "+filename)
    regularised_recons = {}

if run_expt:

    #regularised_recons = {}
    exp = 0

    model = VariationalRegClass('MRI', 'TV', TV_reg_type='complex_TV')

    for output_dim in output_dims:

        if 'output_size=' + str(output_dim) not in regularised_recons.keys():
            regularised_recons['output_size=' + str(output_dim)] = {}

        height, width = output_dim, output_dim
        complex_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1],
                                          shape=[height, width], dtype='complex', interp='linear')
        image_space = complex_space.real_space ** 2


        # defining the forward op - I should do the subsampling in a more efficient way
        fourier_transf = ops.RealFourierTransform(image_space)
        data_height, data_width = Li_fourier.shape

        subsampling_arr = np.zeros((height, width))
        subsampling_arr[height // 2 - data_height // 2: height // 2 + data_height // 2,
        width // 2 - data_width // 2: width // 2 + data_width // 2] = 1
        subsampling_arr = np.fft.fftshift(subsampling_arr)
        subsampling_arr_doubled = np.array([subsampling_arr, subsampling_arr])

        forward_op = fourier_transf.range.element(subsampling_arr_doubled) * fourier_transf

        for k, reg_param in enumerate(reg_params):

            if 'lambda=' + '{:.1e}'.format(reg_param) not in regularised_recons['output_size=' + str(output_dim)].keys():

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

                # residuals
                diff = forward_op(forward_op.domain.element([np.real(recons[0]), np.imag(recons[0])])) - \
                forward_op.range.element([np.real(data), np.imag(data)])
                diff = diff.asarray()[0] + 1j*diff.asarray()[1]
                diff_shifted = np.fft.fftshift(diff)
                diff_shifted_window = diff_shifted[output_dim// 2 - low_res_data_width // 2:output_dim // 2 + low_res_data_width // 2,
                output_dim // 2 - low_res_data_width // 2:output_dim // 2 + low_res_data_width // 2]

                regularised_recons['output_size=' + str(output_dim)]['lambda=' + '{:.1e}'.format(reg_param)][
                    'recon'] = [
                    np.real(recons[0]).tolist(),
                    np.imag(recons[0]).tolist()]
                regularised_recons['output_size=' + str(output_dim)]['lambda=' + '{:.1e}'.format(reg_param)][
                    'fourier_diff'] = [
                    np.real(diff_shifted_window).tolist(),
                    np.imag(diff_shifted_window).tolist()]

    if TV_reg_type=='real_imag_TV':
        np.save('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_'+date+'/TV_reg_recons_32768.npy', recon_arr)

    elif TV_reg_type=='complex_TV':
        # np.save('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_' + date + '/TV_complex_reg_recons_32768.npy',
        #         recon_arr)

        outputfile = '/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_' + date + '/TV_complex_reg_recons_32768.json'

    print("About to write to datafile: " + outputfile + " at " + dt.datetime.now().isoformat())
    json.dump(regularised_recons, open(outputfile, 'w'))
    print("Written outputfile at " + dt.datetime.now().isoformat())

#recon_arr = np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_24052021/TV_complex_reg_recons_16384.npy')
#recon_arr_16384 = np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_24052021/TV_reg_recons_16384.npy')
#recon_arr = np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_24052021/TV_complex_reg_recons_32768.npy')

if plot:

    outputfile = '/Users/jlw31/PycharmProjects/TV_pipeline/dTV/MRI_15032021/Results_' + date + '/TV_complex_reg_recons_32768.json'

    with open(outputfile, 'r') as f:
        d = json.load(f)
    print("Loaded previous datafile at " + dt.datetime.now().isoformat())

    d2 = d['output_size=120']

    recon_images = np.zeros((len(reg_params), 120, 120))
    fourier_diff_images = np.zeros((len(reg_params), 40, 40))

    for i, reg_param in enumerate(reg_params):

        d3 = d2['lambda=' + '{:.1e}'.format(reg_param)]
        recon = np.asarray(d3['recon'])
        f_diff = np.asarray(d3['fourier_diff'])

        recon_images[i, :, :] = np.abs(recon[0] + 1j*recon[1])
        fourier_diff_images[i, :, :] = np.abs(f_diff[0] + 1j*f_diff[1])


    f, axarr = plt.subplots(6, 5, figsize=(10, 12))

    for i, reg_param in enumerate(reg_params):

        axarr[2*(i//5), i%5].imshow(recon_images[i], vmax=np.amax(recon_images), interpolation='none',
                             cmap=plt.cm.gray)
        axarr[2 * (i//5), i%5].axis("off")

        pcm = axarr[2*(i//5)+1, i%5].imshow(fourier_diff_images[i], vmax=np.amax(fourier_diff_images), interpolation='none',
                               cmap=plt.cm.gray)
        axarr[2*(i//5)+1, i%5].axis("off")

    #f.colorbar(pcm, ax=[axarr[1, -1]])
    plt.tight_layout(w_pad=0.3, h_pad=0.3)





#

height, width = (40, 40)
complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                  shape=[height, width], dtype='complex')
image_space = complex_space.real_space ** 2
forward_op = RealFourierTransform(image_space)

recs = np.zeros((15, 40, 40), dtype='complex')
residuals = np.zeros((15, 40, 40), dtype='complex')

for k in range(15):

    rec = recon_arr[k, 0, :, :] + 1j*recon_arr[k, 1, :, :]
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
