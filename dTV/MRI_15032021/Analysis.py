import numpy as np
import json
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform
from skimage.measure import block_reduce
import libpysal
import esda
from Utils import *
from skimage.transform import resize

plot_TV_results = False
best_TV_recons = False
plot_dTV_results = True
plot_Moran = False
plot_TV_results_full_avgs = False
plot_subset_TV_results = False
discrepancy_plots = False
dTV_discrepancy_plots = False
affine_param_plots = False
best_recons = False
hyperparam_sweep_results = False
plot_hyperparam_sweep_results = False

date = '24052021'
#date = '15032021'

TV_reg_type = 'complex_TV'
#TV_reg_type = 'real_imag_TV'

if date=='15032021':
    low_res_shape = (64, 128)
    Li_range = range(3, 35)
    low_res_data_width = 32
    output_dims = [int(32), int(64)]

elif date=='24052021':
    low_res_shape = (80, 128)
    Li_range = range(8, 40)
    low_res_data_width = 40
    output_dims = [int(40), int(80), int(128)]

avgs = ['512', '1024', '2048', '4096', '8192']
reg_params = np.concatenate((np.asarray([0.001, 1., 10**0.5, 10., 10**1.5, 10**2]), np.logspace(3., 4.5, num=20)))
#output_dims = [int(32), int(64)]
#output_dims = [int(64)]
#output_dims = [int(32)]
#output_dims = [int(128)]

dir = 'dTV/MRI_15032021/'
extensions = ['']
save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/Experiments/MRI_birmingham/Results_'+date+'/'

if plot_TV_results:

    if date == '15032021':
        GT_TV_data = np.load(dir + 'Results_15032021/example_TV_recon_15032021_synth_data.npy')
        GT_TV_image = np.load(dir + 'Results_15032021/example_TV_recon_15032021.npy')
        GT_TV_image = np.abs(GT_TV_image[0] + 1j*GT_TV_image[1])
        low_res_H_image = np.load('dTV/MRI_15032021/Results_15032021/pre_registered_H_image_low_res.npy')
        low_res_H_image_normalised = low_res_H_image/np.sqrt(np.sum(np.square(low_res_H_image)))

    elif date == '24052021':
        GT_TV_data = np.load(dir + 'Results_24052021/example_TV_recon_24052021_synth_data.npy')
        GT_TV_image = np.load(dir + 'Results_24052021/example_TV_recon_24052021.npy')
        GT_TV_image = np.abs(GT_TV_image[0] + 1j * GT_TV_image[1])
        image_H_high_res = np.load('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res.npy')
        low_res_H_image = resize(image_H_high_res, (40, 40))
        low_res_H_image_normalised = low_res_H_image / np.sqrt(np.sum(np.square(low_res_H_image)))

    for k, ext in enumerate(extensions):

        GT_norms_dict = {}
        GT_TV_norms_dict = {}
        GT_SSIM_dict = {}
        GT_TV_SSIM_dict = {}
        H_SSIM_dict = {}
        norms_dict = {}
        stdevs = {}
        morans_I_dict = {}

        for j, avg in enumerate(avgs):

            GT_norms_dict['avgs=' + avg] = {}
            GT_TV_norms_dict['avgs=' + avg] = {}
            GT_SSIM_dict['avgs=' + avg] = {}
            GT_TV_SSIM_dict['avgs=' + avg] = {}
            H_SSIM_dict['avgs=' + avg] = {}
            norms_dict['avgs='+ avg] = {}
            stdevs['avgs=' + avg] = {}
            morans_I_dict['avgs=' + avg] = {}

            f_coeff_list = []

            if date == '15032021':
                exten = 'TV_results'

            elif date == '24052021':
                if TV_reg_type == 'real_imag_TV':
                    exten = 'TV_results'

                elif TV_reg_type == 'complex_TV':
                    exten = 'TV_complex_results'

            with open(save_dir + exten + '/TV_7Li_'+date+'_'+str(avg)+'.json') as f:
                d = json.load(f)

            print("read avgs" + avg)

            if k==0:
                # getting the data
                for i in Li_range:
                    f_coeffs = np.reshape(np.fromfile(dir + 'Data_'+date+'/Li_data/' + str(i) + '/fid', dtype=np.int32), low_res_shape)
                    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, low_res_data_width)
                    f_coeff_list.append(f_coeffs_unpacked)

            f_coeff_arr = np.asarray(f_coeff_list)
            fully_averaged_coeffs = np.average(f_coeff_arr, axis=0)
            fully_averaged_shifted = np.fft.fftshift(fully_averaged_coeffs)
            recon_fully_averaged = np.fft.fftshift(np.fft.ifft2(fully_averaged_shifted))
            GT_image = np.abs(recon_fully_averaged)

            f_coeff_list_grouped = []
            num = int(2 ** j)
            for i in range(num):
                data_arr = np.roll(f_coeff_arr, i, axis=0)
                for ele in range(len(f_coeff_list) // num):
                    f_coeff_list_grouped.append(np.sum(data_arr[num * ele:num * (ele + 1)], axis=0) / num)

            coeffs = f_coeff_list_grouped

            try:
                # all the recons for each num of avgs for each reg parameter, in separate plots
                for output_dim in output_dims:

                    GT_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
                    GT_TV_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
                    GT_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
                    GT_TV_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
                    H_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
                    norms_dict['avgs='+ avg]['output_dim=' + str(output_dim)] = {}
                    stdevs['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
                    morans_I_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}

                    complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                                      shape=[output_dim, output_dim], dtype='complex')
                    image_space = complex_space.real_space ** 2
                    forward_op = RealFourierTransform(image_space)

                    l2_norm = odl.solvers.L2Norm(forward_op.range)
                    w = libpysal.weights.lat2W(output_dim, output_dim) # this will change to 32x32 when I correct the Fourier diff dimensions

                    for reg_param in reg_params:

                        GT_diff_norms = []
                        GT_TV_diff_norms = []
                        GT_SSIM_vals = []
                        GT_TV_SSIM_vals = []
                        H_SSIM_vals = []
                        diff_norms = []
                        recons = []
                        morans_I_vals = []

                        fig, axs = plt.subplots(16, 4, figsize=(4, 10))
                        for i in range(32):

                            recon = np.asarray(d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]
                                               ['output_size=' + str(output_dim)]).astype('float64')

                            image = np.abs(recon[0] + 1j*recon[1])

                            if date=='15032021':
                                if output_dim is not int(32):
                                    image = resize(image, (32, 32))

                            elif date == '24052021':
                                if output_dim is not int(40):
                                    image = resize(image, (40, 40))

                            data = np.zeros((output_dim, output_dim), dtype='complex')
                            data[output_dim // 2 - low_res_data_width//2:output_dim // 2 + low_res_data_width//2,
                            output_dim // 2 - low_res_data_width//2:output_dim // 2 + low_res_data_width//2] = coeffs[i]
                            data = np.fft.fftshift(data)
                            #data = np.fft.fftshift(coeffs[i])

                            fully_averaged_data = np.zeros((output_dim, output_dim), dtype='complex')
                            fully_averaged_data[output_dim // 2 - low_res_data_width//2:output_dim // 2 + low_res_data_width//2,
                            output_dim // 2 - low_res_data_width//2:output_dim // 2 + low_res_data_width//2] = \
                            fully_averaged_coeffs
                            fully_averaged_data = np.fft.fftshift(fully_averaged_data)
                            #fully_averaged_data = np.fft.fftshift(fully_averaged_coeffs)

                            GT_proxy = np.zeros((output_dim, output_dim), dtype='complex')
                            GT_proxy[output_dim // 2 - low_res_data_width//2: output_dim // 2 + low_res_data_width//2,
                            output_dim // 2 - low_res_data_width//2: output_dim // 2 + low_res_data_width//2]=\
                                np.fft.ifftshift(GT_TV_data[0]+1j*GT_TV_data[1])
                            GT_proxy = np.fft.fftshift(GT_proxy)

                            subsampling_matrix = np.zeros((output_dim, output_dim))
                            subsampling_matrix[output_dim // 2 - low_res_data_width//2:output_dim // 2 + low_res_data_width//2,
                            output_dim // 2 - low_res_data_width//2:output_dim // 2 + low_res_data_width//2] = 1
                            subsampling_matrix = np.fft.fftshift(subsampling_matrix)

                            synth_data = np.asarray([subsampling_matrix, subsampling_matrix])*forward_op(forward_op.domain.element([recon[0], recon[1]]))
                            diff = synth_data - forward_op.range.element([np.real(data), np.imag(data)])
                            diff_norm = l2_norm(diff)
                            diff_norms.append(diff_norm)

                            morans_I = esda.Moran(np.fft.fftshift(np.abs(diff.asarray()[0] + 1j*diff.asarray()[1])), w)
                            morans_I_vals.append(morans_I.I)

                            GT_diff = synth_data - forward_op.range.element([np.real(fully_averaged_data), np.imag(fully_averaged_data)])
                            GT_diff_norm = l2_norm(GT_diff)
                            GT_diff_norms.append(GT_diff_norm)

                            # comparison with the cleaned up data from TV reconstruction

                            GT_TV_diff = synth_data - forward_op.range.element([np.real(GT_proxy), np.imag(GT_proxy)])
                            GT_TV_diff_norm = l2_norm(GT_TV_diff)
                            GT_TV_diff_norms.append(GT_TV_diff_norm)

                            # SSIM vals
                            # GT_SSIM = recon_error(image, GT_image)[2]
                            # GT_TV_SSIM = recon_error(image, GT_TV_image)[2]
                            # GT_SSIM_vals.append(GT_SSIM)
                            # GT_TV_SSIM_vals.append(GT_TV_SSIM)
                            image_normalised = image / np.sqrt(np.sum(np.square(image)))
                            GT_image_normalised = GT_image / np.sqrt(np.sum(np.square(GT_image)))
                            GT_TV_image_normalised = GT_TV_image / np.sqrt(np.sum(np.square(GT_TV_image)))
                            GT_SSIM = recon_error(image_normalised, GT_image_normalised)[2]
                            GT_TV_SSIM = recon_error(image_normalised, GT_TV_image_normalised)[2]
                            H_SSIM = recon_error(image_normalised, low_res_H_image_normalised)[2]
                            GT_SSIM_vals.append(GT_SSIM)
                            GT_TV_SSIM_vals.append(GT_TV_SSIM)
                            H_SSIM_vals.append(H_SSIM)

                            # example data, just to check consistency of fftshifts etc....
                            if k==0 and j==2 and output_dim==int(32) and reg_param==reg_params[15] and i==5:

                                data_array = np.asarray([synth_data.asarray(), [np.real(fully_averaged_data), np.imag(fully_averaged_data)],
                                                         [np.real(GT_proxy), np.imag(GT_proxy)]])

                                np.save('dTV/Results_MRI_dTV/debugging_fft_shifts.npy', data_array)

                            axs[2*(i // 4), i % 4].imshow(image, cmap=plt.cm.gray, interpolation='none')
                            axs[2*(i // 4), i % 4].axis("off")

                            axs[1+2 * (i // 4), i % 4].imshow(np.fft.fftshift(np.abs(diff.asarray()[0] + 1j*diff.asarray()[1])), cmap=plt.cm.gray, interpolation='none')
                            axs[1+2 * (i // 4), i % 4].axis("off")

                            recons.append(image)

                        fig.tight_layout(w_pad=0.4, h_pad=0.4)
                        plt.savefig(save_dir + exten + "/" + avg +"_avgs/" + str(output_dim) + "/TV_"+date+"_" + avg + "_avgs_32_to_" + str(
                            output_dim) + "reg_param_" + '{:.1e}'.format(reg_param) + ext + "_new.pdf")
                        plt.close()

                        norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = diff_norms

                        GT_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = GT_diff_norms

                        GT_TV_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = GT_TV_diff_norms

                        GT_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = GT_SSIM_vals

                        GT_TV_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = GT_TV_SSIM_vals

                        H_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = H_SSIM_vals

                        stdev = np.sqrt(np.sum(np.square(np.std(recons, axis=0))))
                        stdevs['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = stdev

                        morans_I_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = morans_I_vals

                        np.save(save_dir + exten + "/" + avg +"_avgs/" + str(output_dim) + "/TV_"+date+"_" + avg + "_avgs_32_to_" + str(
                            output_dim) + "reg_param_" + '{:.1e}'.format(reg_param)+"stdev_arr.npy", np.std(recons, axis=0))

                        plt.figure()
                        plt.imshow(np.std(recons, axis=0), cmap=plt.cm.gray)
                        plt.colorbar()
                        plt.savefig(save_dir + exten + "/" + avg +"_avgs/" + str(output_dim) + "/TV_"+date+"_"+ avg + "_avgs_32_to_" + str(
                            output_dim) + "reg_param_" + '{:.1e}'.format(reg_param)+"stdev_plot.pdf")
                        plt.close()

                        plt.figure()
                        plt.hist(np.ndarray.flatten(np.std(recons, axis=0)), bins=40)
                        plt.savefig(save_dir + exten + "/" + avg +"_avgs/" + str(output_dim) + "/TV_"+date+"_" + avg + "_avgs_32_to_" + str(
                            output_dim) + "reg_param_" + '{:.1e}'.format(reg_param) + "stdev_hist.pdf")
                        plt.close()

            except KeyError:
                print("failed to grab key (presumably)")
                continue

        json.dump(norms_dict,
                  open(save_dir + exten + "/" + 'TV_fidelities.json', 'w'))

        json.dump(GT_norms_dict,
                  open(save_dir + exten + "/" + 'TV_GT_fidelities.json', 'w'))

        json.dump(GT_TV_norms_dict,
                  open(save_dir + exten + "/" + 'TV_GT_proxy_fidelities.json', 'w'))

        json.dump(GT_SSIM_dict,
                  open(save_dir + exten + "/" + 'TV_GT_SSIM_vals.json', 'w'))

        json.dump(GT_TV_SSIM_dict,
                  open(save_dir + exten + "/" + 'TV_GT_proxy_SSIM_vals.json', 'w'))

        json.dump(H_SSIM_dict,
                  open(save_dir + exten + "/" + 'TV_H_SSIM_vals.json', 'w'))

        json.dump(stdevs,
                  open(save_dir + exten + "/" + 'TV_aggregated_pixel_stds.json', 'w'))

        json.dump(morans_I_dict,
                  open(save_dir + exten + "/" + 'TV_morans_I.json', 'w'))

if plot_TV_results_full_avgs:

    # plotting TV recons for full number (16384) of averages, small regularisation params
    with open('dTV/Results_MRI_dTV/Robustness_31112020_TV_16384_lower_reg_params.json') as f:
        d = json.load(f)

    d2 = d['measurement=0']
    reg_params = np.logspace(np.log10(1.), np.log10(2*10**3), num=10)

    for output_dim in output_dims:
        for i, reg_param in enumerate(reg_params):

            recon = np.asarray(d2['reg_param='+'{:.1e}'.format(reg_param)]['output_size='+str(output_dim)]).astype('float64')

            plt.figure()
            plt.imshow(np.abs(recon[0] + 1j*recon[1]), cmap=plt.cm.gray)

# grabbing example recons

if best_TV_recons:

    with open(save_dir + '/New/results/TV_results/Robustness_31112020_TV_2048_new.json') as f:
        d = json.load(f)

    reg_param = 2.1*10**3

    recon = np.asarray(d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]
                       ['output_size=' + str(32)]).astype('float64')


# plotting data discrepancies

if discrepancy_plots:

    if TV_reg_type == 'complex_TV':
        ext = 'TV_complex'
    elif TV_reg_type == 'real_imag_TV':
        ext = 'TV'

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/'+ext+'/TV_fidelities.json') as f:
        d = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/'+ext+'/TV_GT_fidelities.json') as f:
        D = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/'+ext+'/TV_GT_proxy_fidelities.json') as f:
        DD = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/'+ext+'/TV_GT_SSIM_vals.json') as f:
        D_SSIM = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/'+ext+'/TV_GT_proxy_SSIM_vals.json') as f:
        DD_SSIM = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/'+ext+'/TV_H_SSIM_vals.json') as f:
        DDD_SSIM = json.load(f)

    f.close()

    if date=='15032021':
        Morozov_thresholds = [101165, 70547, 48230, 31739, 18380]
    elif date=='24052021':
        Morozov_thresholds = [126000, 88000, 60300, 39500, 22800]

    f, axarr = plt.subplots(3, 2, figsize=(10, 10))

    for k, avg in enumerate(avgs):
        print("unpacking average " + str(avg))

        GT_discrep_arr = np.zeros((len(reg_params), 32))
        GT_TV_discrep_arr = np.zeros((len(reg_params), 32))
        discrep_arr = np.zeros((len(reg_params), 32))
        GT_SSIM_arr = np.zeros((len(reg_params), 32))
        GT_proxy_SSIM_arr = np.zeros((len(reg_params), 32))
        H_SSIM_arr = np.zeros((len(reg_params), 32))
        output_dim = str(128)
        d3 = d['avgs='+avg]['output_dim='+output_dim]
        D3 = D['avgs=' + avg]['output_dim='+output_dim]
        DD3 = DD['avgs=' + avg]['output_dim=' + output_dim]
        D_SSIM3 = D_SSIM['avgs='+avg]['output_dim='+output_dim]
        DD_SSIM3 = DD_SSIM['avgs=' + avg]['output_dim=' + output_dim]
        DDD_SSIM3 = DDD_SSIM['avgs=' + avg]['output_dim=' + output_dim]

        for i, reg_param in enumerate(reg_params):
            print("unpacking reg param " + '{:.1e}'.format(reg_param))

            discrep = np.asarray(d3['reg_param='+'{:.1e}'.format(reg_param)]).astype('float64')
            discrep_arr[i, :] = discrep

            GT_discrep = np.asarray(D3['reg_param=' + '{:.1e}'.format(reg_param)]).astype('float64')
            GT_discrep_arr[i, :] = GT_discrep

            GT_TV_discrep = np.asarray(DD3['reg_param=' + '{:.1e}'.format(reg_param)]).astype('float64')
            GT_TV_discrep_arr[i, :] = GT_TV_discrep

            GT_SSIM_vals = np.asarray(D_SSIM3['reg_param=' + '{:.1e}'.format(reg_param)]).astype('float64')
            GT_SSIM_arr[i, :] = GT_SSIM_vals

            GT_proxy_SSIM_vals = np.asarray(DD_SSIM3['reg_param=' + '{:.1e}'.format(reg_param)]).astype('float64')
            GT_proxy_SSIM_arr[i, :] = GT_proxy_SSIM_vals

            H_SSIM_vals = np.asarray(DDD_SSIM3['reg_param=' + '{:.1e}'.format(reg_param)]).astype('float64')
            H_SSIM_arr[i, :] = H_SSIM_vals

        axarr[0, 0].errorbar(np.log10(np.asarray(reg_params))[1:], np.average(GT_discrep_arr[1:], axis=1),
                     yerr=np.std(GT_discrep_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        axarr[0, 0].plot(np.log10(np.asarray(reg_params))[1:], Morozov_thresholds[k] * np.ones(25), color="C" + str(k % 10),
                 linestyle=":")
        axarr[0, 0].set_xlabel("log(lambda)")
        axarr[0, 0].set_ylabel("l2-discrepancy")
        axarr[0, 0].set_title("L2-discrepancy between "+output_dim+"-by-"+output_dim+" TV-regularised recons\n and 16384-averaged data")
        axarr[0, 0].legend()

        #
        axarr[0, 1].errorbar(np.log10(np.asarray(reg_params))[1:], np.average(GT_TV_discrep_arr[1:], axis=1),
                     yerr=np.std(GT_TV_discrep_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        axarr[0, 1].plot(np.log10(np.asarray(reg_params))[1:], Morozov_thresholds[k] * np.ones(25), color="C" + str(k % 10),
                 linestyle=":")
        axarr[0, 1].set_xlabel("log(lambda)")
        axarr[0, 1].set_ylabel("l2-discrepancy")
        axarr[0, 1].set_title("L2-discrepancy between "+output_dim+"-by-"+output_dim+" TV-regularised recons\n and synthetic ground-truth proxy")
        axarr[0, 1].legend()

        axarr[1, 0].errorbar(np.log10(np.asarray(reg_params))[1:], np.average(discrep_arr[1:], axis=1),
                     yerr=np.std(discrep_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        axarr[1, 0].plot(np.log10(np.asarray(reg_params))[1:], Morozov_thresholds[k]* np.ones(25) , color="C" + str(k % 10),
                 linestyle=":")
        axarr[1, 0].set_xlabel("log(lambda)")
        axarr[1, 0].set_ylabel("l2-discrepancy")
        axarr[1, 0].set_title("L2 data discrepancy for "+output_dim+"-by-"+output_dim+" TV-regularised recons")
        axarr[1, 0].legend()

        axarr[1, 1].errorbar(np.log10(np.asarray(reg_params))[1:], np.average(GT_SSIM_arr[1:], axis=1),
                     yerr=np.std(GT_SSIM_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        axarr[1, 1].set_xlabel("log(lambda)")
        axarr[1, 1].set_ylabel("SSIM")
        axarr[1, 1].set_title("SSIM between "+output_dim+"-by-"+output_dim+" TV-regularised recons\n and 16384-averaged data")
        axarr[1, 1].legend()

        axarr[2, 0].errorbar(np.log10(np.asarray(reg_params))[1:], np.average(GT_proxy_SSIM_arr[1:], axis=1),
                     yerr=np.std(GT_proxy_SSIM_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        axarr[2, 0].set_xlabel("log(lambda)")
        axarr[2, 0].set_ylabel("SSIM")
        axarr[2, 0].set_title(
            "SSIM between " + output_dim + "-by-" + output_dim + " TV-regularised recons\n and synthetic ground-truth proxy")
        axarr[2, 0].set_ylim(0., 1.)
        axarr[2, 0].set_yticks(np.linspace(0, 1, 11))
        axarr[2, 0].legend()

        axarr[2, 1].errorbar(np.log10(np.asarray(reg_params))[1:], np.average(H_SSIM_arr[1:], axis=1),
                     yerr=np.std(H_SSIM_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        axarr[2, 1].set_xlabel("log(lambda)")
        axarr[2, 1].set_ylabel("SSIM")
        axarr[2, 1].set_title(
            "SSIM between " + output_dim + "-by-" + output_dim + " TV-regularised recons\n and low-res H image")
        axarr[2, 1].set_ylim(0., 1.)
        axarr[2, 1].set_yticks(np.linspace(0, 1, 11))
        axarr[2, 1].legend()

    plt.tight_layout(w_pad=0.3, h_pad=0.6)
    plt.savefig('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/'+ext+'/discrepancy_plots_'+output_dim+'.pdf')

# stdev plots

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/TV/TV_aggregated_pixel_stds.json') as f:
        d = json.load(f)

    for k, avg in enumerate(avgs):

        stdev_arr = np.zeros(len(reg_params))
        d3 = d['avgs='+avg]['output_dim=32']

        for i, reg_param in enumerate(reg_params):

            stdev = d3['reg_param='+'{:.1e}'.format(reg_param)]
            stdev_arr[i] = stdev

        plt.plot(np.log10(reg_params)[1:], stdev_arr[1:]/32, label=avg+'avgs', color="C"+str(k%10))
        plt.xlabel("log10(lambda)")
        plt.ylabel("mean-squared stdev")
        plt.title("Mean-squared standard deviation of TV-regularised \n reconstructions, 32-by-32")
        plt.legend()

# Moran plots (for Li2SO4)

if plot_Moran:

    with open('/Users/jlw31/Desktop/Robustness_results_new/Robustness_31112020_TV_morans_I_new.json') as f:
        d = json.load(f)

    for k, avg in enumerate(avgs):
        print("unpacking average " + avg)

        Moran_arr = np.zeros((len(reg_params), 32))
        d3 = d['avgs='+avg]['output_dim=32']

        for i, reg_param in enumerate(reg_params):
            print("unpacking reg param " + '{:.1e}'.format(reg_param))

            Moran_I_vals = np.asarray(d3['reg_param=' + '{:.1e}'.format(reg_param)]).astype('float64')
            Moran_arr[i, :] = Moran_I_vals

        plt.errorbar(np.log10(np.asarray(reg_params))[1:], np.average(Moran_arr, axis=1)[1:],
                     yerr=np.std(Moran_arr, axis=1)[1:],
                     label=avg + 'avgs', color="C" + str(k % 10))
        plt.xlabel("log(lambda)")
        plt.ylabel("Moran's I")
        plt.title("Moran's I for 32-by-32 TV-regularised recons")
        plt.legend()

## dTV results

alphas = np.concatenate((np.asarray([0.001, 1., 10**0.5, 10., 10**1.5, 10**2]), np.logspace(2.5, 4.75, num=20)))

if plot_dTV_results:

    # GT_TV_data_arr = np.load(dir + 'Results_15032021/example_TV_recon_15032021_synth_data.npy')
    # GT_TV_data = np.fft.fftshift(GT_TV_data_arr[0] + 1j*GT_TV_data_arr[1])
    # GT_TV_image = np.load(dir + 'Results_15032021/example_TV_recon_15032021.npy')
    # GT_TV_image = np.abs(GT_TV_image[0] + 1j * GT_TV_image[1])
    # low_res_H_image = np.load('dTV/MRI_15032021/Results_15032021/pre_registered_H_image_low_res.npy')
    # low_res_H_image_normalised = low_res_H_image / np.sqrt(np.sum(np.square(low_res_H_image)))

    if date == '15032021':
        GT_TV_data = np.load(dir + 'Results_15032021/example_TV_recon_15032021_synth_data.npy')
        GT_TV_image = np.load(dir + 'Results_15032021/example_TV_recon_15032021.npy')
        GT_TV_image = np.abs(GT_TV_image[0] + 1j*GT_TV_image[1])
        low_res_H_image = np.load('dTV/MRI_15032021/Results_15032021/pre_registered_H_image_low_res.npy')
        low_res_H_image_normalised = low_res_H_image/np.sqrt(np.sum(np.square(low_res_H_image)))

    elif date == '24052021':
        GT_TV_data = np.load(dir + 'Results_24052021/example_TV_recon_24052021_synth_data.npy')
        GT_TV_image = np.load(dir + 'Results_24052021/example_TV_recon_24052021.npy')
        GT_TV_image = np.abs(GT_TV_image[0] + 1j * GT_TV_image[1])
        image_H_high_res = np.load('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res.npy')
        low_res_H_image = resize(image_H_high_res, (40, 40))
        low_res_H_image_normalised = low_res_H_image / np.sqrt(np.sum(np.square(low_res_H_image)))

    norms_dict = {}
    GT_norms_dict = {}
    GT_TV_norms_dict = {}
    GT_SSIM_dict = {}
    GT_TV_SSIM_dict = {}
    H_SSIM_dict = {}
    stdevs = {}
    affine_param_dict = {}

    for j, avg in enumerate(avgs):
        norms_dict['avgs=' + avg] = {}
        GT_norms_dict['avgs=' + avg] = {}
        GT_TV_norms_dict['avgs=' + avg] = {}
        GT_SSIM_dict['avgs=' + avg] = {}
        GT_TV_SSIM_dict['avgs=' + avg] = {}
        H_SSIM_dict['avgs=' + avg] = {}
        stdevs['avgs=' + avg] = {}

        with open(save_dir + 'dTV_results_pre_registered/dTV_7Li_'+date+'_' + avg +'_pre_registered.json') as f:
            d = json.load(f)

        print(save_dir + 'dTV_results_pre_registered/dTV_7Li_'+date+'_' + avg +'_pre_registered.json')

        # grabbing just the affine params, and putting into new dictionary

        affine_param_dict['avgs=' + avg] = {mkey: {skey: {akey: aval['affine_params'] for akey, aval in sval.items()}
                                   for skey, sval in mval.items()} for mkey, mval in d.items()}

        # grabbing dataset again to compute GT diff

        f_coeff_list = []
        for i in Li_range:
            f_coeffs = np.reshape(np.fromfile(dir + 'Data_'+date+'/Li_data/' + str(i) + '/fid', dtype=np.int32), low_res_shape)
            f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs, low_res_data_width)
            f_coeff_list.append(f_coeffs_unpacked)

        f_coeff_arr = np.asarray(f_coeff_list)
        fully_averaged_coeffs = np.average(f_coeff_arr, axis=0)
        fully_averaged_shifted = np.fft.fftshift(fully_averaged_coeffs)
        recon_fully_averaged = np.fft.fftshift(np.fft.ifft2(fully_averaged_shifted))
        GT_image = np.abs(recon_fully_averaged)

        f_coeff_list_grouped = []
        num = int(2 ** j)
        for i in range(num):
            data_arr = np.roll(f_coeff_arr, i, axis=0)
            for ele in range(len(f_coeff_list) // num):
                f_coeff_list_grouped.append(np.sum(data_arr[num * ele:num * (ele + 1)], axis=0) / num)

        coeffs = f_coeff_list_grouped
        coeffs_minus_GT = coeffs - fully_averaged_coeffs
        coeffs_minus_GT_TV = coeffs - (GT_TV_data[0] + 1j*GT_TV_data[1])

        for output_dim in output_dims:
            GT_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            GT_TV_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            GT_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            GT_TV_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            H_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            stdevs['avgs=' + avg]['output_dim=' + str(output_dim)] = {}

            for alpha in alphas:
                GT_diff_norms = []
                GT_TV_diff_norms = []
                GT_SSIM_vals = []
                GT_TV_SSIM_vals = []
                H_SSIM_vals = []
                diff_norms = []
                recons = []

                fig, axs = plt.subplots(16, 4, figsize=(4, 10))
                for i in range(32):

                    print('\noutput_dims_' + str(output_dim))
                    print('measurement_' + str(i))
                    print(d['measurement=' + str(i)]['output_size=' + str(output_dim)].keys())

                    recon = np.asarray(d['measurement=' + str(i)]['output_size=' + str(output_dim)][
                        'alpha=' + '{:.1e}'.format(alpha)]['recon']).astype('float64')

                    # fourier_diff = np.asarray(d['measurement=' + str(i)]['output_size=' + str(output_dim)][
                    #     'alpha=' + '{:.1e}'.format(alpha)]['fourier_diff']).astype('float64')

                    # NOTE: for the 24052021 dataset, there's an error in the fourier diffs that were saved alongside
                    # the reconstructions, so we have to recompute them here

                    complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                                      shape=[output_dim, output_dim], dtype='complex')
                    image_space = complex_space.real_space ** 2
                    forward_op = RealFourierTransform(image_space)

                    data_odl = forward_op.range.element([np.real(np.fft.fftshift(coeffs[i])),
                                                         np.imag(np.fft.fftshift(coeffs[i]))])
                    diff = forward_op(forward_op.domain.element([recon[0], recon[1]])) - data_odl
                    diff = diff[0].asarray() + 1j * diff[1].asarray()
                    diff_shift = np.fft.ifftshift(diff)
                    fourier_diff = diff_shift[output_dim // 2 - low_res_data_width//2:output_dim // 2 + low_res_data_width//2,
                                        output_dim // 2 - low_res_data_width//2:output_dim // 2 + low_res_data_width//2]
                    fourier_diff = np.asarray([np.real(fourier_diff), np.imag(fourier_diff)])

                    print(np.shape(fourier_diff))
                    print(np.shape(coeffs_minus_GT[i, :, :]))

                    GT_fourier_diff = fourier_diff[0] + 1j*fourier_diff[1] + coeffs_minus_GT[i, :, :]
                    GT_TV_fourier_diff = fourier_diff[0] + 1j*fourier_diff[1] + coeffs_minus_GT_TV[i, :, :]

                    recon_image = np.abs(recon[0] + 1j * recon[1])
                    fourier_diff_image = np.abs(fourier_diff[0] + 1j*fourier_diff[1])

                    # grabbing the 32x32 or 40x40 reconstruction from the synthetic K-space data
                    # if output_dim == int(64):
                    #     fourier_complex = np.fft.fft2(recon[0] + 1j * recon[1])
                    #     fourier_shift = np.fft.ifftshift(fourier_complex)
                    #     fourier_shift_subsampled = fourier_shift[32 - 16:32 + 16, 32 - 16:32 + 16]
                    #     rec_fourier = np.fft.ifft2(np.fft.fftshift(fourier_shift_subsampled))
                    #     rec_low_res = np.abs(rec_fourier)
                    #
                    # if output_dim == int(128):
                    #     fourier_complex = np.fft.fft2(recon[0] + 1j * recon[1])
                    #     fourier_shift = np.fft.ifftshift(fourier_complex)
                    #     fourier_shift_subsampled = fourier_shift[64 - 16:64 + 16, 64 - 16:64 + 16]
                    #     rec_fourier = np.fft.ifft2(np.fft.fftshift(fourier_shift_subsampled))
                    #     rec_low_res = np.abs(rec_fourier)

                    fourier_complex = np.fft.fft2(recon[0] + 1j * recon[1])
                    fourier_shift = np.fft.ifftshift(fourier_complex)
                    fourier_shift_subsampled = fourier_shift[output_dim//2 - low_res_data_width//2:output_dim//2 + low_res_data_width//2,
                                               output_dim//2 - low_res_data_width//2:output_dim//2 + low_res_data_width//2]
                    rec_fourier = np.fft.ifft2(np.fft.fftshift(fourier_shift_subsampled))
                    rec_low_res = np.abs(rec_fourier)

                    # example data, just to check consistency of fftshifts etc....
                    if j == 4 and output_dim == int(32) and alpha == alphas[15] and i == 5:
                        data_array = np.asarray(
                            [[np.real(coeffs[i]), np.imag(coeffs[i])], fourier_diff, [np.real(coeffs_minus_GT[i, :, :]), np.imag(coeffs_minus_GT[i, :, :])],
                             [np.real(coeffs_minus_GT_TV[i, :, :]), np.imag(coeffs_minus_GT_TV[i, :, :])],
                             [np.real(GT_TV_data), np.imag(GT_TV_data)]])

                        np.save('dTV/Results_MRI_dTV/dTV_debugging_fft_shifts.npy', data_array)

                    axs[2 * (i // 4), i % 4].imshow(recon_image, cmap=plt.cm.gray, interpolation='none')
                    axs[2 * (i // 4), i % 4].axis("off")

                    axs[1 + 2 * (i // 4), i % 4].imshow(fourier_diff_image, cmap=plt.cm.gray, interpolation='none')
                    axs[1 + 2 * (i // 4), i % 4].axis("off")

                    diff_norms.append(np.sqrt(np.sum(np.square(fourier_diff_image))))
                    recons.append(recon_image)
                    GT_diff_norms.append(np.sqrt(np.sum(np.square(np.abs(GT_fourier_diff)))))
                    GT_TV_diff_norms.append(np.sqrt(np.sum(np.square(np.abs(GT_TV_fourier_diff)))))

                    # SSIM vals

                    if date=='15032021':
                        if output_dim == int(32):
                            image = recon_image
                        elif output_dim == int(64):
                            image = rec_low_res
                        elif output_dim == int(128):
                            image = rec_low_res

                    elif date=='24052021':
                        if output_dim == int(40):
                            image = recon_image
                        elif output_dim == int(80):
                            image = rec_low_res
                        elif output_dim == int(128):
                            image = rec_low_res

                    image_normalised = image/np.sqrt(np.sum(np.square(image)))
                    GT_image_normalised = GT_image/np.sqrt(np.sum(np.square(GT_image)))
                    GT_TV_image_normalised = GT_TV_image / np.sqrt(np.sum(np.square(GT_TV_image)))
                    GT_SSIM = recon_error(image_normalised, GT_image_normalised)[2]
                    GT_TV_SSIM = recon_error(image_normalised, GT_TV_image_normalised)[2]
                    H_SSIM = recon_error(image_normalised, low_res_H_image_normalised)[2]
                    GT_SSIM_vals.append(GT_SSIM)
                    GT_TV_SSIM_vals.append(GT_TV_SSIM)
                    H_SSIM_vals.append(H_SSIM)

                fig.tight_layout(w_pad=0.4, h_pad=0.4)
                plt.savefig(save_dir + "dTV_results_pre_registered/" + avg +"_avgs/" + str(output_dim) +"/dTV_no_regis_"+date+"_data_" + avg + "_avgs_32_to_" + str(
                    output_dim) + "_reg_param_" + '{:.1e}'.format(alpha) + "_new.pdf")
                plt.close()

                norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                    'reg_param=' + '{:.1e}'.format(alpha)] = diff_norms

                GT_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                    'reg_param=' + '{:.1e}'.format(alpha)] = GT_diff_norms

                GT_TV_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                    'reg_param=' + '{:.1e}'.format(alpha)] = GT_TV_diff_norms

                GT_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                    'reg_param=' + '{:.1e}'.format(alpha)] = GT_SSIM_vals

                GT_TV_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                    'reg_param=' + '{:.1e}'.format(alpha)] = GT_TV_SSIM_vals

                H_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                    'reg_param=' + '{:.1e}'.format(alpha)] = H_SSIM_vals

                stdev = np.sqrt(np.sum(np.square(np.std(recons, axis=0))))
                stdevs['avgs=' + avg]['output_dim=' + str(output_dim)]['reg_param=' + '{:.1e}'.format(alpha)] = stdev

                plt.figure()
                plt.imshow(np.std(recons, axis=0), cmap=plt.cm.gray)
                plt.colorbar()
                plt.savefig(save_dir + "dTV_results_pre_registered/" + avg +"_avgs/" + str(output_dim) +"/dTV_no_regis_"+date+"_data_" + avg + "_avgs_32_to_" + str(
                    output_dim) + "reg_param_" + '{:.1e}'.format(alpha) + 'stdev_plot_new.pdf')
                plt.close()

    json.dump(norms_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_'+date+'_pre_registered_fidelities.json', 'w'))

    json.dump(GT_norms_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_'+date+'_pre_registered_GT_fidelities.json', 'w'))

    json.dump(GT_TV_norms_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_'+date+'_pre_registered_GT_from_TV_fidelities.json', 'w'))

    json.dump(GT_SSIM_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_'+date+'_pre_registered_GT_SSIM_vals.json', 'w'))

    json.dump(GT_TV_SSIM_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_'+date+'_pre_registered_GT_proxy_SSIM_vals.json', 'w'))

    json.dump(H_SSIM_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_'+date+'_pre_registered_H_SSIM_vals.json',
                   'w'))

    json.dump(stdevs,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_'+date+'_pre_registered_aggregated_pixel_stds.json', 'w'))

    #json.dump(affine_param_dict,
     #         open(save_dir + '/New/results/dTV_results_no_regis/Robustness_31112020_dTV_affine_params_new.json', 'w'))

# plotting data discrepancies - this is done locally: need to copy above json files into local directory

if dTV_discrepancy_plots:

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/dTV_pre_registered/dTV_7Li_'+date+'_pre_registered_fidelities.json') as f:
        d = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/dTV_pre_registered/dTV_7Li_'+date+'_pre_registered_GT_fidelities.json') as f:
        D = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/dTV_pre_registered/dTV_7Li_'+date+'_pre_registered_GT_from_TV_fidelities.json') as f:
        DD = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/dTV_pre_registered/dTV_7Li_'+date+'_pre_registered_GT_SSIM_vals.json') as f:
        D_SSIM = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/dTV_pre_registered/dTV_7Li_'+date+'_pre_registered_GT_proxy_SSIM_vals.json') as f:
        DD_SSIM = json.load(f)

    f.close()

    with open(
            '/Users/jlw31/Desktop/Results_on_'+date+'_dataset/dTV_pre_registered/dTV_7Li_'+date+'_pre_registered_H_SSIM_vals.json') as f:
        DDD_SSIM = json.load(f)

    f.close()

    if date == '15032021':
        Morozov_thresholds = [101165, 70547, 48230, 31739, 18380]
    elif date == '24052021':
        Morozov_thresholds = [126000, 88000, 60300, 39500, 22800]

    Fourier_SSIMs_GT_proxy = [0.1, 0.16, 0.26, 0.37, 0.5]

    f, axarr = plt.subplots(3, 2, figsize=(10, 10))

    for k, avg in enumerate(avgs):

        discrep_arr = np.zeros((len(alphas), 32))
        GT_discrep_arr = np.zeros((len(alphas), 32))
        GT_TV_discrep_arr = np.zeros((len(alphas), 32))
        GT_SSIM_arr = np.zeros((len(alphas), 32))
        GT_proxy_SSIM_arr = np.zeros((len(alphas), 32))
        H_SSIM_arr = np.zeros((len(alphas), 32))
        output_dim = str(80)
        d3 = d['avgs='+avg]['output_dim='+output_dim]
        D3 = D['avgs=' + avg]['output_dim=' + output_dim]
        DD3 = DD['avgs=' + avg]['output_dim=' + output_dim]
        D_SSIM3 = D_SSIM['avgs=' + avg]['output_dim=' + output_dim]
        DD_SSIM3 = DD_SSIM['avgs=' + avg]['output_dim=' + output_dim]
        DDD_SSIM3 = DDD_SSIM['avgs=' + avg]['output_dim=' + output_dim]

        for i, alpha in enumerate(alphas):

            discrep = np.asarray(d3['reg_param='+'{:.1e}'.format(alpha)]).astype('float64')
            discrep_arr[i, :] = discrep

            GT_discrep = np.asarray(D3['reg_param=' + '{:.1e}'.format(alpha)]).astype('float64')
            GT_discrep_arr[i, :] = GT_discrep

            GT_TV_discrep = np.asarray(DD3['reg_param=' + '{:.1e}'.format(alpha)]).astype('float64')
            GT_TV_discrep_arr[i, :] = GT_TV_discrep

            GT_SSIM_vals = np.asarray(D_SSIM3['reg_param=' + '{:.1e}'.format(alpha)]).astype('float64')
            GT_SSIM_arr[i, :] = GT_SSIM_vals

            GT_proxy_SSIM_vals = np.asarray(DD_SSIM3['reg_param=' + '{:.1e}'.format(alpha)]).astype('float64')
            GT_proxy_SSIM_arr[i, :] = GT_proxy_SSIM_vals

            H_SSIM_vals = np.asarray(DDD_SSIM3['reg_param=' + '{:.1e}'.format(alpha)]).astype('float64')
            H_SSIM_arr[i, :] = H_SSIM_vals

        axarr[0, 1].errorbar(np.log10(np.asarray(alphas))[1:], np.average(GT_discrep_arr, axis=1)[1:],
                     yerr=np.std(GT_discrep_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        axarr[0, 1].plot(np.log10(np.asarray(alphas))[1:], Morozov_thresholds[k] * np.ones(25), color="C" + str(k % 10),
                 linestyle=":")
        axarr[0, 1].set_xlabel("log(alpha)")
        axarr[0, 1].set_ylabel("l2-discrepancy")
        axarr[0, 1].set_title("L2-discrepancy between "+output_dim+"-by-"+output_dim+" dTV-regularised recons\n and 16384-averaged data")
        axarr[0, 1].legend()

        axarr[1, 0].errorbar(np.log10(np.asarray(alphas))[1:], np.average(GT_TV_discrep_arr, axis=1)[1:],
                     yerr=np.std(GT_TV_discrep_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        axarr[1, 0].plot(np.log10(np.asarray(alphas))[1:], Morozov_thresholds[k] * np.ones(25),
                 color="C" + str(k % 10),
                 linestyle=":")
        axarr[1, 0].set_xlabel("log(alpha)")
        axarr[1, 0].set_ylabel("l2-discrepancy")
        axarr[1, 0].set_title(
            "L2-discrepancy between " + output_dim + "-by-" + output_dim + " dTV-regularised recons\n and groundtruth proxy")
        axarr[1, 0].legend()

        axarr[0, 0].errorbar(np.log10(np.asarray(alphas))[1:], np.average(discrep_arr, axis=1)[1:],
                             yerr=np.std(discrep_arr[1:], axis=1),
                             label=avg + 'avgs', color="C" + str(k % 10))
        axarr[0, 0].plot(np.log10(np.asarray(alphas))[1:], Morozov_thresholds[k] * np.ones(25), color="C" + str(k % 10),
                         linestyle=":")
        axarr[0, 0].set_xlabel("log(alpha)")
        axarr[0, 0].set_ylabel("l2-discrepancy")
        axarr[0, 0].set_title("L2 data discrepancy for " + output_dim + "-by-" + output_dim + " dTV-regularised recons")
        axarr[0, 0].legend()

        axarr[1, 1].errorbar(np.log10(np.asarray(alphas))[1:], np.average(GT_proxy_SSIM_arr, axis=1)[1:],
                     yerr=np.std(GT_proxy_SSIM_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        # plt.plot(np.log10(np.asarray(alphas))[1:], Fourier_SSIMs_GT_proxy[k] * np.ones(25),
        #                   color="C" + str(k % 10),
        #                   linestyle=":")
        axarr[1, 1].set_xlabel("log(alpha)")
        axarr[1, 1].set_ylabel("SSIM")
        axarr[1, 1].set_title(
            "SSIM between " + output_dim + "-by-" + output_dim + " dTV-regularised recons\n and groundtruth proxy")
        axarr[1, 1].set_ylim(0., 1.)
        axarr[1, 1].set_yticks(np.linspace(0, 1, 11))
        axarr[1, 1].legend()

        axarr[2, 0].errorbar(np.log10(np.asarray(alphas))[1:], np.average(H_SSIM_arr, axis=1)[1:],
                     yerr=np.std(H_SSIM_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        axarr[2, 0].set_xlabel("log(alpha)")
        axarr[2, 0].set_ylabel("SSIM")
        axarr[2, 0].set_title(
            "SSIM between " + output_dim + "-by-" + output_dim + " dTV-regularised recons\n and low-res H image")
        axarr[2, 0].set_ylim(0., 1.)
        axarr[2, 0].set_yticks(np.linspace(0, 1, 11))
        axarr[2, 0].legend()

    plt.tight_layout(w_pad=0.3, h_pad=0.6)
    plt.savefig('/Users/jlw31/Desktop/Results_on_'+date+'_dataset/dTV_pre_registered/discrepancy_plot_'+str(output_dim)+'_.pdf')

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/dTV_pre_registered/dTV_7Li_15032021_pre_registered_aggregated_pixel_stds.json') as f:
        d = json.load(f)

    for k, avg in enumerate(avgs):

        stdev_arr = np.zeros(len(alphas))
        d3 = d['avgs='+avg]['output_dim=128']

        for i, alpha in enumerate(alphas):

            stdev = d3['reg_param='+'{:.1e}'.format(alpha)]
            stdev_arr[i] = stdev

        plt.plot(np.log10(alphas)[1:], stdev_arr[1:]/32, label=avg+'avgs', color="C"+str(k%10))
        plt.xlabel("log10(lambda)")
        plt.ylabel("mean-squared stdev")
        plt.title("Mean-squared standard deviation of dTV-regularised \n reconstructions, 128-by-128")
        plt.legend()

if affine_param_plots:

    with open('/Users/jlw31/Desktop/Robustness_results_new/Li2SO4_results/Li2SO4_dTV_results/Robustness_31112020_dTV_affine_params_new.json') as f:
        d = json.load(f)

    for output_dim in output_dims:

        fig, axs = plt.subplots(3, 2, figsize=(2, 3))

        for k in range(6):
            for m, avg in enumerate(avgs):
                affine_params = np.zeros((32, len(alphas)))
                for l, alpha in enumerate(alphas):
                    for i in range(32):
                        affine_param = d['avgs=' + avg]['measurement=' + str(i)]['output_size=' + str(output_dim)]['alpha=' + '{:.1e}'.format(alpha)][k]

                        affine_params[i, l] = affine_param

                axs[k//2, k%2].errorbar(np.log10(alphas), np.average(affine_params, axis=0), yerr=np.std(affine_params, axis=0),
                                        label=avg+'avgs', color="C"+str(m%10))

            plt.legend()

    ## Rough
    # using some example affine params to deform a template (we take the H-MRI image, here), to get an idea of their variability

    # template = np.zeros((32, 32))
    # template[10:22, 15:17] = 1
    # template[15:17, 10:22] = 1
    # template = np.load('/Users/jlw31/PycharmProjects/TV_pipeline/dTV/Results_MRI_dTV/'
    #                            'example_TV_recon_Li2SO4_16384_avgs_reg_param_1000.npy').T[:, ::-1]

    image_H = np.reshape(np.fromfile('dTV/7Li_1H_MRI_Data_31112020/1H_Li2SO4/6/pdata/1/2dseq', dtype=np.uint16), (128, 128))
    template = block_reduce(image_H.T, block_size=(4, 4), func=np.mean)
    X = odl.uniform_discr([-1, -1], [1, 1], [32, 32], dtype='float32')
    #deform_op = odl.deform.LinDeformFixedTempl(X.element(template))
    deform_op = dTV.myDeform.LinDeformFixedTempl(X.element(template))

    V = X.tangent_bundle
    Y = odl.tensor_space(6)

    embed = Embedding_Affine(Y, V)
    transl_operator = deform_op * embed

    alpha = alphas[len(alphas) // 2]
    #alpha = alphas[-1]

    fig, axs = plt.subplots(10, 5, figsize=(5, 10))

    for m, avg in enumerate(avgs):
        axs[0, m].imshow(template)
        axs[0, m].axis("off")
        for i, j in enumerate(np.random.choice(range(32), size=9, replace=False)):
            affine_params = d['avgs=' + avg]['measurement=' + str(j)]['output_size=' + str(32)][
                'alpha=' + '{:.1e}'.format(alpha)]

            deformed_template = transl_operator(affine_params).asarray()

            #axs[i, m].imshow(np.abs(deformed_template - template), cmap=plt.cm.gray)
            axs[i+1, m].imshow(deformed_template)
            axs[i+1, m].axis("off")

## Best (defensible) reconstructions from various experiments

avgs = [512, 1024, 2048]
model_param_dict = {'512': [8.9*10**3, 1.1*10**4, 1.9*10**4, 1.9*10**4], '1024': [5.1*10**3, 6.3*10**3, 8.3*10**3, 8.3*10**3],
                    '2048': [3.0*10**3, 2.8*10**3, 3.7*10**3, 4.8*10**3]}

if best_recons:
    # grabbing the fully-averaged recon
    f_coeff_list = []
    for i in range(3, 35):
        f_coeffs = np.reshape(np.fromfile(dir + 'Data_15032021/Li_data/' + str(i) + '/fid', dtype=np.int32), (64, 128))
        f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs)
        f_coeff_list.append(f_coeffs_unpacked)

    f_coeff_arr = np.asarray(f_coeff_list)
    fully_averaged_coeffs = np.average(f_coeff_arr, axis=0)
    fully_averaged_shifted = np.fft.fftshift(fully_averaged_coeffs)
    recon_fully_averaged = np.fft.fftshift(np.fft.ifft2(fully_averaged_shifted))
    GT_image = np.abs(recon_fully_averaged)


    for avg in avgs:

        with open(save_dir + 'TV_results/TV_7Li_15032021_' + str(avg) + '.json') as f:
            d_TV = json.load(f)

        f.close()

        with open(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_' + str(avg) + '_pre_registered.json') as f:
            d_dTV = json.load(f)

        f.close()

        model_params = model_param_dict[str(avg)]
        # I shouldn't be copying this code all over the place! Put it somewhere more central
        if avg != 512:
            # f_coeff_arr = np.asarray(f_coeff_list)
            f_coeff_list_grouped = []
            num = avg // 512
            for i in range(num):
                data_arr = np.roll(f_coeff_arr, i, axis=0)
                for ele in range(len(f_coeff_list) // num):
                    f_coeff_list_grouped.append(np.sum(data_arr[num * ele:num * (ele + 1)], axis=0) / num)

            f_coeff_list = f_coeff_list_grouped
            f_coeff_arr = np.asarray(f_coeff_list)

        fig, axs = plt.subplots(6, 5, figsize=(10, 14))

        for j in range(5):
            axs[0, j].imshow(GT_image, cmap=plt.cm.gray)
            axs[0, j].axis("off")

        for i in range(5):

            recon_TV = np.asarray(d_TV['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(model_params[0])]
                               ['output_size=' + str(32)]).astype('float64')
            image_TV = np.abs(recon_TV[0] + 1j * recon_TV[1])

            recon_dTV_32 = np.asarray(d_dTV['measurement=' + str(i)]['output_size=' + str(32)][
                                   'alpha=' + '{:.1e}'.format(model_params[1])]['recon']).astype('float64')
            image_dTV_32 = np.abs(recon_dTV_32[0] + 1j * recon_dTV_32[1])

            recon_dTV_64 = np.asarray(d_dTV['measurement=' + str(i)]['output_size=' + str(64)][
                                           'alpha=' + '{:.1e}'.format(model_params[2])]['recon']).astype('float64')

            recon_dTV_128 = np.asarray(d_dTV['measurement=' + str(i)]['output_size=' + str(128)][
                                          'alpha=' + '{:.1e}'.format(model_params[3])]['recon']).astype('float64')

            fourier_complex = np.fft.fft2(recon_dTV_64[0] + 1j * recon_dTV_64[1])
            fourier_shift = np.fft.ifftshift(fourier_complex)
            fourier_shift_subsampled = fourier_shift[32 - 16:32 + 16, 32 - 16:32 + 16]
            rec_fourier = np.fft.ifft2(np.fft.fftshift(fourier_shift_subsampled))
            dTV_recon_from_64 = np.abs(rec_fourier)

            fourier_complex = np.fft.fft2(recon_dTV_128[0] + 1j * recon_dTV_128[1])
            fourier_shift = np.fft.ifftshift(fourier_complex)
            fourier_shift_subsampled = fourier_shift[64 - 16:64 + 16, 64 - 16:64 + 16]
            rec_fourier = np.fft.ifft2(np.fft.fftshift(fourier_shift_subsampled))
            dTV_recon_from_128 = np.abs(rec_fourier)

            f_data = f_coeff_arr[i, :, :]
            f_data_shifted = np.fft.fftshift(f_data)
            recon_fourier = np.fft.fftshift(np.fft.ifft2(f_data_shifted))

            axs[i+1, 0].imshow(np.abs(recon_fourier), cmap=plt.cm.gray)
            axs[i+1, 0].axis("off")
            axs[i+1, 1].imshow(image_TV, cmap=plt.cm.gray)
            axs[i+1, 1].axis("off")
            axs[i+1, 2].imshow(image_dTV_32, cmap=plt.cm.gray)
            axs[i+1, 2].axis("off")
            axs[i + 1, 3].imshow(dTV_recon_from_64, cmap=plt.cm.gray)
            axs[i + 1, 3].axis("off")
            axs[i+1, 4].imshow(dTV_recon_from_128, cmap=plt.cm.gray)
            axs[i+1, 4].axis("off")

            fig.tight_layout(w_pad=0.3, h_pad=0.2)
            plt.savefig(save_dir + "best_recons_"+str(avg)+".pdf")

# processing data from hyperparameter sweep

etas = np.logspace(-3, -1, num=10)

def retrieving_lower_res_image(recon):

    fourier_complex = np.fft.fft2(recon[0] + 1j * recon[1])
    fourier_shift = np.fft.ifftshift(fourier_complex)
    fourier_shift_subsampled = fourier_shift[32 - 16:32 + 16, 32 - 16:32 + 16]
    rec_fourier = np.fft.ifft2(np.fft.fftshift(fourier_shift_subsampled))
    rec_low_res = np.abs(rec_fourier)
    rec_low_res_normalised = rec_low_res/np.sqrt(np.sum(np.square(rec_low_res)))

    return rec_low_res_normalised

if hyperparam_sweep_results:

    GT_TV_image = np.load(dir + 'Results_15032021/example_TV_recon_15032021.npy')
    GT_TV_image = np.abs(GT_TV_image[0] + 1j * GT_TV_image[1])
    GT_TV_image_normalised = GT_TV_image / np.sqrt(np.sum(np.square(GT_TV_image)))

    with open(
            save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_1024_pre_registered_hyper_search_gamma_9.0e-01.json') as f:
        d1 = json.load(f)
    f.close()

    with open(
            save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_1024_pre_registered_hyper_search_gamma_9.3e-01.json') as f:
        d2 = json.load(f)
    f.close()

    with open(
            save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_1024_pre_registered_hyper_search_gamma_9.5e-01.json') as f:
        d3 = json.load(f)
    f.close()

    with open(
            save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_1024_pre_registered_hyper_search_gamma_9.7e-01.json') as f:
        d4 = json.load(f)
    f.close()

    with open(
            save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_1024_pre_registered_hyper_search_gamma_9.9e-01.json') as f:
        d5 = json.load(f)
    f.close()

    #SSIMs = np.array((5, len(etas), 32))
    D = {}

    for i in range(32):
        D['measurement=' + str(i)] = {}
        D['measurement=' + str(i)]['gamma=' + str(9.0e-01)] = {}
        D['measurement=' + str(i)]['gamma=' + str(9.3e-01)] = {}
        D['measurement=' + str(i)]['gamma=' + str(9.5e-01)] = {}
        D['measurement=' + str(i)]['gamma=' + str(9.7e-01)] = {}
        D['measurement=' + str(i)]['gamma=' + str(9.9e-01)] = {}

        for k, eta in enumerate(etas):

            rec_1 = np.asarray(d1['measurement=' + str(i)]['output_size=' + str(64)]['eta=' + '{:.1e}'.format(eta)]['recon'])
            rec_2 = np.asarray(
                d2['measurement=' + str(i)]['output_size=' + str(64)]['eta=' + '{:.1e}'.format(eta)]['recon'])
            rec_3 = np.asarray(
                d3['measurement=' + str(i)]['output_size=' + str(64)]['eta=' + '{:.1e}'.format(eta)]['recon'])
            rec_4 = np.asarray(
                d4['measurement=' + str(i)]['output_size=' + str(64)]['eta=' + '{:.1e}'.format(eta)]['recon'])
            rec_5 = np.asarray(
                d5['measurement=' + str(i)]['output_size=' + str(64)]['eta=' + '{:.1e}'.format(eta)]['recon'])

            rec_1_32 = retrieving_lower_res_image(rec_1)
            rec_2_32 = retrieving_lower_res_image(rec_2)
            rec_3_32 = retrieving_lower_res_image(rec_3)
            rec_4_32 = retrieving_lower_res_image(rec_4)
            rec_5_32 = retrieving_lower_res_image(rec_5)

            SSIM_1 = recon_error(rec_1_32, GT_TV_image_normalised)[2]
            SSIM_2 = recon_error(rec_2_32, GT_TV_image_normalised)[2]
            SSIM_3 = recon_error(rec_3_32, GT_TV_image_normalised)[2]
            SSIM_4 = recon_error(rec_4_32, GT_TV_image_normalised)[2]
            SSIM_5 = recon_error(rec_5_32, GT_TV_image_normalised)[2]

            # SSIMs[0, k, i] = SSIM_1
            # SSIMs[1, k, i] = SSIM_2
            # SSIMs[2, k, i] = SSIM_3
            # SSIMs[3, k, i] = SSIM_4
            # SSIMs[4, k, i] = SSIM_5

            D['measurement=' + str(i)]['gamma=' + str(9.0e-01)]['eta=' + '{:.1e}'.format(eta)] = SSIM_1
            D['measurement=' + str(i)]['gamma=' + str(9.3e-01)]['eta=' + '{:.1e}'.format(eta)] = SSIM_2
            D['measurement=' + str(i)]['gamma=' + str(9.5e-01)]['eta=' + '{:.1e}'.format(eta)] = SSIM_3
            D['measurement=' + str(i)]['gamma=' + str(9.7e-01)]['eta=' + '{:.1e}'.format(eta)] = SSIM_4
            D['measurement=' + str(i)]['gamma=' + str(9.9e-01)]['eta=' + '{:.1e}'.format(eta)] = SSIM_5

    json.dump(D,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_pre_registered_hyper_search_proxy_SSIMs.json',
                   'w'))

gammas = [9.0e-01, 9.3e-01, 9.5e-01, 9.7e-01, 9.9e-01]

if plot_hyperparam_sweep_results:

    d = json.load(open('/Users/jlw31/Desktop/Results_on_15032021_dataset/dTV_pre_registered/'
                       'dTV_7Li_15032021_pre_registered_hyper_search_proxy_SSIMs.json', 'r'))

    output_dim = str(64)

    for k, gamma in enumerate(gammas):
        SSIM_arr = np.zeros((len(etas), 32))
        for i in range(32):
            for j, eta in enumerate(etas):

                d2 = d['measurement=' + str(i)]
                d3 = d2['gamma='+str(gamma)]
                SSIM = d3['eta=' + '{:.1e}'.format(eta)]
                SSIM_arr[j, i] = SSIM

        plt.errorbar(np.log10(np.asarray(etas)), np.average(SSIM_arr, axis=1),
                     yerr=np.std(SSIM_arr, axis=1),
                     label='gamma=' + str(gamma), color="C" + str(k%10))
        plt.xlabel("log(eta)")
        plt.ylabel("SSIM")
        plt.title(
            "SSIM between " + output_dim + "-by-" + output_dim + " dTV-regularised recons\n and groundtruth proxy, varying hyperparams")
        plt.ylim(0., 1.)
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend()


