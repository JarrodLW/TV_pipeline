import numpy as np
import json
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform

plot_TV_results = True
plot_dTV_results = False
plot_TV_results_full_avgs = False
discrepancy_plots = False
dTV_discrepancy_plots = False

#avgs = ['512', '1024', '2048', '4096', '8192']
avgs = ['1024', '2048', '4096', '8192']
#avgs = ['512']
#reg_params = np.logspace(np.log10(2e3), np.log10(1e5), num=20)
reg_params = np.logspace(3., 4.5, num=20)
output_dims = [int(32), int(64)]

dir = '7Li_1H_MRI_Data_31112020/'

def unpacking_fourier_coeffs(arr):

    fourier_real_im = arr[:, 1:65]
    fourier_real_im = fourier_real_im[::2, :]

    fourier_real = fourier_real_im[:, 1::2]
    fourier_im = fourier_real_im[:, ::2]
    fourier = fourier_real + fourier_im * 1j

    return fourier

# f_coeff_list = []
#
# for i in range(2, 34):
#     f_coeffs = np.reshape(np.fromfile(dir + 'Li2SO4/'+str(i)+'/fid', dtype=np.int32), (64, 128))
#     f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
#     f_coeff_list.append(f_coeffs_unpacked)
#
# f_coeff_list_Li_LS = []
#
# for i in range(2, 34):
#     f_coeffs = np.reshape(np.fromfile(dir + 'Li_LS/'+str(i)+'/fid', dtype=np.int32), (64, 128))
#     f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
#     f_coeff_list_Li_LS.append(f_coeffs_unpacked)

extensions = ['', '_Li_LS']
save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/Experiments/MRI_birmingham/Results_MRI_dTV'

if plot_TV_results:

    for k, ext in enumerate(extensions):

        GT_norms_dict = {}
        norms_dict = {}
        stdevs = {}

        for j, avg in enumerate(avgs):

            GT_norms_dict['avgs=' + avg] = {}
            norms_dict['avgs='+ avg] = {}
            stdevs['avgs=' + avg] = {}

            f_coeff_list = []

            with open(save_dir + '/Robustness_31112020_TV_' + avg + ext + '_new.json') as f:
                d = json.load(f)

            print("read avgs" + avg)

            if k==0:
                # getting the data
                for i in range(2, 34):
                    f_coeffs = np.reshape(np.fromfile(dir + 'Li2SO4/' + str(i) + '/fid', dtype=np.int32), (64, 128))
                    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
                    f_coeff_list.append(f_coeffs_unpacked)

                #coeffs = f_coeff_list

            if k==1:

                for i in range(2, 34):
                    f_coeffs = np.reshape(np.fromfile(dir + 'Li_LS/' + str(i) + '/fid', dtype=np.int32), (64, 128))
                    f_coeffs_unpacked = unpacking_fourier_coeffs(f_coeffs)
                    f_coeff_list.append(f_coeffs_unpacked)

            f_coeff_arr = np.asarray(f_coeff_list)
            fully_averaged_coeffs = np.average(f_coeff_arr, axis=0)
            f_coeff_list_grouped = []
            num = int(2 ** j)
            for i in range(num):
                data_arr = np.roll(f_coeff_arr, i, axis=0)
                for ele in range(len(f_coeff_list) // num):
                    f_coeff_list_grouped.append(np.sum(data_arr[num * ele:num * (ele + 1)], axis=0) / num)

            coeffs = f_coeff_list_grouped

            # all the recons for each num of avgs for each reg parameter, in separate plots
            for output_dim in output_dims:

                GT_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
                norms_dict['avgs='+ avg]['output_dim=' + str(output_dim)] = {}
                stdevs['avgs=' + avg]['output_dim=' + str(output_dim)] = {}

                complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                                  shape=[output_dim, output_dim], dtype='complex')
                image_space = complex_space.real_space ** 2
                forward_op = RealFourierTransform(image_space)

                l2_norm = odl.solvers.L2Norm(forward_op.range)

                for reg_param in reg_params:

                    GT_diff_norms = []
                    diff_norms = []
                    recons = []

                    fig, axs = plt.subplots(16, 4, figsize=(4, 10))
                    for i in range(32):

                        recon = np.asarray(d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]
                                           ['output_size=' + str(output_dim)]).astype('float64')
                        #axs[i//4, i % 4].imshow(image, cmap=plt.cm.gray)
                        #axs[i//4, i % 4].axis("off")

                        # stupidly, my code (see "processing") is still rotating the reconstruction, so I have to correct here
                        recon_rotated = np.asarray([recon[0].T[:, ::-1], recon[1].T[:, ::-1]])
                        image = np.abs(recon_rotated[0] + 1j * recon_rotated[1])

                        data = np.zeros((output_dim, output_dim), dtype='complex')
                        data[output_dim // 2 - 16:output_dim // 2 + 16, output_dim // 2 - 16:output_dim // 2 + 16] = coeffs[i]
                        data = np.fft.fftshift(data)

                        fully_averaged_data = np.zeros((output_dim, output_dim), dtype='complex')
                        fully_averaged_data[output_dim // 2 - 16:output_dim // 2 + 16, output_dim // 2 - 16:output_dim // 2 + 16] = \
                        fully_averaged_coeffs
                        fully_averaged_data = np.fft.fftshift(fully_averaged_data)

                        subsampling_matrix = np.zeros((output_dim, output_dim))
                        subsampling_matrix[output_dim // 2 - 16:output_dim // 2 + 16,
                        output_dim // 2 - 16:output_dim // 2 + 16] = 1
                        subsampling_matrix = np.fft.fftshift(subsampling_matrix)

                        synth_data = np.asarray([subsampling_matrix, subsampling_matrix])*forward_op(forward_op.domain.element([recon_rotated[0], recon_rotated[1]]))
                        diff = synth_data - forward_op.range.element([np.real(data), np.imag(data)])
                        diff_norm = l2_norm(diff)
                        diff_norms.append(diff_norm)

                        GT_diff = synth_data - forward_op.range.element([np.real(fully_averaged_data), np.imag(fully_averaged_data)])
                        GT_diff_norm = l2_norm(GT_diff)
                        GT_diff_norms.append(GT_diff_norm)
                        #diff = diff[0].asarray() + 1j * diff[1].asarray()
                        #diff_shift = np.fft.ifftshift(diff)probability

                        axs[2*(i // 4), i % 4].imshow(image, cmap=plt.cm.gray)
                        axs[2*(i // 4), i % 4].axis("off")

                        axs[1+2 * (i // 4), i % 4].imshow(np.fft.fftshift(np.abs(diff.asarray()[0] + 1j*diff.asarray()[1])), cmap=plt.cm.gray)
                        axs[1+2 * (i // 4), i % 4].axis("off")

                        recons.append(image)

                    fig.tight_layout(w_pad=0.4, h_pad=0.4)
                    plt.savefig(save_dir + "/New/results" + ext + "/TV_results" + ext + "/" + avg +"_avgs/" + str(output_dim) + "/TV_31112020_data_" + avg + "_avgs_32_to_" + str(
                        output_dim) + "reg_param_" + '{:.1e}'.format(reg_param) + ext + "_new.pdf")
                    plt.close()

                    norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                        'reg_param=' + '{:.1e}'.format(reg_param)] = diff_norms

                    GT_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                        'reg_param=' + '{:.1e}'.format(reg_param)] = GT_diff_norms

                    # np.save("7Li_1H_MRI_Data_31112020/norms_"+ avg + "_avgs_32_to_" + str(
                    #     output_dim) + "reg_param_" + '{:.1e}'.format(reg_param) + ext, diff_norms)

                    stdev = np.sqrt(np.sum(np.square(np.std(recons, axis=0))))
                    stdevs['avgs=' + avg]['output_dim=' + str(output_dim)][
                        'reg_param=' + '{:.1e}'.format(reg_param)] = stdev

                    plt.figure()
                    plt.imshow(np.std(recons, axis=0), cmap=plt.cm.gray)
                    plt.colorbar()
                    plt.savefig(save_dir + "/New/results" + ext + "/TV_results" + ext + "/" + avg +"_avgs/" + str(output_dim) + "/TV_31112020_data_" + avg + "_avgs_32_to_" + str(
                        output_dim) + "reg_param_" + '{:.1e}'.format(reg_param)+'stdev_plot_' + ext + "_new.pdf")
                    plt.close()

                    plt.figure()
                    plt.hist(np.ndarray.flatten(np.std(recons, axis=0)), bins=40)
                    plt.savefig(save_dir + "/New/results" + ext + "/TV_results" + ext + "/" + avg +"_avgs/" + str(output_dim) + "/TV_31112020_data_" + avg + "_avgs_32_to_" + str(
                        output_dim) + "reg_param_" + '{:.1e}'.format(reg_param) + 'stdev_hist_' + ext + "_new.pdf")
                    plt.close()

        json.dump(norms_dict,
                  open(save_dir + '/Robustness_31112020_TV_fidelities_' + ext + '_new.json', 'w'))

        json.dump(GT_norms_dict,
                  open(save_dir + '/Robustness_31112020_TV_GT_fidelities_' + ext + '_new.json', 'w'))

        json.dump(stdevs,
                  open(save_dir + '/Robustness_31112020_TV_aggregated_pixel_stds' + ext + '_new.json', 'w'))

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


# plotting data discrepancies

if discrepancy_plots:

    # with open('/Users/jlw31/Desktop/Robustness_results/Li_LS_TV_results/Robustness_31112020_TV_fidelities__Li_LS.json') as f:
    #     d = json.load(f)

    with open('/Users/jlw31/Desktop/Robustness_results/Li2SO4_results/Li2SO4_TV_results/Robustness_31112020_TV_fidelities_.json') as f:
        d = json.load(f)

    with open('/Users/jlw31/Desktop/Robustness_results/Li2SO4_results/Li2SO4_TV_results/Robustness_31112020_TV_GT_fidelities_.json') as f:
        D = json.load(f)

    for k, avg in enumerate(avgs):

        GT_discrep_arr = np.zeros((len(reg_params), 32))
        discrep_arr = np.zeros((len(reg_params), 32))
        d3 = d['avgs='+avg]['output_dim=64']
        D3 = D['avgs=' + avg]['output_dim=64']

        for i, reg_param in enumerate(reg_params):

            discrep = np.asarray(d3['reg_param='+'{:.1e}'.format(reg_param)]).astype('float64')
            discrep_arr[i, :] = discrep

            GT_discrep = np.asarray(D3['reg_param=' + '{:.1e}'.format(reg_param)]).astype('float64')
            GT_discrep_arr[i, :] = GT_discrep


        # plt.errorbar(np.log10(np.asarray(reg_params)), np.average(discrep_arr, axis=1), yerr=np.std(discrep_arr, axis=1),
        #              label=avg+'avgs', color="C"+str(k%10))
        # plt.plot(np.log10(np.asarray(reg_params))[:10], 63000*np.ones(10)/np.sqrt(2)**k, color="C"+str(k%10), linestyle=":")
        # plt.legend()


        plt.errorbar(np.log10(np.asarray(reg_params)), np.average(GT_discrep_arr, axis=1),
                     yerr=np.std(GT_discrep_arr, axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        plt.plot(np.log10(np.asarray(reg_params))[:10], 63000 * np.ones(10) / np.sqrt(2) ** k, color="C" + str(k % 10),
                 linestyle=":")
        plt.legend()

# stdev plots

    with open('/Users/jlw31/Desktop/Robustness_results/Li2SO4_results/Li2SO4_TV_results/Robustness_31112020_TV_aggregated_pixel_stds.json') as f:
        d = json.load(f)

    for k, avg in enumerate(avgs):

        stdev_arr = np.zeros(len(reg_params))
        d3 = d['avgs='+avg]['output_dim=64']

        for i, reg_param in enumerate(reg_params):

            stdev = d3['reg_param='+'{:.1e}'.format(reg_param)]
            stdev_arr[i] = stdev

        plt.plot(np.log10(reg_params), np.log10(stdev_arr), label=avg+'avgs', color="C"+str(k%10))
        plt.xlabel("log10(lambda)")
        plt.ylabel("recon. standard deviation")
        plt.legend()


## dTV results

alphas = [50, 10**2, 5*10**2, 10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6]

if plot_dTV_results:

    norms_dict = {}
    GT_norms_dict = {}
    stdevs = {}

    for avg in avgs:
        norms_dict['avgs=' + avg] = {}
        GT_norms_dict['avgs=' + avg] = {}
        stdevs['avgs=' + avg] = {}

        with open('Results_MRI_dTV/Robustness_31112020_dTV_' + avg + '.json') as f:
            d = json.load(f)

            for output_dim in output_dims:
                GT_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
                norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
                stdevs['avgs=' + avg]['output_dim=' + str(output_dim)] = {}

                for alpha in alphas:
                    GT_diff_norms = []
                    diff_norms = []
                    recons = []

                    fig, axs = plt.subplots(16, 4, figsize=(4, 10))
                    for i in range(32):

                        recon = np.asarray(d['measurement=' + str(i)]['output_size=' + str(output_dim)][
                            'alpha=' + '{:.1e}'.format(alpha)]['recon']).astype('float64')

                        fourier_diff = np.asarray(d['measurement=' + str(i)]['output_size=' + str(output_dim)][
                            'alpha=' + '{:.1e}'.format(alpha)]['fourier_diff']).astype('float64')

                        recon_image = np.abs(recon[0] + 1j * recon[1])
                        fourier_diff_image = np.abs(fourier_diff[0] + 1j*fourier_diff[1])

                        axs[2 * (i // 4), i % 4].imshow(recon_image, cmap=plt.cm.gray)
                        axs[2 * (i // 4), i % 4].axis("off")

                        axs[1 + 2 * (i // 4), i % 4].imshow(fourier_diff_image, cmap=plt.cm.gray)
                        axs[1 + 2 * (i // 4), i % 4].axis("off")

                        diff_norms.append(np.sqrt(np.sum(np.square(fourier_diff_image))))
                        recons.append(recon_image)

                    fig.tight_layout(w_pad=0.4, h_pad=0.4)
                    plt.savefig("7Li_1H_MRI_Data_31112020/dTV_31112020_data_" + avg + "_avgs_32_to_" + str(
                        output_dim) + "_reg_param_" + '{:.1e}'.format(alpha) + ".pdf")

                    norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                        'reg_param=' + '{:.1e}'.format(alpha)] = diff_norms

                    stdev = np.sqrt(np.sum(np.square(np.std(recons, axis=0))))
                    stdevs['avgs=' + avg]['output_dim=' + str(output_dim)]['reg_param=' + '{:.1e}'.format(alpha)] = stdev

                    plt.figure()
                    plt.imshow(np.std(recons, axis=0), cmap=plt.cm.gray)
                    plt.colorbar()
                    plt.savefig("7Li_1H_MRI_Data_31112020/stdev_plots/dTV_31112020_data_" + avg + "_avgs_32_to_" + str(
                        output_dim) + "reg_param_" + '{:.1e}'.format(alpha) + 'stdev_plot.pdf')
                    plt.close()

    json.dump(norms_dict,
              open('7Li_1H_MRI_Data_31112020/Robustness_31112020_dTV_fidelities.json', 'w'))

    json.dump(stdevs,
              open('7Li_1H_MRI_Data_31112020/Robustness_31112020_dTV_aggregated_pixel_stds.json', 'w'))

# plotting data discrepancies

if dTV_discrepancy_plots:

    with open('/Users/jlw31/Desktop/Robustness_results/Li2SO4_dTV_results/Robustness_31112020_dTV_fidelities.json') as f:
        d = json.load(f)

    for k, avg in enumerate(avgs):

        discrep_arr = np.zeros((len(alphas), 32))
        d3 = d['avgs='+avg]['output_dim=32']

        for i, alpha in enumerate(alphas):

            discrep = np.asarray(d3['reg_param='+'{:.1e}'.format(alpha)]).astype('float64')
            discrep_arr[i, :] = discrep

        plt.errorbar(np.log10(np.asarray(alphas)), np.average(discrep_arr, axis=1), yerr=np.std(discrep_arr, axis=1),
                     label=avg+'avgs', color="C"+str(k%10))
        plt.plot(np.log10(np.asarray(alphas)), 63000*np.ones(10)/np.sqrt(2)**k, color="C"+str(k%10), linestyle=":")
        plt.legend()


    with open('/Users/jlw31/Desktop/Robustness_results/Li2SO4_results/Li2SO4_dTV_results/Robustness_31112020_dTV_aggregated_pixel_stds.json') as f:
        d = json.load(f)

    for k, avg in enumerate(avgs):

        stdev_arr = np.zeros(len(alphas))
        d3 = d['avgs='+avg]['output_dim=64']

        for i, alpha in enumerate(alphas):

            stdev = d3['reg_param='+'{:.1e}'.format(alpha)]
            stdev_arr[i] = stdev

        plt.plot(np.log10(alphas), stdev_arr, label=avg+'avgs', color="C"+str(k%10))
        plt.xlabel("log10(alpha)")
        plt.ylabel("recon. standard deviation")
        plt.legend()

