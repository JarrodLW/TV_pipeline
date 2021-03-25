import numpy as np
import json
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform
from skimage.measure import block_reduce
import libpysal
import esda
from Utils import *

plot_TV_results = False
best_TV_recons = False
plot_dTV_results = True
plot_Moran = False
plot_TV_results_full_avgs = False
plot_subset_TV_results = False
discrepancy_plots = False
dTV_discrepancy_plots = False
affine_param_plots = False

avgs = ['512', '1024', '2048', '4096', '8192']
reg_params = np.concatenate((np.asarray([0.001, 1., 10**0.5, 10., 10**1.5, 10**2]), np.logspace(3., 4.5, num=20)))
#output_dims = [int(32), int(64)]
output_dims = [int(128)]

dir = 'dTV/MRI_15032021/'
extensions = ['']
save_dir = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday/Experiments/MRI_birmingham/Results_15032021/'

if plot_TV_results:

    GT_TV_data = np.load(dir + 'Results_15032021/example_TV_recon_15032021_synth_data.npy')
    GT_TV_image = np.load(dir + 'Results_15032021/example_TV_recon_15032021.npy')
    GT_TV_image = np.abs(GT_TV_image[0] + 1j*GT_TV_image[1])

    for k, ext in enumerate(extensions):

        GT_norms_dict = {}
        GT_TV_norms_dict = {}
        GT_SSIM_dict = {}
        GT_TV_SSIM_dict = {}
        norms_dict = {}
        stdevs = {}
        morans_I_dict = {}

        for j, avg in enumerate(avgs):

            GT_norms_dict['avgs=' + avg] = {}
            GT_TV_norms_dict['avgs=' + avg] = {}
            GT_SSIM_dict['avgs=' + avg] = {}
            GT_TV_SSIM_dict['avgs=' + avg] = {}
            norms_dict['avgs='+ avg] = {}
            stdevs['avgs=' + avg] = {}
            morans_I_dict['avgs=' + avg] = {}

            f_coeff_list = []

            with open(save_dir + 'TV_results/TV_7Li_15032021_'+str(avg)+'.json') as f:
                d = json.load(f)

            print("read avgs" + avg)

            if k==0:
                # getting the data
                for i in range(3, 35):
                    f_coeffs = np.reshape(np.fromfile(dir + 'Data_15032021/Li_data/' + str(i) + '/fid', dtype=np.int32), (64, 128))
                    f_coeffs_unpacked = unpacking_fourier_coeffs_15032021(f_coeffs)
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
                        diff_norms = []
                        recons = []
                        morans_I_vals = []

                        fig, axs = plt.subplots(16, 4, figsize=(4, 10))
                        for i in range(32):

                            recon = np.asarray(d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]
                                               ['output_size=' + str(output_dim)]).astype('float64')

                            image = np.abs(recon[0] + 1j*recon[1])

                            data = np.zeros((output_dim, output_dim), dtype='complex')
                            data[output_dim // 2 - 16:output_dim // 2 + 16, output_dim // 2 - 16:output_dim // 2 + 16] = coeffs[i]
                            data = np.fft.fftshift(data)
                            #data = np.fft.fftshift(coeffs[i])

                            fully_averaged_data = np.zeros((output_dim, output_dim), dtype='complex')
                            fully_averaged_data[output_dim // 2 - 16:output_dim // 2 + 16, output_dim // 2 - 16:output_dim // 2 + 16] = \
                            fully_averaged_coeffs
                            fully_averaged_data = np.fft.fftshift(fully_averaged_data)
                            #fully_averaged_data = np.fft.fftshift(fully_averaged_coeffs)

                            GT_proxy = np.zeros((output_dim, output_dim), dtype='complex')
                            GT_proxy[output_dim // 2 - 16: output_dim // 2 + 16, output_dim // 2 - 16: output_dim // 2 + 16]=\
                                np.fft.ifftshift(GT_TV_data[0]+1j*GT_TV_data[1])
                            GT_proxy = np.fft.fftshift(GT_proxy)

                            subsampling_matrix = np.zeros((output_dim, output_dim))
                            subsampling_matrix[output_dim // 2 - 16:output_dim // 2 + 16,
                            output_dim // 2 - 16:output_dim // 2 + 16] = 1
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
                        plt.savefig(save_dir + "TV_results/" + avg +"_avgs/" + str(output_dim) + "/TV_31112020_data_" + avg + "_avgs_32_to_" + str(
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

                        stdev = np.sqrt(np.sum(np.square(np.std(recons, axis=0))))
                        stdevs['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = stdev

                        morans_I_dict['avgs=' + avg]['output_dim=' + str(output_dim)][
                            'reg_param=' + '{:.1e}'.format(reg_param)] = morans_I_vals

                        np.save(save_dir + "TV_results/" + avg +"_avgs/" + str(output_dim) + "/TV_15032021_" + avg + "_avgs_32_to_" + str(
                            output_dim) + "reg_param_" + '{:.1e}'.format(reg_param)+"stdev_arr.npy", np.std(recons, axis=0))

                        plt.figure()
                        plt.imshow(np.std(recons, axis=0), cmap=plt.cm.gray)
                        plt.colorbar()
                        plt.savefig(save_dir + "TV_results/" + avg +"_avgs/" + str(output_dim) + "/TV_15032021_" + avg + "_avgs_32_to_" + str(
                            output_dim) + "reg_param_" + '{:.1e}'.format(reg_param)+"stdev_plot.pdf")
                        plt.close()

                        plt.figure()
                        plt.hist(np.ndarray.flatten(np.std(recons, axis=0)), bins=40)
                        plt.savefig(save_dir + "TV_results/" + avg +"_avgs/" + str(output_dim) + "/TV_15032021_" + avg + "_avgs_32_to_" + str(
                            output_dim) + "reg_param_" + '{:.1e}'.format(reg_param) + "stdev_hist.pdf")
                        plt.close()

            except KeyError:
                print("failed to grab key (presumably)")
                continue

        json.dump(norms_dict,
                  open(save_dir + "TV_results/" + 'TV_fidelities.json', 'w'))

        json.dump(GT_norms_dict,
                  open(save_dir + "TV_results/" + 'TV_GT_fidelities.json', 'w'))

        json.dump(GT_TV_norms_dict,
                  open(save_dir + "TV_results/" + 'TV_GT_proxy_fidelities.json', 'w'))

        json.dump(GT_SSIM_dict,
                  open(save_dir + "TV_results/" + 'TV_GT_SSIM_vals.json', 'w'))

        json.dump(GT_TV_SSIM_dict,
                  open(save_dir + "TV_results/" + 'TV_GT_proxy_SSIM_vals.json', 'w'))

        json.dump(stdevs,
                  open(save_dir + "TV_results/" + 'TV_aggregated_pixel_stds.json', 'w'))

        json.dump(morans_I_dict,
                  open(save_dir + "TV_results/" + 'TV_morans_I.json', 'w'))

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

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/TV/TV_fidelities.json') as f:
        d = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/TV/TV_GT_fidelities.json') as f:
        D = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/TV/TV_GT_proxy_fidelities.json') as f:
        DD = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/TV/TV_GT_SSIM_vals.json') as f:
        D_SSIM = json.load(f)

    f.close()

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/TV/TV_GT_proxy_SSIM_vals.json') as f:
        DD_SSIM = json.load(f)

    f.close()

    Morozov_thresholds = [101165, 70547, 48230, 31739, 18380]

    for k, avg in enumerate(avgs):
        print("unpacking average " + avg)

        GT_discrep_arr = np.zeros((len(reg_params), 32))
        GT_TV_discrep_arr = np.zeros((len(reg_params), 32))
        discrep_arr = np.zeros((len(reg_params), 32))
        GT_SSIM_arr = np.zeros((len(reg_params), 32))
        GT_proxy_SSIM_arr = np.zeros((len(reg_params), 32))
        output_dim = str(32)
        d3 = d['avgs='+avg]['output_dim='+output_dim]
        D3 = D['avgs=' + avg]['output_dim='+output_dim]
        DD3 = DD['avgs=' + avg]['output_dim=' + output_dim]
        D_SSIM3 = D_SSIM['avgs='+avg]['output_dim='+output_dim]
        DD_SSIM3 = DD_SSIM['avgs=' + avg]['output_dim=' + output_dim]

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

        # plt.errorbar(np.log10(np.asarray(reg_params))[1:], np.average(GT_discrep_arr[1:], axis=1),
        #              yerr=np.std(GT_discrep_arr[1:], axis=1),
        #              label=avg + 'avgs', color="C" + str(k % 10))
        # plt.plot(np.log10(np.asarray(reg_params))[1:], Morozov_thresholds[k] * np.ones(25), color="C" + str(k % 10),
        #          linestyle=":")
        # plt.xlabel("log(lambda)")
        # plt.ylabel("l2-discrepancy")
        # plt.title("L2-discrepancy between "+output_dim+"-by-"+output_dim+" TV-regularised recons\n and 16384-averaged data")
        # plt.legend()
        #
        # plt.errorbar(np.log10(np.asarray(reg_params))[1:], np.average(GT_TV_discrep_arr[1:], axis=1),
        #              yerr=np.std(GT_TV_discrep_arr[1:], axis=1),
        #              label=avg + 'avgs', color="C" + str(k % 10))
        # plt.plot(np.log10(np.asarray(reg_params))[1:], Morozov_thresholds[k] * np.ones(25), color="C" + str(k % 10),
        #          linestyle=":")
        # plt.xlabel("log(lambda)")
        # plt.ylabel("l2-discrepancy")
        # plt.title("L2-discrepancy between "+output_dim+"-by-"+output_dim+" TV-regularised recons\n and synthetic ground-truth proxy")
        # plt.legend()

        # plt.errorbar(np.log10(np.asarray(reg_params))[1:], np.average(discrep_arr[1:], axis=1),
        #              yerr=np.std(discrep_arr[1:], axis=1),
        #              label=avg + 'avgs', color="C" + str(k % 10))
        # plt.plot(np.log10(np.asarray(reg_params))[1:], Morozov_thresholds[k]* np.ones(25) , color="C" + str(k % 10),
        #          linestyle=":")
        # plt.xlabel("log(lambda)")
        # plt.ylabel("l2-discrepancy")
        # plt.title("L2 data discrepancy for "+output_dim+"-by-"+output_dim+" TV-regularised recons")
        # plt.legend()

        # plt.errorbar(np.log10(np.asarray(reg_params))[1:], np.average(GT_SSIM_arr[1:], axis=1),
        #              yerr=np.std(GT_SSIM_arr[1:], axis=1),
        #              label=avg + 'avgs', color="C" + str(k % 10))
        # plt.xlabel("log(lambda)")
        # plt.ylabel("SSIM")
        # plt.title("SSIM between "+output_dim+"-by-"+output_dim+" TV-regularised recons\n and 16384-averaged data")
        # plt.legend()

        plt.errorbar(np.log10(np.asarray(reg_params))[1:], np.average(GT_proxy_SSIM_arr[1:], axis=1),
                     yerr=np.std(GT_proxy_SSIM_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        plt.xlabel("log(lambda)")
        plt.ylabel("SSIM")
        plt.title(
            "SSIM between " + output_dim + "-by-" + output_dim + " TV-regularised recons\n and synthetic ground-truth proxy")
        plt.ylim(0., 1.)
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend()

# stdev plots

    with open('/Users/jlw31/Desktop/Robustness_results_new/Li2SO4_results/Li2SO4_TV_results/Robustness_31112020_TV_aggregated_pixel_stds_new.json') as f:
        d = json.load(f)

    for k, avg in enumerate(avgs):

        stdev_arr = np.zeros(len(reg_params))
        d3 = d['avgs='+avg]['output_dim=64']

        for i, reg_param in enumerate(reg_params):

            stdev = d3['reg_param='+'{:.1e}'.format(reg_param)]
            stdev_arr[i] = stdev

        plt.plot(np.log10(reg_params)[1:], stdev_arr[1:]/32, label=avg+'avgs', color="C"+str(k%10))
        plt.xlabel("log10(lambda)")
        plt.ylabel("mean-squared stdev")
        plt.title("Mean-squared standard deviation of TV-regularised \n reconstructions, 64-by-64")
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

    #GT_TV_data = np.load('Results_MRI_dTV/example_TV_recon_Li2SO4_16384_avgs_reg_param_1000_synth_data.npy')
    #GT_TV_data = np.fft.fftshift(np.load(dir + 'Results_15032021/example_TV_recon_15032021_synth_data.npy'))
    GT_TV_data_arr = np.load(dir + 'Results_15032021/example_TV_recon_15032021_synth_data.npy')
    GT_TV_data = np.fft.fftshift(GT_TV_data_arr[0] + 1j*GT_TV_data_arr[1])
    GT_TV_image = np.load(dir + 'Results_15032021/example_TV_recon_15032021.npy')
    GT_TV_image = np.abs(GT_TV_image[0] + 1j * GT_TV_image[1])

    norms_dict = {}
    GT_norms_dict = {}
    GT_TV_norms_dict = {}
    GT_SSIM_dict = {}
    GT_TV_SSIM_dict = {}
    stdevs = {}
    affine_param_dict = {}

    for j, avg in enumerate(avgs):
        norms_dict['avgs=' + avg] = {}
        GT_norms_dict['avgs=' + avg] = {}
        GT_TV_norms_dict['avgs=' + avg] = {}
        GT_SSIM_dict['avgs=' + avg] = {}
        GT_TV_SSIM_dict['avgs=' + avg] = {}
        stdevs['avgs=' + avg] = {}

        with open(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_' + avg +'_pre_registered.json') as f:
            d = json.load(f)

        print(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_' + avg +'_pre_registered.json')

        # grabbing just the affine params, and putting into new dictionary

        affine_param_dict['avgs=' + avg] = {mkey: {skey: {akey: aval['affine_params'] for akey, aval in sval.items()}
                                   for skey, sval in mval.items()} for mkey, mval in d.items()}

        # grabbing dataset again to compute GT diff

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

        f_coeff_list_grouped = []
        num = int(2 ** j)
        for i in range(num):
            data_arr = np.roll(f_coeff_arr, i, axis=0)
            for ele in range(len(f_coeff_list) // num):
                f_coeff_list_grouped.append(np.sum(data_arr[num * ele:num * (ele + 1)], axis=0) / num)

        coeffs = f_coeff_list_grouped
        coeffs_minus_GT = coeffs - fully_averaged_coeffs
        coeffs_minus_GT_TV = coeffs - GT_TV_data

        for output_dim in output_dims:
            GT_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            GT_TV_norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            GT_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            GT_TV_SSIM_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            norms_dict['avgs=' + avg]['output_dim=' + str(output_dim)] = {}
            stdevs['avgs=' + avg]['output_dim=' + str(output_dim)] = {}

            for alpha in alphas:
                GT_diff_norms = []
                GT_TV_diff_norms = []
                GT_SSIM_vals = []
                GT_TV_SSIM_vals = []
                diff_norms = []
                recons = []

                fig, axs = plt.subplots(16, 4, figsize=(4, 10))
                for i in range(32):

                    print('\noutput_dims_' + str(output_dim))
                    print('measurement_' + str(i))
                    print(d['measurement=' + str(i)]['output_size=' + str(output_dim)].keys())

                    recon = np.asarray(d['measurement=' + str(i)]['output_size=' + str(output_dim)][
                        'alpha=' + '{:.1e}'.format(alpha)]['recon']).astype('float64')

                    fourier_diff = np.asarray(d['measurement=' + str(i)]['output_size=' + str(output_dim)][
                        'alpha=' + '{:.1e}'.format(alpha)]['fourier_diff']).astype('float64')

                    print(np.shape(fourier_diff))
                    print(np.shape(coeffs_minus_GT[i, :, :]))

                    GT_fourier_diff = fourier_diff[0] + 1j*fourier_diff[1] + coeffs_minus_GT[i, :, :]
                    GT_TV_fourier_diff = fourier_diff[0] + 1j*fourier_diff[1] + coeffs_minus_GT_TV[i, :, :]

                    recon_image = np.abs(recon[0] + 1j * recon[1])
                    fourier_diff_image = np.abs(fourier_diff[0] + 1j*fourier_diff[1])

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
                    GT_SSIM = recon_error(recon_image, GT_image)[2]
                    GT_TV_SSIM = recon_error(recon_image, GT_TV_image)[2]
                    GT_SSIM_vals.append(GT_SSIM)
                    GT_TV_SSIM_vals.append(GT_TV_SSIM)

                fig.tight_layout(w_pad=0.4, h_pad=0.4)
                plt.savefig(save_dir + "dTV_results_pre_registered/" + avg +"_avgs/" + str(output_dim) +"/dTV_no_regis_31112020_data_" + avg + "_avgs_32_to_" + str(
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

                stdev = np.sqrt(np.sum(np.square(np.std(recons, axis=0))))
                stdevs['avgs=' + avg]['output_dim=' + str(output_dim)]['reg_param=' + '{:.1e}'.format(alpha)] = stdev

                plt.figure()
                plt.imshow(np.std(recons, axis=0), cmap=plt.cm.gray)
                plt.colorbar()
                plt.savefig(save_dir + "dTV_results_pre_registered/" + avg +"_avgs/" + str(output_dim) +"/dTV_no_regis_31112020_data_" + avg + "_avgs_32_to_" + str(
                    output_dim) + "reg_param_" + '{:.1e}'.format(alpha) + 'stdev_plot_new.pdf')
                plt.close()

    json.dump(norms_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_pre_registered_fidelities.json', 'w'))

    json.dump(GT_norms_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_pre_registered_GT_fidelities.json', 'w'))

    json.dump(GT_TV_norms_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_pre_registered_GT_from_TV_fidelities.json', 'w'))

    json.dump(GT_SSIM_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_pre_registered_GT_SSIM_vals.json', 'w'))

    json.dump(GT_TV_SSIM_dict,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_pre_registered_GT_proxy_SSIM_vals.json', 'w'))

    json.dump(stdevs,
              open(save_dir + 'dTV_results_pre_registered/dTV_7Li_15032021_pre_registered_aggregated_pixel_stds.json', 'w'))

    #json.dump(affine_param_dict,
     #         open(save_dir + '/New/results/dTV_results_no_regis/Robustness_31112020_dTV_affine_params_new.json', 'w'))

# plotting data discrepancies - this is done locally: need to copy above json files into local directory

if dTV_discrepancy_plots:

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/dTV_pre_registered/dTV_7Li_15032021_pre_registered_fidelities.json') as f:
        d = json.load(f)

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/dTV_pre_registered/dTV_7Li_15032021_pre_registered_GT_fidelities.json') as f:
        D = json.load(f)

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/dTV_pre_registered/dTV_7Li_15032021_pre_registered_GT_from_TV_fidelities.json') as f:
        DD = json.load(f)

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/dTV_pre_registered/dTV_7Li_15032021_pre_registered_GT_SSIM_vals.json') as f:
        D_SSIM = json.load(f)

    with open('/Users/jlw31/Desktop/Results_on_15032021_dataset/dTV_pre_registered/dTV_7Li_15032021_pre_registered_GT_proxy_SSIM_vals.json') as f:
        DD_SSIM = json.load(f)

    Morozov_thresholds = [101165, 70547, 48230, 31739, 18380]

    for k, avg in enumerate(avgs):

        discrep_arr = np.zeros((len(alphas), 32))
        GT_discrep_arr = np.zeros((len(alphas), 32))
        GT_TV_discrep_arr = np.zeros((len(alphas), 32))
        GT_SSIM_arr = np.zeros((len(alphas), 32))
        GT_proxy_SSIM_arr = np.zeros((len(alphas), 32))
        output_dim = str(32)
        d3 = d['avgs='+avg]['output_dim='+output_dim]
        D3 = D['avgs=' + avg]['output_dim=' + output_dim]
        DD3 = DD['avgs=' + avg]['output_dim=' + output_dim]
        D_SSIM3 = D_SSIM['avgs=' + avg]['output_dim=' + output_dim]
        DD_SSIM3 = DD_SSIM['avgs=' + avg]['output_dim=' + output_dim]

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

        # plt.errorbar(np.log10(np.asarray(alphas))[1:], np.average(discrep_arr, axis=1)[1:], yerr=np.std(discrep_arr[1:], axis=1),
        #              label=avg+'avgs', color="C"+str(k%10))
        # plt.plot(np.log10(np.asarray(alphas))[1:], Morozov_thresholds[k] * np.ones(25), color="C"+str(k%10), linestyle=":")
        # plt.xlabel("log(alpha)")
        # plt.ylabel("l2-discrepancy")
        # plt.title("L2 data discrepancy for " + output_dim + "-by-" + output_dim + " dTV-regularised recons")
        # plt.legend()

        # plt.errorbar(np.log10(np.asarray(alphas))[1:], np.average(GT_discrep_arr, axis=1)[1:],
        #              yerr=np.std(GT_discrep_arr[1:], axis=1),
        #              label=avg + 'avgs', color="C" + str(k % 10))
        # plt.plot(np.log10(np.asarray(alphas))[1:], Morozov_thresholds[k] * np.ones(25), color="C" + str(k % 10),
        #          linestyle=":")
        # plt.xlabel("log(alpha)")
        # plt.ylabel("l2-discrepancy")
        # plt.title("L2-discrepancy between "+output_dim+"-by-"+output_dim+" dTV-regularised recons\n and 16384-averaged data")
        # plt.legend()

        # plt.errorbar(np.log10(np.asarray(alphas))[1:], np.average(GT_TV_discrep_arr, axis=1)[1:],
        #              yerr=np.std(GT_TV_discrep_arr[1:], axis=1),
        #              label=avg + 'avgs', color="C" + str(k % 10))
        # plt.plot(np.log10(np.asarray(alphas))[1:], Morozov_thresholds[k] * np.ones(25),
        #          color="C" + str(k % 10),
        #          linestyle=":")
        # plt.xlabel("log(alpha)")
        # plt.ylabel("l2-discrepancy")
        # plt.title(
        #     "L2-discrepancy between " + output_dim + "-by-" + output_dim + " dTV-regularised recons\n and groundtruth proxy")
        # plt.legend()

        plt.errorbar(np.log10(np.asarray(alphas))[1:], np.average(GT_proxy_SSIM_arr, axis=1)[1:],
                     yerr=np.std(GT_proxy_SSIM_arr[1:], axis=1),
                     label=avg + 'avgs', color="C" + str(k % 10))
        plt.xlabel("log(alpha)")
        plt.ylabel("SSIM")
        plt.title(
            "SSIM between " + output_dim + "-by-" + output_dim + " dTV-regularised recons\n and groundtruth proxy")
        plt.ylim(0., 1.)
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend()


    with open('/Users/jlw31/Desktop/Robustness_results_new/Li2SO4_results/Li2SO4_TV_initialised_dTV_results/Robustness_31112020_TV_init_dTV_aggregated_pixel_stds_new.json') as f:
        d = json.load(f)

    for k, avg in enumerate(avgs):

        stdev_arr = np.zeros(len(alphas))
        d3 = d['avgs='+avg]['output_dim=64']

        for i, alpha in enumerate(alphas):

            stdev = d3['reg_param='+'{:.1e}'.format(alpha)]
            stdev_arr[i] = stdev

        plt.plot(np.log10(alphas)[1:], stdev_arr[1:]/32, label=avg+'avgs', color="C"+str(k%10))
        plt.xlabel("log10(lambda)")
        plt.ylabel("mean-squared stdev")
        plt.title("Mean-squared standard deviation of dTV-regularised \n reconstructions, 64-by-64")
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
