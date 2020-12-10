import numpy as np
import json
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform

#avgs = ['512', '1024', '2048', '4096', '8192']
avgs = ['512']
reg_params = np.logspace(np.log10(2e3), np.log10(1e5), num=20)
output_dims = [int(32), int(64)]

dir = '7Li_1H_MRI_Data_31112020/'

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

for avg in avgs:

    with open('Results_MRI_dTV/Robustness_31112020_TV_' + avg + '.json') as f:
        d = json.load(f)

    # all the recons for each num of avgs for each reg parameter, in separate plots
    for output_dim in output_dims:

        complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[output_dim, output_dim], dtype='complex')
        image_space = complex_space.real_space ** 2
        forward_op = RealFourierTransform(image_space)

        l2_norm = odl.solvers.L2Norm(forward_op.range)
        diff_norms = []

        for reg_param in reg_params:

            fig, axs = plt.subplots(16, 4, figsize=(4, 10))
            for i in range(32):

                recon = np.asarray(d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]
                                   ['output_size=' + str(output_dim)]).astype('float64')
                image = np.abs(recon[0] + 1j * recon[1])
                #axs[i//4, i % 4].imshow(image, cmap=plt.cm.gray)
                #axs[i//4, i % 4].axis("off")

                # stupidly, my code (see "processing") is still rotating the reconstruction, so I have to correct here
                recon_rotated = np.asarray([recon[0].T[:, ::-1], recon[1].T[:, ::-1]])

                data = np.zeros((output_dim, output_dim), dtype='complex')
                data[output_dim // 2 - 16:output_dim // 2 + 16, output_dim // 2 - 16:output_dim // 2 + 16] = f_coeff_list[i]
                data = np.fft.fftshift(data)
                subsampling_matrix = np.zeros((output_dim, output_dim))
                subsampling_matrix[output_dim // 2 - 16:output_dim // 2 + 16,
                output_dim // 2 - 16:output_dim // 2 + 16] = 1
                subsampling_matrix = np.fft.fftshift(subsampling_matrix)

                synth_data = np.asarray([subsampling_matrix, subsampling_matrix])*forward_op(forward_op.domain.element([recon_rotated[0], recon_rotated[1]]))
                diff = synth_data - forward_op.range.element([np.real(data), np.imag(data)])
                diff_norm = l2_norm(diff)
                diff_norms.append(diff_norm)
                #diff = diff[0].asarray() + 1j * diff[1].asarray()
                #diff_shift = np.fft.ifftshift(diff)

                axs[2*(i // 4), i % 4].imshow(np.fft.fftshift(np.abs(synth_data.asarray()[0] + 1j*synth_data.asarray()[1])), cmap=plt.cm.gray)
                axs[2*(i // 4), i % 4].axis("off")

                axs[1+2 * (i // 4), i % 4].imshow(np.fft.fftshift(np.abs(data)), cmap=plt.cm.gray)
                axs[1+2 * (i // 4), i % 4].axis("off")

            np.save("7Li_1H_MRI_Data_31112020/norms_"+str(output_dim), diff_norms)

            fig.tight_layout(w_pad=0.4, h_pad=0.4)
            plt.savefig("7Li_1H_MRI_Data_31112020/TV_31112020_data_" + avg + "_avgs_32_to_" + str(
                output_dim) + "reg_param_" + '{:.1e}'.format(reg_param) + ".pdf")


    # # example recons, subset of regularisation params, with K-space diffs
    # for output_dim in output_dims:
    #     for reg_param in reg_params[::4]:
    #
    #         fig, axs = plt.subplots(10, 6, figsize=(5, 4))
    #         for i in range(30):
    #
    #             recon = np.asarray(d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]
    #                                ['output_size=' + str(output_dim)]).astype('float64')
    #             image = np.abs(recon[0] + 1j * recon[1])
    #             axs[2*i//6, i % 6].imshow(image, cmap=plt.cm.gray)
    #             axs[2*i//6, i % 6].axis("off")
    #
    #             axs[1+2*i//6, i % 6].imshow(fourier_diff, cmap=plt.cm.gray)
    #             axs[1+2*i//6, i % 6].axis("off")



    print("done")




# with open('Results_MRI_dTV/Robustness_31112020_TV_2048.json') as f:
#     d = json.load(f)
#
# test_recon = np.asarray(d['measurement=' + str(2)]['reg_param=3.0e+03']
#                                        ['output_size=' + str(64)]).astype('float64')
#
# plt.figure()
# plt.imshow(np.abs(test_recon[0]+1j*test_recon[1]), cmap=plt.cm.gray)
# plt.savefig("7Li_1H_MRI_Data_31112020/test_image.pdf")
