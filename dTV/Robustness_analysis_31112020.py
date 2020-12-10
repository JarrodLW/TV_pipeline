import numpy as np
import json
import matplotlib.pyplot as plt

avgs = ['512', '1024', '2048', '4096', '8192']
reg_params = np.logspace(np.log10(2e3), np.log10(1e5), num=20)
output_dims = [int(32), int(64)]

for avg in enumerate(avgs):

    with open('Results_MRI_dTV/Robustness_31112020_TV_' + avg + '.json') as f:
        d = json.load(f)

    for output_dim in output_dims:
        for reg_param in reg_params:

            fig, axs = plt.subplots(8, 4, figsize=(5, 4))
            for i in range(32):

                recon = np.asarray(d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]
                                   ['output_size=' + str(output_dim)]).astype('float64')
                image = np.abs(recon[0] + 1j * recon[1])
                axs[i//4, i % 4].imshow(image, cmap=plt.cm.gray)
                axs[i//4, i % 4].axis("off")

            fig.tight_layout(w_pad=0.4, h_pad=0.4)
            plt.savefig("7Li_1H_MRI_Data_31112020/TV_31112020_data_" + avg + "_avgs_32_to_" + str(
                output_dim) + "reg_param_" + '{:.1e}'.format(reg_param) + ".pdf")

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
