import numpy as np
import json
import matplotlib.pyplot as plt

with open('dTV/Results_MRI_dTV/Robustness_31112020_TV_512.json') as f:
    d = json.load(f)

reg_params = np.logspace(np.log10(2e3), np.log10(1e5), num=20)
output_dims = [int(32), int(64)]

for output_dim in output_dims:
    for reg_param in enumerate(reg_params):
        fig, axs = plt.subplots(8, 4, figsize=(5, 4))
        for i in np.range(32):

            recon = np.asarray(d['measurement=' + str(i)]['reg_param=' + '{:.1e}'.format(reg_param)]
                               ['output_size=' + str(output_dim)]).astype('float64')
            image = np.abs(recon[0] + 1j*recon[1])

            #image_rotated_flipped = image[:, ::-1].T[:, ::-1]

            axs[i//5, i % 5].imshow(image, cmap=plt.cm.gray)
            axs[i//5, i % 5].axis("off")

        fig.tight_layout(w_pad=0.4, h_pad=0.4)
        plt.savefig("dTV/7Li_1H_MRI_Data_31112020/TV_31112020_data_512_avgs_32_to_" + str(
            output_dim) + "reg_param" + str(reg_param) + ".pdf")
