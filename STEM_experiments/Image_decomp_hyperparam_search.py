# created 04/01/21. Based on "Image_decomposition.py" script

import odl
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.restoration import (denoise_wavelet, estimate_sigma)

im = np.asarray(Image.open('STEM_experiments/9565_NMC811_after_second_delithiation.tif'), dtype=float)
image = im[1000:1750, 750:1500]
height, width = image.shape

# estimating noise level
noisy_background_patch = im[1700:, 1700:]
sigma_est = estimate_sigma(noisy_background_patch, multichannel=False, average_sigmas=False)

# setting up spaces
image_space = odl.uniform_discr(min_pt=[0., 0.], max_pt=[float(height), float(width)],
                                            shape=[height, width], dtype='float')

data = image_space.element(image)
noisy_data = data

# TV version
grad = odl.Gradient(image_space)
tangent_space = grad.range
div = odl.Divergence(tangent_space)

data_fidelity = odl.solvers.L2NormSquared(image_space).translated(noisy_data)
TV_semi_norm = odl.solvers.GroupL1Norm(tangent_space) # bad name!!!

reg_params_1 = np.logspace(np.log10(0.001), np.log10(1000.), num=10)
#reg_params_1 = [10.]
reg_params_2 = np.logspace(np.log10(0.001), np.log10(1000.), num=10)

niter = 300

# cb = (odl.solvers.CallbackPrintIteration(end=', ') &
#                   odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
#                   odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
#                   odl.solvers.CallbackShow(step=5))

exp = 0
losses = {}

for reg_param_1 in reg_params_1:
    losses['reg_param_1='+'{:.1e}'.format(reg_param_1)] = {}
    geometric_parts = []
    texture_parts = []
    denoised_texture_parts = []

    for reg_param_2 in reg_params_2:
        losses['reg_param_1=' + '{:.1e}'.format(reg_param_1)]['reg_param_2=' + '{:.1e}'.format(reg_param_2)] = {}

        print("Experiment_"+str(exp))
        exp += 1

        f = odl.solvers.SeparableSum(odl.solvers.ZeroFunctional(image_space), reg_param_2*odl.solvers.L2Norm(tangent_space))
        g = odl.solvers.SeparableSum(data_fidelity, reg_param_1*TV_semi_norm)
        L = odl.operator.pspace_ops.ProductSpaceOperator([[odl.IdentityOperator(image_space), div],
                                                          [grad, odl.ZeroOperator(tangent_space)]])

        op_norm = 1.1 * odl.power_method_opnorm(L)
        tau = 1.0 / op_norm  # Step size for the primal variable
        sigma = 1.0 / op_norm

        # minimises f(x)+g(Lx)
        x = L.domain.zero()
        odl.solvers.pdhg(x, f, g, L, niter=niter, tau=tau, callback=None, sigma=sigma)

        geometric_part = x[0]
        texture_part = div(x[1])

        # denoising texture part
        denoised_texture_part_arr = denoise_wavelet(texture_part.asarray(), multichannel=False, convert2ycbcr=False,
                                        method='VisuShrink', mode='soft',
                                        sigma=sigma_est, rescale_sigma=True)

        denoised_texture_part = image_space.element(denoised_texture_part_arr)

        TV_of_geom_part = TV_semi_norm(grad(geometric_part))
        TV_of_text_part = TV_semi_norm(grad(texture_part))
        data_fid = data_fidelity(geometric_part + texture_part)
        data_fid_denoised = data_fidelity(geometric_part + denoised_texture_part)

        geometric_parts.append(geometric_part.asarray())
        texture_parts.append(texture_part.asarray())
        denoised_texture_parts.append(denoised_texture_part_arr)

        losses['reg_param_1=' + '{:.1e}'.format(reg_param_1)]['reg_param_2=' + '{:.1e}'.format(reg_param_2)][
            'TV_of_geom'] = TV_of_geom_part
        losses['reg_param_1=' + '{:.1e}'.format(reg_param_1)]['reg_param_2=' + '{:.1e}'.format(reg_param_2)][
            'TV_of_text'] = TV_of_text_part
        losses['reg_param_1=' + '{:.1e}'.format(reg_param_1)]['reg_param_2=' + '{:.1e}'.format(reg_param_2)][
            'data_fid'] = data_fid
        losses['reg_param_1=' + '{:.1e}'.format(reg_param_1)]['reg_param_2=' + '{:.1e}'.format(reg_param_2)][
            'data_fid_denoised'] = data_fid_denoised

    plt.figure()
    fig, axs = plt.subplots(10, 4, figsize=(4, 10))

    for i in np.arange(len(reg_params_2)):
        geom = geometric_parts[i]
        text = texture_parts[i]
        text_denoised = denoised_texture_parts[i]

        axs[i, 0].imshow(geom, cmap=plt.cm.gray)
        axs[i, 0].axis("off")
        axs[i, 1].imshow(text, cmap=plt.cm.gray)
        axs[i, 1].axis("off")
        axs[i, 2].imshow(text_denoised, cmap=plt.cm.gray)
        axs[i, 2].axis("off")
        axs[i, 3].imshow(geom + text_denoised, cmap=plt.cm.gray)
        axs[i, 3].axis("off")

    fig.tight_layout(w_pad=0.4, h_pad=0.4)
    plt.savefig("/Users/jlw31/Desktop/STEM_decomposition_results/decomp_" + "reg_param_1_" + '{:.1e}'.format(reg_param_1) + ".pdf")
    plt.close()

json.dump(losses, open('/Users/jlw31/Desktop/STEM_decomposition_results/image_decomp_losses.json', 'w'))
