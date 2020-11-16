import numpy as np
import matplotlib.pyplot as plt
import json
import dTV.myOperators as ops
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
from processing import *
from Utils import *
import dTV.myDeform.linearized as defs

dir = 'dTV/7LI_1H_MRI_Data_22102020/'

image_H = np.reshape(np.fromfile(dir+'1mm_1H_high_res/2dseq', dtype=np.uint16), (128, 128))
plt.figure()
plt.imshow(image_H, cmap=plt.cm.gray)

image_Li = np.reshape(np.fromfile(dir+'1mm_7Li_512_avgs/2dseq', dtype=np.uint16), (32, 32))
plt.figure()
plt.imshow(image_Li, cmap=plt.cm.gray)

height, width = image_H.shape
image_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height, width], dtype='float')
image_H_odl = image_space.element(image_H)

# downsampling
coarse_image_space_1 = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height//2, width//2], dtype='float')
coarse_image_space_2 = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height//4, width//4], dtype='float')
subsampling_op_1 = ops.Subsampling(image_space, coarse_image_space_1)
subsampling_op_2 = ops.Subsampling(image_space, coarse_image_space_2)
subsampling_op_3 = ops.Subsampling(coarse_image_space_1, coarse_image_space_2)
downsampled_image_H_1_odl = subsampling_op_1(image_H_odl)
downsampled_image_H_2_odl = subsampling_op_2(image_H_odl)
downsampled_image_H_1 = downsampled_image_H_1_odl.asarray()
downsampled_image_H_2 = downsampled_image_H_2_odl.asarray()

image_H_odl.show()
downsampled_image_H_1_odl.show()
downsampled_image_H_2_odl.show()

image_Li_odl = coarse_image_space_2.element(image_Li)

subsampling_ops = [odl.operator.default_ops.IdentityOperator(coarse_image_space_2), subsampling_op_3, subsampling_op_2]
sinfos = [downsampled_image_H_2_odl, downsampled_image_H_1_odl, image_H_odl]

# running dTV
dTV_recon = True

# running dTV
if dTV_recon:

    gamma = 0.995
    strong_cvx = 1e-5
    niter_prox = 20
    niter = 2000

    alphas = [50, 10**2, 5*10**2, 10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6]
    etas = [10.**(-i) for i in np.arange(6)]
    #alphas = [10**4]
    #etas = [10**(-5)]

    dTV_regularised_recons = {}
    pixels = ['32', '64', '128']
    for i in range(len(sinfos)):
        dTV_regularised_recons['32_to_'+pixels[i]] = {}

        # Create the forward operator
        forward_op = subsampling_ops[i]

        data_odl = image_Li_odl
        sinfo = sinfos[i]

        Yaff = odl.tensor_space(6)

        # space of optimised variables
        X = odl.ProductSpace(forward_op.domain, Yaff)

        # Set some parameters and the general TV prox options
        prox_options = {}
        prox_options['name'] = 'FGP'
        prox_options['warmstart'] = True
        prox_options['p'] = None
        prox_options['tol'] = None
        prox_options['niter'] = niter_prox

        reg_affine = odl.solvers.ZeroFunctional(Yaff)
        # x0 = X.zero()
        x0 = X.element([forward_op.adjoint(data_odl), X[1].zero()])

        f = fctls.DataFitL2Disp(X, data_odl, forward_op)

        for alpha in alphas:
            dTV_regularised_recons['32_to_'+pixels[i]]['alpha=' + '{:.1e}'.format(alpha)] = {}
            for eta in etas:

                reg_im = fctls.directionalTotalVariationNonnegative(forward_op.domain, alpha=alpha, sinfo=sinfo,
                                                                    gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                                    prox_options=prox_options)

                g = odl.solvers.SeparableSum(reg_im, reg_affine)

                cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
                      odl.solvers.CallbackShow(step=10))

                L = [1, 1e+2]
                ud_vars = [0, 1]

                # %%
                palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
                palm.run(niter)

                recon = palm.x[0].asarray()
                affine_params = palm.x[1].asarray()

                dTV_regularised_recons['32_to_'+pixels[i]]['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)] \
                    = [recon.tolist(), affine_params.tolist()]

    json.dump(dTV_regularised_recons, open('dTV/dTV_regularised_SR_512_avgs_22102020.json', 'w'))

with open('dTV/Results_MRI_dTV/dTV_regularised_SR_512_avgs_22102020.json') as f:
    d = json.load(f)

dir = '/Users/jlw31/Desktop/Presentations:Reports/dTV results/Applications_of_dTV'

loss_ratios = np.zeros((len(pixels), len(alphas), len(etas)))

for k, pixel_num in enumerate(pixels):
    d2 = d['32_to_'+pixel_num]

    # redefining relevant functionals
    forward_op = subsampling_ops[k]
    sinfo = sinfos[k]
    Yaff = odl.tensor_space(6)
    X = odl.ProductSpace(forward_op.domain, Yaff)
    f = fctls.DataFitL2Disp(X, data_odl, forward_op)

    fig, axs = plt.subplots(10, 6, figsize=(6, 10))

    for j, eta in enumerate(etas):

        reg_im_unit = fctls.directionalTotalVariationNonnegative(forward_op.domain, alpha=1, sinfo=sinfo,
                                                                 gamma=gamma, eta=eta, NonNeg=True,
                                                                 strong_convexity=strong_cvx,
                                                                 prox_options=prox_options)

        for i, alpha in enumerate(alphas):
            # dTV_regularised_recons['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)]

            recon = np.asarray(d2['alpha=' + '{:.1e}'.format(alpha)]['eta=' + '{:.1e}'.format(eta)][0])

            dTV_loss = reg_im_unit(recon)
            x = X.element([recon, X[1].zero()])
            data_loss = f(x)
            #x2 = X.element([sinfo, X[1].zero()])
            #data_loss_2 = f(x2)

            #print(data_loss/data_loss_2)

            loss_ratios[k, i, j] = data_loss/dTV_loss

            axs[i, j].imshow(recon.T[::-1, :], cmap=plt.cm.gray)
            axs[i, j].axis("off")

    fig.tight_layout(w_pad=0.4, h_pad=0.4)
    plt.savefig(dir+"/SR_22102020_data_512_avgs_32_to_"+pixel_num+".pdf")
