# created 27/04/2021. Here I want to perform TV and dTV-upsampled reconstructions of the 27/04/2021 CT dataset.
# The forward operator is just a subsampling operator

from PIL import Image
import numpy as np
import odl
import matplotlib.pyplot as plt
from dTV.Ptycho_XRF_project.misc import TotalVariationNonNegative, Embedding, EmbeddingAdjoint
import dTV.myFunctionals as fctls

ptycho_image = np.load('dTV/CT_data/Ptycho_XRF_27042021/Ptycho.npy')
XRF_image = np.load('dTV/CT_data/Ptycho_XRF_27042021/XRF_Ca_Ka.npy')

fit_type = 'L2sq'
#fit_type = 'KL'
reg_type = 'TVNN'
reg_param = 0.018
upsample_factor = 1.

## (d)TV denoising

height = int(upsample_factor*XRF_image.shape[0])
width = int(upsample_factor*XRF_image.shape[1])
image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')

im = Image.fromarray(ptycho_image)
ptycho_image_resized = np.array(im.resize((width, height)))

data_height, data_width = XRF_image.shape
data_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[data_height, data_width], dtype='float')

# grabbing data
XRF_image_odl = data_space.element(XRF_image)
ptycho_image_odl = image_space.element(ptycho_image_resized)

XRF_image_odl.show()
ptycho_image_odl.show()

# checking registration - not sure yet how to do this
# XRF_image_odl_normalised = XRF_image_odl/np.amax(XRF_image_odl)
# ptycho_image_odl_normalised = (ptycho_image_odl - np.amin(ptycho_image_odl))/np.amax(ptycho_image_odl - np.amin(ptycho_image_odl))

# XRF_image_odl_normalised.show()
# ptycho_image_odl_normalised.show()
# (XRF_image_odl_normalised - ptycho_image_odl_normalised).show()

# forward op
#forward_op = odl.IdentityOperator(image_space)
if upsample_factor == 1:
    forward_op = odl.IdentityOperator(image_space)

else:
    vert_ind = np.sort(list(int(upsample_factor)*np.arange(data_height))*int(data_width))
    horiz_ind = list(int(upsample_factor)*np.arange(data_width))*int(data_height)
    sampling_points = [vert_ind, horiz_ind]
    emb = Embedding(data_space, image_space, sampling_points=sampling_points, adjoint=None)
    forward_op = emb.adjoint
    #forward_op = EmbeddingAdjoint(image_space, data_space, sampling_points, emb)

data = XRF_image_odl
#data = odl.phantom.noise.white_noise(image_space)**2 # a sanity check that we're not massively overfitting

if fit_type=='L2sq':
    datafit = odl.solvers.L2NormSquared(data_space).translated(data)

elif fit_type=='L1':
    datafit = odl.solvers.L1Norm(data_space).translated(data)

elif fit_type=='KL':
    datafit = odl.solvers.functional.default_functionals.KullbackLeibler(data_space, prior=data)

if reg_type=='TVNN':
    reg = reg_param*TotalVariationNonNegative(image_space)

elif reg_type=='dTVNN':
    #eta = 0.01
    eta = 0.001
    gamma = 0.9995
    #gamma = 0.995
    niter_prox = 20
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = niter_prox
    reg = fctls.directionalTotalVariationNonnegative(image_space, alpha=reg_param, sinfo=ptycho_image_odl,
                                                        gamma=gamma, eta=eta, NonNeg=True,
                                                        prox_options=prox_options)

op_norm = 1.1 * odl.power_method_opnorm(forward_op)
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

x = image_space.zero()
niter = 200

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
      odl.solvers.CallbackShow(step=5))

odl.solvers.pdhg(x, reg, datafit, forward_op, niter=niter, tau=tau, sigma=sigma, callback=cb)

# plt.figure()
# plt.imshow(x.asarray(), cmap=plt.cm.gray)

(x - data).show()
np.abs((x - data)).show()


#
# from dTV.Ptycho_XRF_project.misc import Embedding
# from dTV.Ptycho_XRF_project.misc import get_central_sampling_points
#
# X = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
#                                           shape=[7, 5], dtype='float')
# Y = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
#                                           shape=[28, 20], dtype='float')
#
# sampling_points = [np.sort([0, 4, 8, 12, 16, 20, 24]*5), [0, 4, 8, 12, 16]*7]
#
# emb = Embedding(X, Y, sampling_points=sampling_points, adjoint=None)
#
# A = np.reshape(np.arange(35), (7, 5))
# B = emb(A)
#
# plt.figure()
# plt.imshow(A, cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(B, cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(emb.adjoint(B), cmap=plt.cm.gray)

from scipy.stats import poisson

rv = poisson(x.asarray())
rv.pmf(data.asarray()//1)

plt.imshow(rv.pmf(data.asarray()//1), cmap=plt.cm.gray)
plt.colorbar()

plt.imshow(x.asarray(), cmap=plt.cm.gray)
