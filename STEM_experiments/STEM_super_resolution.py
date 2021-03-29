import odl
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import myOperators as ops
from skimage.transform import resize

im = np.asarray(Image.open('STEM_experiments/9565_NMC811_after_second_delithiation.tif'), dtype=float)
height, width = im.shape
im_patch = im[height//4:height//4+150, width//4: width//4 + 150]
height_patch, width_patch = im_patch.shape

plt.figure()
plt.imshow(im, cmap=plt.cm.gray)

plt.figure()
plt.imshow(im_patch, cmap=plt.cm.gray)

fourier = np.fft.fftshift(np.fft.fft2(im_patch))

plt.figure()
plt.imshow(np.abs(fourier), cmap=plt.cm.gray)

# optimisation
reg_param = 10000000.
niter = 500

image = im_patch

height, width = image.shape

height_new = 2*height
width_new = 2*width
recon_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                shape=[height_new, width_new], dtype='float')

# image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
#                                 shape=[height, width], dtype='float')

image_space_flattened = odl.rn(height*width)

#data_odl = image_space.element(im_patch.T) # Not sure why I have to take the transpose in order for result of subsampling op to match!
data_odl = image_space_flattened.element(np.ndarray.flatten(image))

complex_embedding = odl.operator.default_ops.ComplexEmbedding(recon_space)
complex_to_real = ops.Complex2Real(complex_embedding.range)
embedding = complex_to_real*complex_embedding
fourier = ops. RealFourierTransform(recon_space ** 2)
f_transf = fourier * embedding

#resize_op = odl.ResizingOperator(image_space, ran_shp=(150, 150))
x_1 = 2*np.arange(height)
y_1 = 2*np.arange(width)
x_2 = 1 + 2*np.arange(height)
y_2 = 1 + 2*np.arange(width)
#flatten_op = odl.operator.tensor_ops.FlatteningOperator(image_space)
# sampling_points_1 = np.asarray([np.tile(x_1, len(y_1)), np.repeat(y_1, len(x_1))])
# sampling_points_2 = np.asarray([np.tile(x_1, len(y_1)), np.repeat(y_2, len(x_1))])
# sampling_points_3 = np.asarray([np.tile(x_2, len(y_1)), np.repeat(y_1, len(x_1))])
# sampling_points_4 = np.asarray([np.tile(x_2, len(y_1)), np.repeat(y_2, len(x_1))])
sampling_points_1 = np.asarray([np.repeat(y_1, len(x_1)), np.tile(x_1, len(y_1))])
sampling_points_2 = np.asarray([np.repeat(y_2, len(x_1)), np.tile(x_1, len(y_1))])
sampling_points_3 = np.asarray([np.repeat(y_1, len(x_1)), np.tile(x_2, len(y_1))])
sampling_points_4 = np.asarray([np.repeat(y_2, len(x_1)), np.tile(x_2, len(y_1))])
subsample_op_1 = odl.operator.tensor_ops.SamplingOperator(recon_space, sampling_points=sampling_points_1)
subsample_op_2 = odl.operator.tensor_ops.SamplingOperator(recon_space, sampling_points=sampling_points_2)
subsample_op_3 = odl.operator.tensor_ops.SamplingOperator(recon_space, sampling_points=sampling_points_3)
subsample_op_4 = odl.operator.tensor_ops.SamplingOperator(recon_space, sampling_points=sampling_points_4)
pooling_op = (1/4)*(subsample_op_1 + subsample_op_2 + subsample_op_3 + subsample_op_4)

op = odl.BroadcastOperator(f_transf, pooling_op)
sparsifying_norm = odl.solvers.SeparableSum(odl.solvers.L1Norm(f_transf.range[0]),
                                                      odl.solvers.L1Norm(f_transf.range[1]))
datafit = odl.solvers.L2NormSquared(image_space_flattened).translated(data_odl)
g = odl.solvers.SeparableSum(reg_param*sparsifying_norm, datafit)
f = odl.solvers.ZeroFunctional(recon_space)

op_norm = 1.1 * odl.power_method_opnorm(op)
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
      odl.solvers.CallbackShow(step=5))

x = op.domain.zero()
odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=cb)

# Testing the pooling
test_image = np.zeros((height_new, width_new))

for i in range(height):
    for j in range(width):

        test_image[2*i, 2*j] = im_patch[i, j]

pooled = pooling_op(recon_space.element(test_image))
pooled.show()
data_odl.show()
datafit(pooled)

pooled_recon = np.reshape(pooled.asarray(), (150, 150))

plt.figure()
plt.imshow(pooled_recon, cmap=plt.cm.gray)

plt.figure()
plt.imshow(im_patch, cmap=plt.cm.gray)
