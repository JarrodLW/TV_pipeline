# Created 8/04/2021. Based on "Dynamic_IP_second_attempt.py"
import odl
from Utils import *
from dTV.Dynamic_MRI.Real_Fourier_op_stacked import RealFourierTransformStacked
import matplotlib.pyplot as plt
from myOperators import RealFourierTransform
from skimage.transform import resize
import imageio
from dTV.myDeform import LinDeformFixedDisp
#from dTV.myOperators import Embedding_Affine
import scipy as sp
from skimage.transform import PiecewiseAffineTransform, warp

# phantom1 = imageio.imread('dTV/MRI_15032021/Data_15032021/Phantom_data/Phantom_circle_resolution1.png')
# template = np.zeros((128, 128))
# template[64-50: 64+50, 64-50: 64+50] = resize(phantom1, (100, 100))
#
# plt.figure()
# plt.imshow(template, cmap=plt.cm.gray)
# plt.axis("off")

frame_num = 15
height = 256
width = 256

rec_space = odl.uniform_discr(
    min_pt=[-1., -1.], max_pt=[1., 1.], shape=[height, width],
    dtype='float32')

# constructing a deformation field

background= 0.0
val0=0.5
val1=5.
val2=1.5
val3=4.
val4=2.5
val5=3
val6=3.5

ellipsoids0 = np.asarray([[4.00, .9200, .9200, 0.0000, 0.0000, 0],
            [-3., .874, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .20, 0.350, -0.1000, 18],
            [val1, .0500, .3100, -.5200, -0.15000, 10],
            [val2, .2100, .2500, 0.0000, 0.3500, 0],
            [val3, .0460, .0460, 0.0000, -.1000, 0],
            [val4, .0460, .0230, -.1800, -.6050, 0],
            [val5, .0230, .0460, 0.0600, -.6050, 0],
            [val6, .0260, .0260, 0.400, .4050, 0]])

I0 = odl.phantom.ellipsoid_phantom(rec_space, ellipsoids0) + background
I0.show()

mean_intensity = np.random.normal(size=7, scale=0.01)
x_momenta = np.random.normal(size=7, scale=0.005)
y_momenta = np.random.normal(size=7, scale=0.005)
angular_momenta = np.random.normal(size=7, scale=0.05)

mean_ellipsoid_params = np.zeros((6, 7))
mean_ellipsoid_params[0] = mean_intensity
mean_ellipsoid_params[3] = x_momenta
mean_ellipsoid_params[4] = y_momenta
mean_ellipsoid_params[5] = angular_momenta
mean_ellipsoid_params = mean_ellipsoid_params.T

ellipsoids = ellipsoids0
im_array = np.zeros((frame_num, height, width))
im_array[0] = I0.asarray()

for i in range(frame_num-1):
    random_array = np.random.normal(loc=mean_ellipsoid_params, scale=0.005, size=(7, 6))
    #random_array = np.random.normal(loc=0, scale=0.01, size=(7, 6))
    pert = np.zeros((9, 6))
    pert[2:] = random_array
    ellipsoids += pert

    I = odl.phantom.ellipsoid_phantom(rec_space, ellipsoids) + background
    I.show()

    im_array[i+1] = I.asarray()

np.save('dTV/Dynamic_MRI/dynamic_phantom_images.npy', im_array)
