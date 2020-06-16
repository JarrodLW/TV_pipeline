import odl
import numpy as np
from skimage import io
from processing import *
import matplotlib.pyplot as plt

height = 200
width = 200

sino = np.array(io.imread('/Users/jlw31/PycharmProjects/SingleDLAppliedToCT/data_1_top_slice_sino.tif')
                , dtype=float)

data_stack = sino
data_stack = np.expand_dims(data_stack, axis=0)

image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                shape=[height, width], dtype='float32')
# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, data_stack.shape[1])
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, data_stack.shape[2])
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
# Create the forward operator
forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')

discr_phantom = odl.phantom.shepp_logan(image_space, modified=True)
raw_data_CT = forward_op(discr_phantom).asarray()
raw_data_CT_normalised = (raw_data_CT - np.amin(raw_data_CT)) / (np.amax(raw_data_CT) - np.amin(raw_data_CT))

forward_op = odl.trafos.FourierTransform(image_space)
raw_data_MRI = forward_op(discr_phantom).asarray()

# preparing some synthetic data from Paul's reconstruction
directory = '/Users/jlw31/Desktop/ProjectData/Paul_Quinn_data_subset'
rec_Paul = np.array(io.imread(directory + '/phase/rec_0050.tif'), dtype=float)
angle_partition_2 = odl.uniform_partition(-np.pi/2, -np.pi/2+2*np.pi, 231)
detector_partition_2 = odl.uniform_partition(-20, 20, 167)
geometry_2 = odl.tomo.Parallel2dGeometry(angle_partition_2, detector_partition_2)
image_space_2 = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                  shape=[rec_Paul.shape[0], rec_Paul.shape[1]], dtype='float32')
forward_op_2 = odl.tomo.RayTransform(image_space_2, geometry_2, impl='skimage')
raw_data_CT_2 = forward_op_2(rec_Paul).asarray()

# getting some of Paul's raw data
raw_data_phase = np.array(io.imread(directory + '/phase/sino_0050_cleaned.tif'), dtype=float)

from Utils import *
import matplotlib.pyplot as plt

directory = '/Users/jlw31/PycharmProjects/SRProject'

recon_horiz = np.reshape(np.fromfile(directory+'/7_2dseq', dtype=np.uint16), (128, 128))
fourier_horiz_real_im = np.reshape(np.fromfile(directory+'/7_fid', dtype=np.int32), (128, 256))
fourier_horiz_real = fourier_horiz_real_im[:, ::2]
fourier_horiz_im = fourier_horiz_real_im[:, 1::2]

fourier_horiz = fourier_horiz_real + fourier_horiz_im*1j
#my_recon_horiz = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier_horiz)))
fourier_horiz_reordered = np.fft.fftshift(fourier_horiz)

data = fourier_horiz
#data = (data - data[np.min(np.real(data))])/(data[np.max(np.real(data))] - data[np.min(np.real(data))])
#height = data.shape[0]
#width = 2 * (data.shape[1] - 1)  # is this correct?
image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                shape=[128, 128], dtype='complex')
forward_op = odl.trafos.FourierTransform(image_space, halfcomplex=False)

recon_attempt = forward_op.inverse(forward_op.range.element(data))

plt.figure()
plt.imshow(np.abs(recon_attempt), cmap=plt.cm.gray)
plt.title('abs recon attempt')
# plt.figure()
# plt.imshow(np.imag(recon_attempt), cmap=plt.cm.gray)
# plt.title('recon attempt, real part')
# plt.colorbar()
# plt.figure()
# plt.imshow(np.imag(recon_attempt), cmap=plt.cm.gray)
# plt.title('recon attempt, imag part')
# plt.colorbar()
# plt.figure()
# plt.imshow(recon_horiz, cmap=plt.cm.gray)
# plt.colorbar()


image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                shape=[128, 128], dtype='float32')

discr_phantom = odl.phantom.shepp_logan(image_space, modified=True)
forward_op = odl.trafos.FourierTransform(image_space, halfcomplex=False)
raw_data_MRI = forward_op(discr_phantom).asarray()

fourier_horiz_normalised = fourier_horiz/np.amax(np.abs(fourier_horiz))

height = fourier_horiz.shape[0]
width = fourier_horiz.shape[1]

mask = horiz_rand_walk_mask(height, width, round(0.6*height), distr='uniform', )[0]

recons_1 = regularised_recons_from_subsampled_data(fourier_horiz_normalised, 'MRI',
                                            'TV', 0.0003, subsampling_arr=mask,
                                            niter=1)

recons_0 = np.zeros(recons_1[0].shape, dtype='complex')
recons_0[:, :] = recons_1[0][:, :]

recons_2 = regularised_recons_from_subsampled_data(fourier_horiz_normalised, 'MRI',
                                            'TV', 0.0003, subsampling_arr=mask,
                                            niter=2000, recon_init=recons_0)
plt.figure()
plt.imshow(recon_horiz, cmap=plt.cm.gray)
plt.title('Melanie recon')
plt.figure()
plt.imshow(np.abs(recons_1[0]), cmap=plt.cm.gray)
plt.figure()
plt.imshow(np.abs(recons_2[0]), cmap=plt.cm.gray)


step = 1.5

list1 = ((np.arange(87830, 87862) - 87830) * step).tolist()
list2 = ((np.arange(87872, 87978) - 87872) * step + 58.5).tolist()
list3 = ((np.arange(87980, 88073) - 87872) * step + 58.5).tolist()

angle_list = list1+list2+list3


sino_new, mask = pad_sino(raw_data_phase, step, 0, 240, angle_list)

recons_1 = regularised_recons_from_subsampled_data(sino_new, 'CT',
                                            'TGV', 0.00, recon_dims=(167, 167),
                                            niter=200, a_offset=0, a_range=2*np.pi,
                                            d_offset=0, subsampling_arr=mask, d_width=40)

recons_2 = regularised_recons_from_subsampled_data(raw_data_phase, 'CT',
                                            'TGV', 0.00, recon_dims=(167, 167),
                                            niter=200, a_offset=0, a_range=2*np.pi,
                                            d_offset=0, d_width=40)




mask = circle_mask(recons_1[0].shape[0], 0.95)
recon_1_masked = recons_1[0]*mask
recon_2_masked = recons_2[0]*mask

plt.figure()
plt.imshow(recon_1_masked, cmap=plt.cm.gray)
plt.title("with padding")
plt.figure()
plt.imshow(recon_2_masked, cmap=plt.cm.gray)
plt.title("without padding")
plt.figure()
plt.imshow(rec_Paul, cmap=plt.cm.gray)

plt.show()

