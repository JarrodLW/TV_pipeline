# Created 13/11/2020
# The purpose of this script is to check how well the results of the 22102020 experiments conform to the data

import json
import numpy as np
import matplotlib.pyplot as plt
import odl
from myOperators import RealFourierTransform

with open('dTV/Results_MRI_dTV/TV_recons_multiple_avgs_22102020_finer_hyperparam_full_recon.json') as f:
    d = json.load(f)

d2 = d['avgs=8192']
d3 = d2['reg_type=TV']
recon = np.asarray(d3['reg_param=3.0e+04'])
#recon = np.asarray(d3['reg_param=2.0e+03'])
recon = recon[0] + 1j*recon[1]
recon_rotated = recon.T[:, ::-1]

plt.figure()
plt.imshow(recon, cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(recon_rotated), cmap=plt.cm.gray)

height, width = recon.shape

complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                              shape=[height, width], dtype='complex')
image_space = complex_space.real_space ** 2
forward_op = RealFourierTransform(image_space)

synth_data = forward_op(image_space.element([np.real(recon_rotated), np.imag(recon_rotated)])).asarray()
synth_data_complex = synth_data[0] + 1j*synth_data[1]
synth_data_ifft_shifted = np.fft.ifftshift(synth_data_complex)


#image_Li = np.reshape(np.fromfile('dTV/7LI_1H_MRI_Data_22102020/1mm_7Li_8192_avgs/2dseq', dtype=np.uint16), (32, 32))
#synth_data_2 = forward_op(image_space.element([image_Li, np.zeros(image_Li.shape)])).asarray()
#synth_data_complex_2 = synth_data_2[0] + 1j*synth_data_2[1]
#synth_data_ifft_shifted_2 = np.fft.ifftshift(synth_data_complex_2)

fourier_Li_real_im_padded = np.reshape(np.fromfile('dTV/7Li_1H_MRI_Data_22102020/1mm_7Li_8192_avgs/fid', dtype=np.int32), (64, 128))
fourier_Li_real_im = fourier_Li_real_im_padded[:, 1:65]
fourier_Li_real_im = fourier_Li_real_im[::2, :]
fourier_Li_real = fourier_Li_real_im[:, 1::2]
fourier_Li_im = fourier_Li_real_im[:, ::2]
fourier_Li = fourier_Li_real + fourier_Li_im*1j

recon_fourier = forward_op.inverse(forward_op.range.element([np.fft.fftshift(fourier_Li_real), np.fft.fftshift(fourier_Li_im)]))
recon_fourier_arr = recon_fourier.asarray()[0] + 1j*recon_fourier.asarray()[1]

norm_synth = np.sqrt(np.sum(np.square(np.abs(synth_data_ifft_shifted))))
norm_data = np.sqrt(np.sum(np.square(np.abs(fourier_Li))))

diff = synth_data_ifft_shifted/norm_synth - fourier_Li/norm_data
norm_diff = np.sqrt(np.sum(np.square(np.abs(diff))))

plt.figure()
plt.imshow(np.abs(diff), cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(fourier_Li), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.abs(synth_data_ifft_shifted), cmap=plt.cm.gray)


# recon_fourier_2 = forward_op.inverse(forward_op.range.element([fourier_Li_real, fourier_Li_im]))
# recon_fourier_arr_2 = recon_fourier_2.asarray()[0] + 1j*recon_fourier_2.asarray()[1]
#
# plt.figure()
# plt.imshow(np.abs(recon_fourier_arr), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(np.abs(recon_fourier_arr_2), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(np.abs(np.fft.fftshift(fourier_Li)), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(np.abs(fourier_Li), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(np.abs(np.fft.fftshift(fourier_Li)), cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(np.abs(synth_data_ifft_shifted), cmap=plt.cm.gray)
# plt.colorbar()
#
# plt.figure()
# plt.imshow(np.abs(synth_data_ifft_shifted_2), cmap=plt.cm.gray)
# plt.colorbar()
#
# plt.figure()
# plt.imshow(np.abs(fourier_Li), cmap=plt.cm.gray)
# plt.colorbar()

# synth_data_ifft_shifted_normalised = synth_data_ifft_shifted/np.sqrt(np.sum(np.square(np.abs(synth_data_ifft_shifted))))
# fourier_Li_normalised = fourier_Li/np.sqrt(np.sum(np.square(np.abs(fourier_Li))))
#
# plt.figure()
# plt.imshow(np.abs(synth_data_ifft_shifted_normalised), cmap=plt.cm.gray)
# plt.colorbar()
#
# plt.figure()
# plt.imshow(np.abs(fourier_Li_normalised), cmap=plt.cm.gray)
# plt.colorbar()
