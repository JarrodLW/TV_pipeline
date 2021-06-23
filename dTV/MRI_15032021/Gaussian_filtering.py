import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res.mat')
image_H_high_res = mat['movingRegisteredRigid']

# image_H_high_res = np.load('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res.npy')
# date = '24052021'
# dir_H = 'dTV/MRI_15032021/Data_' + date + '/H_data/'
#
# H_index_high_res = 32
#
# f_coeffs = np.reshape(np.fromfile(dir_H +str(H_index_high_res)+'/fid', dtype=np.int32), (128, 256))
# f_coeffs = f_coeffs[:, 1::2] + 1j*f_coeffs[:, ::2]
# recon_high_res = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f_coeffs)))

#image_H_high_res = np.abs(recon_high_res)

plt.figure()
plt.imshow(image_H_high_res, cmap=plt.cm.gray)
plt.colorbar()

filtered_image_1 = sp.ndimage.filters.gaussian_filter(image_H_high_res, 0.5)
filtered_image_2 = sp.ndimage.filters.gaussian_filter(image_H_high_res, 0.65)

plt.figure()
plt.imshow(image_H_high_res, cmap=plt.cm.gray)

plt.figure()
plt.imshow(filtered_image_1, cmap=plt.cm.gray)

plt.figure()
plt.imshow(filtered_image_2, cmap=plt.cm.gray)

np.save('dTV/MRI_15032021/Results_24052021/pre_registered_H_high_res_filtered.npy', filtered_image_2)
