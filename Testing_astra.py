from Utils import *
import os
import matplotlib.pyplot as plt
from skimage import io
from processing import *

directory = '/mnt/jlw31-XDrive/BIMI/ResearchProjects/MJEhrhardt/RC-MA1244_Faraday'
#data_path = directory + '/Data/04-20_CT_Paul_Quinn/phase/sino_cleaned/sino_0050.tif'
data_path = directory + '/Experiments/CT_diamond/sino_0050_cleaned.tif'

data = np.array(io.imread(data_path), dtype=float)

#(height, width) = losa.load_image(list_file[depth // 2]).shape
(height, width) = (167, 167)

step = 1.5
list1 = (np.arange(87830, 87862) - 87830) * step
list2 = (np.arange(87872, 87978) - 87872) * step + 58.5
list3 = (np.arange(87980, 88073) - 87872) * step + 58.5

list_angle = np.concatenate((list1, list2, list3)) * np.pi / 180.0
center = 83

#recon_astra = recon_astra(data, center, list_angle, 0.95, method="SIRT", num_iter=200)

t = VariationalRegClass('CT', 'TV')
recon_tv = t.regularised_recons_from_subsampled_data(data, 0.001, subsampling_arr=None,
                                                   recon_dims=(167, 167), niter=200, a_offset=0, a_range=2*np.pi,
                                                   d_offset=0, d_width=40)[0]

# plt.figure()
# plt.imshow(recon_astra, cmap=plt.cm.gray)
# plt.title("recon using astra")
# plt.savefig("astra_recon.png")

plt.figure()
plt.imshow(recon_tv, cmap=plt.cm.gray)
plt.title("recon using tv")
plt.savefig("tv_recon.png")

