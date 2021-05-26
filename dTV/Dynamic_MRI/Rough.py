import odl
import numpy as np
import scipy as sp
from dTV.Dynamic_MRI.DeformationModulesODL.deform import Kernel
from dTV.Dynamic_MRI.DeformationModulesODL.deform import DeformationModuleAbstract
from dTV.Dynamic_MRI.DeformationModulesODL.deform import SumTranslations
from dTV.Dynamic_MRI.DeformationModulesODL.deform import UnconstrainedAffine
from dTV.Dynamic_MRI.DeformationModulesODL.deform import LocalScaling
from dTV.Dynamic_MRI.DeformationModulesODL.deform import LocalRotation
from dTV.Dynamic_MRI.DeformationModulesODL.deform import EllipseMvt
from dTV.Dynamic_MRI.DeformationModulesODL.deform import TemporalAttachmentModulesGeom

rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')

I0 = odl.phantom.transmission.shepp_logan(rec_space, modified=True)

#template= odl.phantom.shepp_logan(space)
#I0 = rec_space.element(sp.ndimage.filters.gaussian_filter(I0.asarray(), 1))

row_coords, col_coords = np.meshgrid(np.arange(template.shape[0]), np.arange(template.shape[1]), indexing='ij')
v = disp.asarray()[0]
u = disp.asarray()[1]
#deformed_im = warp(template, np.array([row_coords + v, col_coords + u]), mode='nearest')

frame_num = 20
init = template
for i in range(frame_num):

    deformed_im = warp(init, np.array([row_coords + v, col_coords + u]), mode='nearest')
    init = deformed_im

plt.figure()
plt.imshow(template, cmap=plt.cm.gray)

plt.figure()
plt.imshow(deformed_im, cmap=plt.cm.gray)




circ_mask = circle_mask(128, 0.75)

V = rec_space.tangent_bundle
Y = odl.tensor_space(6)
embed = Embedding_Affine(Y, V)
phi = np.pi/180
aff_arr = [0, 0, np.cos(phi) - 1, np.sin(phi), -np.sin(phi), np.cos(phi) - 1]
disp = embed(Y.element(aff_arr))
deformation = LinDeformFixedDisp(disp)

deformed_im = deformation(template)

plt.figure()
plt.imshow(template, cmap=plt.cm.gray)

plt.figure()
plt.imshow(deformed_im, cmap=plt.cm.gray)


potential = np.zeros(template.shape)

for i in range(template.shape[0]):
    for j in range(template.shape[1]):

        potential[i, j] = (i-template.shape[0]//2)**2 - (j-template.shape[1]//2)**2

plt.imshow(potential, cmap=plt.cm.gray)

disp = [circ_mask, circ_mask]*odl.Gradient(rec_space)((1/10**5)*potential)

# disp = rec_space.tangent_bundle.element([(1/100)*circ_mask*np.ones(template.shape),
#                                          (1/100)*circ_mask*np.ones(template.shape)])
deformation = LinDeformFixedDisp(disp)

deformed_im = deformation(template)
deformed_im_2 = deformation(deformed_im)

frame_num = 20
init = template
for i in range(frame_num):

    deformed_im = deformation(init)
    init = deformed_im

plt.figure()
plt.imshow(template, cmap=plt.cm.gray)

plt.figure()
plt.imshow(deformed_im, cmap=plt.cm.gray)



#
image = template
rows, cols = image.shape[0], image.shape[1]

src_cols = np.linspace(0, cols, 10)
src_rows = np.linspace(0, rows, 10)
src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]

# add sinusoidal oscillation to row coordinates
dst_rows = src[:, 1] - np.sin(np.linspace(0, np.pi, src.shape[0])) * 0.5
dst_cols = src[:, 0]
#dst_rows *= 1.5
#dst_rows -= 1.5 * 50
#dst_rows *= 1.5
dst = np.vstack([dst_cols, dst_rows]).T

tform = PiecewiseAffineTransform()
tform.estimate(src, dst)

#out_rows = image.shape[0] - 1.5 * 50
out_rows = rows
out_cols = cols
out = warp(image, tform, output_shape=(out_rows, out_cols))

frame_num = 20
init = template
for i in range(frame_num):

    deformed_im = warp(init, tform, output_shape=(out_rows, out_cols))
    init = deformed_im

plt.figure()
plt.imshow(image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(deformed_im, cmap=plt.cm.gray)



