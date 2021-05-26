#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:52:08 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:16:55 2017

@author: bgris
"""
import odl
import numpy as np

##%% Create data from lddmm registration
import matplotlib.pyplot as plt

from DeformationModulesODL.deform import Kernel
from DeformationModulesODL.deform import DeformationModuleAbstract
from DeformationModulesODL.deform import SumTranslations
from DeformationModulesODL.deform import UnconstrainedAffine
from DeformationModulesODL.deform import LocalScaling
from DeformationModulesODL.deform import LocalRotation
from DeformationModulesODL.deform import EllipseMvt
from DeformationModulesODL.deform import TemporalAttachmentModulesGeom

import scipy



def compute_grid_deformation(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size

    grid_points = initial_grid

    for t in range(nb_time_points):
        velocity = np.empty_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            velocity[i, ...] = vi.interpolation(grid_points)
        grid_points += time_step*velocity

    return grid_points


def compute_grid_deformation_list(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size
    grid_list=[]
    grid_points=initial_grid.copy()
    grid_list.append(initial_grid)

    for t in range(nb_time_points):
        velocity = np.empty_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            velocity[i, ...] = vi.interpolation(grid_points)
        grid_points += time_step*velocity
        grid_list.append(grid_points.copy())

    return grid_list


# As previously but check if points of the grids are in the
# Domain and if they are not the velocity is equal to zero
def compute_grid_deformation_list_bis(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size
    grid_list=[]
    grid_points=initial_grid.T.copy()
    grid_list.append(initial_grid.T)
    mini=vector_fields_list[0].space[0].min_pt
    maxi=vector_fields_list[0].space[0].max_pt
    for t in range(nb_time_points):
        print(t)
        velocity = np.zeros_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            for k in range(len(initial_grid.T)):
                isindomain=1
                point=grid_points[k]
                for u in range(len(mini)):
                    if (point[u]<mini[u] or point[u]>maxi[u] ):
                        isindomain=0
                if (isindomain==1):
                    velocity[k][i] = vi.interpolation(point)

        grid_points += time_step*velocity
        grid_list.append(grid_points.copy().T)

    return grid_list


def plot_grid(grid, skip):
    for i in range(0, grid.shape[1], skip):
        plt.plot(grid[0, i, :], grid[1, i, :], 'r', linewidth=0.5)

    for i in range(0, grid.shape[2], skip):
        plt.plot(grid[0, :, i], grid[1, :, i], 'r', linewidth=0.5)


def make_video_LDDMM(I,grid, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        img = image.copy()
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid







def make_video_path(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


#


#%% Generate data

#I0name = '/home/bgris/Downloads/pictures/i_highres.png'
#I1name = '/home/bgris/Downloads/pictures/c_highres.png'
#
#
## Get digital images
#I0 = plt.imread(I0name)
#I1 =plt.imread(I1name)
#
#I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
#I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]
#
#
#space = odl.uniform_discr(
#    min_pt=[-16, -16], max_pt=[16, 16], shape=[128, 128],
#    dtype='float32', interp='linear')
#




I1name = 'bgris/Downloads/pictures/j.png'
I0name = 'bgris/Downloads/pictures/v.png'
I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)

# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=I0.shape,
    dtype='float32', interp='linear')

# Create the ground truth as the given image
ground_truth =space.element(I1)


I0=space.element(I0)
I1=space.element(I1)


# Create the template as the given image
template = space.element(I0)


##### Cas shepp logan
#space = odl.uniform_discr(
#    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
#    dtype='float32', interp='linear')
#
#template= odl.phantom.shepp_logan(space)
#template.show(clim=[1,1.1])

#### Cas ellipse

space=odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')


a_list=[0.2,0.4,0.6,0.8,1]
b_list=[1,0.8,0.6,0.4,0.2]
fac=0.3
nb_ellipses=len(a_list)
images_ellipses=[]
for i in range(nb_ellipses):
    ellipses=[[1,fac* a_list[i], fac*b_list[i], 0.0000, 0.0000, 0]]
    images_ellipses.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
I0=space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[0].asarray(),3))
I1=space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[nb_ellipses-1].asarray(),3))
template=I0

# Give the number of directions
num_angles = 10

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0, np.pi, num_angles,
                                        nodes_on_bdry=[(True, False)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, 192)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                       detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
forward_op=odl.IdentityOperator(space)





# Give the number of directions
num_angles = 10

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0.0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, True)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, int(round(space.shape[0]*np.sqrt(2))))

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
##forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')



#
#space = odl.uniform_discr(
#    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
#    dtype='float32', interp='linear')
#forward_op=odl.IdentityOperator(space)
#
#
#template= odl.phantom.shepp_logan(space)
#template.show(clim=[1,1.1])


template=template.space.element(scipy.ndimage.filters.gaussian_filter(template.asarray(),1.5))

#
#NRotation=1
#space_mod = odl.uniform_discr(
#    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
#    dtype='float32', interp='nearest')
#
#kernelrot=Kernel.GaussianKernel(3)
#rotation=LocalRotation.LocalRotation(space_mod, NRotation, kernelrot)
#
#GD=rotation.GDspace.element([[-0.0500, -9.5]])
##GD=rotation.GDspace.element([[-2.5,3.5]])
#Cont=rotation.Contspace.element([1])
#
#I1=template.copy()
#inv_N=1
#for i in range(5):
#    vect_field=rotation.ComputeField(GD,Cont).copy()
#    I1=template.space.element(
#                odl.deform.linearized._linear_deform(I1,
#                               -inv_N * vect_field)).copy()
#
#I1.show()


ellipses= [[1.50, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.2, .1100, .3100, 0.2200, 0.0000, -18],
            [-.2, .1600, .4100, -.2200, 0.0000, 18],
            [0.1, .2100, .2500, 0.0000, 0.3500, 0],
            [0.1, .0460, .0460, 0.0000, 0.1000, 0],
            [0.1, .0460, .0460, 0.0000, -.1000, 0],
            [0.8, .0460, .0330, -.0800, -.6050, 0],
            [0.8, .0330, .0330, 0.0000, -.6060, 0],
            [0.8, .0330, .0460, 0.0600, -.6050, 0]]


ellipses= [[1.50, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.2, .1100, .3100, 0.2200, 0.0000, -18],
            [-.2, .1600, .4100, -.2200, 0.0000, 18],
            [0.1, .2100, .2500, 0.0000, 0.3500, 0],
            [0.1, .0460, .0460, 0.0000, 0.1000, 0],
            [0.1, .0460, .0460, 0.0000, -.1000, 0],
            [0.8, .0460, .0230, -.06500, -0.6550, 45],
            [0.8, .0230, .0230, 0.0000, -.6060, 0],
            [0.8, .0230, .0460, 0.0400, -.5550, 45]]


ellipses= [[1.50, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.2, .1100, .3100, 0.2200, 0.0000, -18],
            [-.2, .1600, .4100, -.2200, 0.0000, 18],
            [0.1, .2100, .2500, 0.0000, 0.3500, 0],
            [0.1, .0460, .0460, 0.0000, 0.1000, 0],
            [0.1, .0460, .0460, 0.0000, -.1000, 0],
            [0.8, .0460, .0230, -0.01, -0.670, 80],
            [0.8, .0230, .0230, 0.0000, -.6060, 0],
            [0.8, .0230, .0460, 0.01, -.560, 80]]


ellipses1= [[1.50, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.2, .1100, .3100, 0.2200, 0.0000, -18],
            [-.2, .1600, .4100, -.2200, 0.0000, 18],
            [0.1, .2100, .2500, 0.0000, 0.3500, 0],
            [0.1, .0460, .0460, 0.0000, 0.1000, 0],
            [0.1, .0460, .0460, 0.0000, -.1000, 0],
            [0.8, .120, .0830, 0.0, -0.670, 0]]





phantom=odl.phantom.geometric.ellipsoid_phantom(space,ellipses)
phantom.show()

template=template.space.element(scipy.ndimage.filters.gaussian_filter(phantom.asarray(),1.5))




NRotation=1
space_mod = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='nearest')

kernelrot=Kernel.GaussianKernel(5)
rotation=LocalRotation.LocalRotation(space_mod, NRotation, kernelrot)

#GD=rotation.GDspace.element([[-0.0500, -9.5]])
GD=rotation.GDspace.element([[3,0]])
Cont=rotation.Contspace.element([0.4])

I1=template.copy()
inv_N=1
for i in range(5):
    vect_field=rotation.ComputeField(GD,Cont).copy()
    I1=template.space.element(
                odl.deform.linearized._linear_deform(I1,
                               -inv_N * vect_field)).copy()

NAffine=2
kernelaff=Kernel.GaussianKernel(3)
affine=UnconstrainedAffine.UnconstrainedAffine(space_mod, NAffine, kernelaff)

GD_affine=affine.GDspace.element([[-5,5],[3,4]])
Cont_affine=-1*affine.Contspace.element([[[0.5,0],[1,-1],[1,1]],[[-1,0.5],[-1,0],[0.5,0]]])
vect_field_affine=affine.ComputeField(GD_affine,Cont_affine)

I1=I1=template.space.element(odl.deform.linearized._linear_deform(I1.copy(),vect_field_affine))




#I1=odl.phantom.geometric.ellipsoid_phantom(space,ellipses1)
#I1.show(clim=[1 , 1.1])
#I1=template.space.element(scipy.ndimage.filters.gaussian_filter(I1,1.5))

template.show()
I1.show()

# Create projection data by calling the ray transform on the phantom
proj_data = forward_op(I1)


space_mod = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256],
    dtype='float32', interp='nearest')

#%% Define Module

dim=2
Ntrans=1
NAffine=2
NScaling=3
NRotation=1
NEllipse=1
##
#miniX=-5
#maxiX=5
#miniY=-10
#maxiY=10
#GD_init_trans=[]
#ec=4
#cont=0
#nbsqX=round((maxiX-miniX)/ec)
#nbsqY=round((maxiY-miniY)/ec)
#for i in range(nbsqX):
#    for j in range(nbsqY):
#      GD_init_trans.append(odl.rn(2).element([miniX+(i+0.5)*ec, miniY+(j+0.5)*ec]))
#      cont+=1

#Ntrans=cont
kerneltrans=Kernel.GaussianKernel(6)
translation=SumTranslations.SumTranslations(space_mod, Ntrans, kerneltrans)
#translationF=SumTranslations.SumTranslationsFourier(space_mod, Ntrans, kernel)

kernelaff=Kernel.GaussianKernel(5)
affine=UnconstrainedAffine.UnconstrainedAffine(space_mod, NAffine, kernelaff)

#scaling=LocalScaling.LocalScaling(space_mod, NScaling, kernel)

kernelrot=Kernel.GaussianKernel(5)
rotation=LocalRotation.LocalRotation(space_mod, NRotation, kernelrot)


kernelelli=Kernel.GaussianKernel(2)
Name='DeformationModulesODL/deform/vect_field_ellipses'
elli=EllipseMvt.EllipseMvt(space_mod, Name, kernelelli)

#Module=DeformationModuleAbstract.Compound([translation,rotation])
Module=DeformationModuleAbstract.Compound([elli])
#ModuleF=translationF
#Module=affine
#Module=DeformationModuleAbstract.Compound([translation,translation])
#
##%%
#nb_time_point_int=10
#GD=Module.GDspace.element([[[-0.5,-10]]])
#Cont=5*odl.ProductSpace(Module.Contspace,nb_time_point_int+1).one()
#
#X=functional_mod.domain.element([GD,Cont].copy())
#
#vector_fields_list=energy_op_lddmm.domain.zero()
#
#I_t=Shoot_mixt(vector_fields_list,X[0],X[1])
#I_t[0].show()
#I_t[nb_time_point_int].show()
#I1=I_t[nb_time_point_int].copy()
#
#GD_affine=affine.GDspace.element([[-2,-4],[3,4]])
#Cont_affine=-1*affine.Contspace.element([[[1,0],[1,-1],[0,1]],[[-1,0.5],[-1,0],[0.5,0]]])
#vect_field_affine=affine.ComputeField(GD_affine,Cont_affine)
#
#I1=odl.deform.linearized._linear_deform(I_t[nb_time_point_int],vect_field_affine)
#I1=template.space.element(I1)
#I1.show()
#template.show()
#%% Define functional
lam=0.0001
nb_time_point_int=10
#template=I0


##data_time_points=np.array([0,0.5,0.8,1])
#data_time_points=np.array([0,0.2,0.4,0.6,0.8,1])
#data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
#data=data_space.element([forward_op(image_N0[0]),forward_op(image_N0[10]),
#              forward_op(image_N0[0]),forward_op(image_N0[10]),
#              forward_op(image_N0[0]),forward_op(image_N0[10])])
##data=data_space.element([forward_op(image_N0[0]),forward_op(image_N0[5]),
##              forward_op(image_N0[8]),forward_op(image_N0[10])])
#
#forward_operators=[forward_op,forward_op,forward_op,forward_op,
#                   forward_op, forward_op, forward_op]
#data_image=[(image_N0[0]),(image_N0[10]),
#              (image_N0[0]),(image_N0[10]),
#              (image_N0[0]),(image_N0[10])]

lamb0=1e-7
lamb1=1e-2
import scipy
data_time_points=np.array([1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
data=data_space.element([forward_op(I1)])
forward_operators=[forward_op]
data_image=[I1]


Norm=odl.solvers.L2NormSquared(forward_op.range)

functional_mod = TemporalAttachmentModulesGeom.FunctionalModulesGeom(lamb0, nb_time_point_int, template, data, data_time_points, forward_operators,Norm, Module)



# The parameter for kernel function
sigma = 1

# Give kernel function
def kernel_lddmm(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


# Define energy operator
energy_op_lddmm=odl.deform.TemporalAttachmentLDDMMGeom(nb_time_point_int, template ,data,
                            data_time_points, forward_operators,Norm, kernel_lddmm,
                            domain=None)


Reg=odl.deform.RegularityLDDMM(kernel_lddmm,energy_op_lddmm.domain)

functional_lddmm=energy_op_lddmm + lamb1*Reg

def Mix_vect_field(vect_field_list,GD_init,Cont):
    space_pts=template.space.points()
    GD=GD_init.copy()
    vect_field_list_tot=vect_field_list.copy()
    for i in range(nb_time_point_int):
        vect_field_list_mod=functional_mod.Module.ComputeField(GD,Cont[i]).copy()
        vect_field_list_mod_interp=template.space.tangent_bundle.element([vect_field_list_mod[u].interpolation(space_pts.T) for u in range(dim)]).copy()
        vect_field_list_tot[i]+=vect_field_list_mod_interp.copy()
        GD+=(1/nb_time_point_int)*functional_mod.Module.ApplyVectorField(GD,vect_field_list_tot[i]).copy()

    return vect_field_list_tot

def Shoot_mixt(vect_field_list,GD,Cont):
    vect_field_list_mod=Mix_vect_field(vect_field_list,GD,Cont).copy()
    I=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_list_mod, template)
    return I

def attach_tot(vect_field_list,GD,Cont):
    I=Shoot_mixt(vect_field_list,GD,Cont)
    return Norm(forward_operators[0](I[nb_time_point_int])-data[0])



def grad_attach_vector_field(vect_field_list,GD,Cont):
    vect_field_list_tot=Mix_vect_field(vect_field_list,GD,Cont)
    grad=energy_op_lddmm.gradient(vect_field_list_tot).copy()
    return grad
#

def ComputeGD_mixt(vect_field_list,GD,Cont):
    GD_list=[]
    GD_list.append(GD.copy())
    vect_field_list_mixt=Mix_vect_field(vect_field_list,GD,Cont)
    for i in range(nb_time_point_int):
        GD_temp=GD_list[i].copy()
        GD_temp+=(1/nb_time_point_int)*functional_mod.Module.ApplyVectorField(GD_temp,vect_field_list_mixt[i])
        GD_list.append(GD_temp.copy())

    return GD_list.copy()
#


def ComputeModularVectorFields(vect_field_list,GD,Cont):
    GD_list=ComputeGD_mixt(vect_field_list,GD,Cont)
    vect_field_list_tot=[]
    for i in range(nb_time_point_int+1):
        vect_field_list_tot.append(functional_mod.Module.ComputeField(GD_list[i],Cont[i]))

    return vect_field_list_tot
#

#%% LDDMM  gradient descent
niter=200
eps=0.1
vector_fields_list_init=energy_op_lddmm.domain.zero()
vector_fields_list=vector_fields_list_init.copy()
attachment_term=energy_op_lddmm(vector_fields_list)
print(" Initial ,  attachment term : {}".format(attachment_term))

for k in range(niter):
    grad=functional_lddmm.gradient(vector_fields_list)
    vector_fields_list= (vector_fields_list- eps *grad).copy()
    attachment_term=energy_op_lddmm(vector_fields_list)
    print(" iter : {}  ,  attachment term : {}".format(k,attachment_term))
#

#%% see result LDDMM
I_t=odl.deform.ShootTemplateFromVectorFields(vector_fields_list, template)

grid_points=compute_grid_deformation_list(vector_fields_list, 1/nb_time_point_int, template.space.points().T)

for t in range(nb_time_point_int+1):

    #t=nb_time_point_int
    I_t[t].show('LDDMM time {}'.format(t+1))
    grid=grid_points[t].reshape(2, space.shape[0], space.shape[1]).copy()
    plot_grid(grid, 5)
#
#%% save images  LDDMM
I_t=odl.deform.ShootTemplateFromVectorFields(vector_fields_list_load, template)

grid_points=compute_grid_deformation_list(vector_fields_list, 1/nb_time_point_int, template.space.points().T)

for t in range(nb_time_point_int+1):
    plt.figure()
    #t=nb_time_point_int
    I_t[t].show('LDDMM time {}'.format(t+1))
    grid=grid_points[t].reshape(2, space.shape[0], space.shape[1]).copy()
    plot_grid(grid, 5)
    plt.savefig('LDDMM_bis_time{}.png'.format(t+1))
#


#%% video lddmm
I_t=odl.deform.ShootTemplateFromVectorFields(vector_fields_list_load, template)

grid_points=compute_grid_deformation_list(vector_fields_list, 1/nb_time_point_int, template.space.points().T)

from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
format="XVID"
outimg=None
size=None
fps=5
is_color=True
fourcc = VideoWriter_fourcc(*format)
vid = None
for t in range(nb_time_point_int+1):
    img=plt.figure()
    I_t[t].show('LDDMM time {}'.format(t+1))
    grid=grid_points[t].reshape(2, space.shape[0], space.shape[1]).copy()
    plot_grid(grid, 5)
    plt.savefig('LDDMMtime{}.pdf'.format(t+1))
    if vid is None:
        if size is None:
            size = I_t[t].shape[1], I_t[t].shape[0]
        vid = VideoWriter("LDDMM_video.avi", fourcc, float(fps), size, is_color)
    if size[0] != I_t[t].shape[1] and size[1] != I_t[t].shape[0]:
        img = resize(img, size)
    vid.write(img)
vid.release()



#%%
#Saving results in files
for t in range(nb_time_point_int+1):
    np.savetxt('odl/examples/Mixed_SL/LDDMM_bis__Vector_Field_time_{}'.format(t),vector_fields_list[t])
    np.savetxt('odl/examples/Mixed_SL/LDDMM_bis__Deformed_Grid_time_{}'.format(t),grid_points[t])


#%% Load data
vector_fields_list_load=energy_op_lddmm.domain.zero()
grid_load=[]
for t in range(nb_time_point_int+1):
    vector_fields_list_load[t]=template.space.tangent_bundle.element(np.loadtxt('odl/examples/Mixed_SL/LDDMM_bis__Vector_Field_time_{}'.format(t))).copy()
    grid_t=template.space.tangent_bundle.element(np.loadtxt('odl/examples/Mixed_SL/LDDMM_bis__Deformed_Grid_time_{}'.format(t))).copy()
    grid_load.append(np.array(grid_t).copy())


#%%  see result Mixed strategy
I_t=Shoot_mixt(vector_fields_list,X[0],X[1])

vector_fields_list_tot=Mix_vect_field(vector_fields_list,X[0],X[1])
grid_points=compute_grid_deformation_list(vector_fields_list_tot, 1/nb_time_point_int, template.space.points().T)

for t in range(nb_time_point_int+1):
    #t=nb_time_point_int
    I_t[t].show('Mixed strategy time {}'.format(t+1))
    grid=grid_points[t].reshape(2, space.shape[0], space.shape[1]).copy()
    plot_grid(grid, 5)
#
#%% save images  Mixed strategy
I_t=odl.deform.ShootTemplateFromVectorFields(vector_fields_list_tot, template)

grid_points=compute_grid_deformation_list(vector_fields_list, 1/nb_time_point_int, template.space.points().T)

for t in range(nb_time_point_int+1):
    plt.figure()
    #t=nb_time_point_int
    I_t[t].show('Mixed strategy time {}'.format(t+1))
    grid=grid_points[t].reshape(2, space.shape[0], space.shape[1]).copy()
    plot_grid(grid, 5)
    plt.savefig('Mixed_bis_ strategy time {}.pdf'.format(t+1))
#
#%%
#Saving results in files
for i in range(NbMod):
    np.savetxt('odl/examples/Mixed_SL/Mixed_bis__GD_time_0_Mod_{}'.format(i),np.asarray(X[0][0]))
for t in range(nb_time_point_int+1):
    for i in range(NbMod):
        np.savetxt('odl/examples/Mixed_SL/Mixed_bis__Cont_time_{}_Mod_{}'.format(t,i),X[1][t][i])
    np.savetxt('odl/examples/Mixed_SL/Mixed_bis__Vector_Field_time_{}'.format(t),vector_fields_list[t])
    np.savetxt('odl/examples/Mixed_SL/Mixed_bis__Deformed_Grid_time_{}'.format(t),grid_points[t])


#%% Load data
vector_fields_list_load=energy_op_lddmm.domain.zero()
X=functional_mod.domain.element()
grid_load=[]
for i in range(NbMod):
    X[0][i][0]=np.loadtxt('odl/examples/Mixed_SL/Mixed_bis__GD_time_0_Mod_{}'.format(i)).copy()
    
for t in range(nb_time_point_int+1):
    for i in range(NbMod):
        X[1][t][i]=np.loadtxt('odl/examples/Mixed_SL/Mixed_bis__Cont_time_{}_Mod_{}'.format(t,i)).copy()
    vector_fields_list_load[t]=template.space.tangent_bundle.element(np.loadtxt('odl/examples/Mixed_SL/Mixed_bis__Vector_Field_time_{}'.format(t))).copy()
#    #grid_t=template.space.tangent_bundle.element(np.loadtxt('/home/bgris/odl/examples/Mixed_SL/MixedDeformed_Grid_time_{}'.format(t))).copy()
    #grid_load.append(np.array(grid_t).copy())

#%%
GD_init=Module.GDspace.element([[[0,-10]]])
Cont_init=odl.ProductSpace(Module.Contspace,nb_time_point_int+1).zero()

#%% Naive Gradient descent : gradient computed by finite differences
# Descent of vect field too
# Descent for all times simultaneously
#Combination of modules
#functional=functionalF
niter=100
eps = 0.01

X=functional_mod.domain.element([GD_init,Cont_init].copy())

vector_fields_list_init=energy_op_lddmm.domain.zero()
vector_fields_list=vector_fields_list_init.copy()
#%%
attachment_term=attach_tot(vector_fields_list,X[0],X[1])
energy=attachment_term
Reg_mod=functional_mod.ComputeReg(X)
print(" Initial , attachment term : {}, reg_mod = {}".format(attachment_term,Reg_mod))
gradGD=functional_mod.Module.GDspace.element()
gradCont=odl.ProductSpace(functional_mod.Module.Contspace,nb_time_point_int+1).element()

d_GD=functional_mod.Module.GDspace.zero()
d_Cont=odl.ProductSpace(functional_mod.Module.Contspace,nb_time_point_int+1).zero()
ModulesList=Module.ModulesList
NbMod=len(ModulesList)
delta=0.1
epsmax=1
deltamin=0.1
# 0=SumTranslations, 1=affine, 2=scaling, 3=rotation
Types=[3]
energy=functional(X)+1
eps0Cont=[0.01,1]
eps0GD=[0.1,1]
eps_vect_field=0.01
cont=1
for k in range(niter):

    ## gradient with respect to vector field
    #grad_vect_field=grad_attach_vector_field(vector_fields_list,X[0],X[1])
    ## (1-lamb1) because of the gradient of the regularity term
    #vector_fields_list=((1-eps_vect_field*lamb1)*vector_fields_list-eps_vect_field*grad_vect_field ).copy()

    energy=attach_tot(vector_fields_list,X[0],X[1])
    Reg_mod=functional_mod.ComputeReg(X)
    energy_mod=Reg_mod+energy
    print('k={}  attachment term = {}, reg_mod={}'.format(k,energy,Reg_mod))
    #Computation of the gradient by finite differences

    for i in range(NbMod):
        if (Types[i]==1):
            NAffine=ModulesList[i].NAffine
            for n in range(NAffine):
                for d in range(dim+1):
                    for u in range(dim):
                        X_temp=X.copy()
                        if eps0Cont[i]>epsmax:
                            eps0Cont[i]=epsmax
                        eps=eps0Cont[i]
                        ismax=0
                        der=np.empty(nb_time_point_int)
                        print('k={} i={}  n={}  d={}  u={} eps={} attachment term = {}'.format(k,i,n,d,u,eps,energy))
                        for t in range(nb_time_point_int):
                            X_temp_diff=X.copy()
                            delta=0.1*np.abs(X_temp_diff[1][t][i][n][d][u])
                            if (delta==0):
                                delta=0.1
                            X_temp_diff[1][t][i][n][d][u]+=delta
                            energy_diff=attach_tot(vector_fields_list,X_temp_diff[0],X_temp_diff[1])
                            Reg_mod_diff=functional_mod.ComputeReg(X_temp_diff)
                            der[t]=(energy_diff+Reg_mod_diff-energy-Reg_mod)/delta
                            X_temp[1][t][i][n][d][u]-=eps*der[t]
                        energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                        if(energy_temp>energy_mod):
                            for ite in range(10):
                                eps*=0.8
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n][d][u]-=eps*der[t]
                                energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                                if (energy_mod<energy):
                                    ismax=1
                                    eps0Cont[i]=eps
                                    break
                        else:
                            for ite in range(10):
                                eps*=1.2
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n][d][u]-=eps*der[t]
                                energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                                if (energy_temp>=energy_mod):
                                    eps/=1.2
                                    break
                            eps0Cont[i]=eps
                            ismax=1


                        # Now we have 'the best' eps
                        if (ismax==1):
                            for t in range(nb_time_point_int):
                                X[1][t][i][n][d][u]-=eps*der[t]
                            energy=attach_tot(vector_fields_list,X[0],X[1])
                            energy_mod=functional_mod.ComputeReg(X)+attach_tot(vector_fields_list,X[0],X[1])
                            Reg_mod=functional_mod.ComputeReg(X)

        elif (Types[i]==0):
                Ntrans=ModulesList[i].Ntrans
                for n in range(Ntrans):
                    for d in range(dim):
                        if eps0Cont[i]>epsmax:
                            eps0Cont[i]=epsmax
                        eps=eps0Cont[i]
                        print('k={} i={}  n={}  d={} eps={}  attachment term = {}'.format(k,i,n,d,eps,energy))
                        X_temp=X.copy()
                        ismax=0
                        der=np.empty(nb_time_point_int)
                        for t in range(nb_time_point_int):
                            X_temp_diff=X.copy()
                            delta=0.1*np.abs(X_temp_diff[1][t][i][n][d])
                            if (delta<1e-3):
                                delta=deltamin
                            X_temp_diff[1][t][i][n][d]+=delta
                            energy_diff=attach_tot(vector_fields_list,X_temp_diff[0],X_temp_diff[1])
                            Reg_mod_diff=functional_mod.ComputeReg(X_temp_diff)
                            der[t]=(energy_diff+Reg_mod_diff-energy-Reg_mod)/delta
                            X_temp[1][t][i][n][d]-=eps*der[t]
                        energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                        if(energy_temp>energy_mod):
                            for ite in range(10):
                                eps*=0.8
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n][d]-=eps*der[t]
                                energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                                if (energy_temp<energy_mod):
                                    ismax=1
                                    eps0Cont[i]=eps
                                    break
                        else:
                            for ite in range(10):
                                eps*=1.2
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n][d]-=eps*der[t]
                                energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                                if (energy_temp>=energy_mod):
                                    eps/=1.2
                                    break
                            eps0Cont[i]=eps
                            ismax=1


                        # Now we have 'the best' eps
                        if (ismax==1):
                            for t in range(nb_time_point_int):
                                X[1][t][i][n][d]-=eps*der[t]
                            energy=attach_tot(vector_fields_list,X[0],X[1])
                            energy_mod=functional_mod.ComputeReg(X)+attach_tot(vector_fields_list,X[0],X[1])
                            Reg_mod=functional_mod.ComputeReg(X)

        elif (Types[i]==3 or Types[i]==2):
                if (Types[i]==3):
                    Nrot=ModulesList[i].NRotation
                else:
                      Nrot=ModulesList[i].NScaling
                for n in range(Nrot):
                        X_temp=X.copy()
                        if (eps0Cont[i]>epsmax):
                            eps0Cont[i]=epsmax
                        eps=eps0Cont[i]
                        print('k={}  i={}  n={}   energy= {}'.format(k,i,n,energy))
                        ismax=0
                        der=np.empty(nb_time_point_int)
                        for t in range(nb_time_point_int):
                            X_temp_diff=X.copy()
                            delta=0.1*np.abs(X_temp_diff[1][t][i][n])
                            if (delta<1e-3):
                                delta=deltamin
                            X_temp_diff[1][t][i][n]+=delta
                            energy_diff=attach_tot(vector_fields_list,X_temp_diff[0],X_temp_diff[1])
                            Reg_mod_diff=functional_mod.ComputeReg(X_temp_diff)
                            der[t]=(energy_diff+Reg_mod_diff-energy-Reg_mod)/delta
                            X_temp[1][t][i][n]-=eps*der[t]
                        energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                        if(energy_temp>energy_mod):
                            for ite in range(10):
                                eps*=0.8
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n]-=eps*der[t]
                                energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                                if (energy_temp<energy_mod):
                                    ismax=1
                                    eps0Cont[i]=eps
                                    break
                        else:
                            for ite in range(10):
                                eps*=1.2
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n]-=eps*der[t]
                                energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                                if (energy_temp>=energy):
                                    eps/=1.2
                                    break
                            eps0Cont[i]=eps
                            ismax=1


                        # Now we have 'the best' eps
                        if(ismax==1):
                            for t in range(nb_time_point_int):
                                X[1][t][i][n]-=eps*der[t]
                            energy=attach_tot(vector_fields_list,X[0],X[1])
                            energy_mod=functional_mod.ComputeReg(X)+attach_tot(vector_fields_list,X[0],X[1])
                            Reg_mod=functional_mod.ComputeReg(X)



    for i in range(NbMod):
            #if (Types[i]==0):
            if (Types[i]==3):
                Ntrans=ModulesList[i].NRotation
            elif (Types[i]==3):
                Ntrans=ModulesList[i].NScaling
            elif (Types[i]==1):
                Ntrans=ModulesList[i].NAffine
            elif (Types[i]==0):
                Ntrans=ModulesList[i].Ntrans
            for n in range(Ntrans):
                for d in range(dim):
                    if eps0GD[i]>epsmax:
                        eps0GD[i]=epsmax
                    eps=eps0GD[i]
                    print('k={} i={}  n={}  d={} eps={} energy= {}'.format(k,i,n,d,eps,energy))
                    print(X[0])
                    eps1=eps
                    ismax=0
                    X_temp=X.copy()
                    delta=0.1*functional_mod.Module.ModulesList[i].KernelClass.scale
                    X_temp[0][i][n][d]+=delta
                    energy_diff=attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                    print('energy_diff = {}'.format(energy_diff))
                    Reg_mod_diff=functional_mod.ComputeReg(X_temp)
                    der=(energy_diff+Reg_mod_diff-energy-Reg_mod)/delta
                    print('der={}'.format(der))
                    X_temp=X.copy()
                    X_temp[0][i][n][d]-=eps*der
                    energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                    print('energy-temp = {}'.format(energy_temp))
                    if(energy_temp>energy):
                        for ite in range(10):
                            eps*=0.8
                            X_temp=X.copy()
                            X_temp[0][i][n][d]-=eps*der
                            energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                            if (energy_temp<energy_mod):
                                ismax=1
                                eps0GD[i]=eps
                                break
                    else:
                        for ite in range(10):
                            eps*=1.2
                            X_temp=X.copy()
                            X_temp[0][i][n][d]-=eps*der
                            energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                            if (energy_temp>energy_mod):
                                eps/=1.2
                                break
                        eps0GD[i]=eps
                        ismax=1

                    # Now we have 'the best' eps
                    if (ismax==1):
                        X[0][i][n][d]-=eps*der
                        energy=attach_tot(vector_fields_list,X[0],X[1])
                        energy_mod=functional_mod.ComputeReg(X)+attach_tot(vector_fields_list,X[0],X[1])
                        Reg_mod=functional_mod.ComputeReg(X)
#
#%%
        """elif (Types[i]==3 or Types[i]==2 or Types[i]==1):
            if (Types[i]==3):
                Nrot=ModulesList[i].NRotation
            elif (Types[i]==3):
                Nrot=ModulesList[i].NScaling
            elif (Types[i]==1):
                Nrot=ModulesList[i].NAffine
            for n in range(Nrot):
                for d in range(dim):
                    if eps0GD[i]>epsmax:
                        eps0GD[i]=epsmax
                    eps=eps0GD[i]
                    print('k={} i={}  n={}  d={}  energy= {}'.format(k,i,n,d,energy))
                    ismax=0
                    X_temp=X.copy()
                    delta=0.1*functional.Module.ModulesList[i].KernelClass.scale
                    X_temp[0][i][n][d]+=delta
                    energy_diff=functional(X_temp)
                    der=(energy_diff-energy)/delta
                    X_temp=X.copy()
                    X_temp[0][i][n][d]-=eps*der
                    energy_temp=functional(X_temp)
                    if(energy_temp>energy):
                        for ite in range(10):
                            eps*=0.8
                            X_temp=X.copy()
                            X_temp[0][i][n][d]-=eps*der
                            energy_temp=functional(X_temp)
                            if (energy_temp<energy):
                                ismax=1
                                eps0GD[i]=eps
                                break
                    else:
                        for ite in range(10):
                            eps*=1.2
                            X_temp=X.copy()
                            X_temp[0][i][n][d]-=eps*der
                            energy_temp=functional(X_temp)
                            if (energy_temp>energy):
                                eps/=1.2
                                break
                        eps0GD[i]=eps
                        ismax=1

                    # Now we have 'the best' eps
                    if (ismax==1):
                        X[0][i][n][d]-=eps*der
                        energy=functional(X)
#           """



#%% Naive Gradient descent : gradient computed by finite differences
#Combination of modules
#functional=functionalF
niter=10
eps = 0.1
X=functional.domain.element([GD_init,Cont_init].copy())
attachment_term=functional(X)
energy=attachment_term
print(" Initial , attachment term : {}".format(attachment_term))
gradGD=functional.Module.GDspace.element()
gradCont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).element()

d_GD=functional.Module.GDspace.zero()
d_Cont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).zero()
ModulesList=Module.ModulesList
NbMod=len(ModulesList)
delta=10
# 0=SumTranslations, 1=affine, 2=scaling, 3=rotation
Types=[0]
#energy=functional(X)+1
eps0Cont=[0.0001,0.0001]
eps0GD=[0.0001,0.0001]
cont=1
for k in range(niter):

    #Computation of the gradient by finite differences
    for t in range(nb_time_point_int+1):
        for i in range(NbMod):
            if (Types[i]==1):
                NAffine=ModulesList[i].NAffine
                for n in range(NAffine):
                    for d in range(dim+1):
                        for u in range(dim):
                            print('k={}, t={}  i={}  n={}  d={}  u={}  eps = {}  energy= {}'.format(k,t,i,n,d,u,eps,energy))
                            if (eps0Cont[i]>1):
                                eps0Cont[i]=1
                            eps=eps0Cont[i]
                            eps1=eps
                            ismax=0
                            X_temp=X.copy()
                            X_temp[1][t][i][n][d][u]+=delta
                            energy_diff=functional(X_temp)
                            der=(energy_diff-energy)/delta
                            X_temp=X.copy()
                            X_temp[1][t][i][n][d][u]-=eps*der
                            energy_temp=functional(X_temp)
                            if(energy_temp>energy):
                                for ite in range(10):
                                    eps*=0.8
                                    X_temp=X.copy()
                                    X_temp[1][t][i][n][d][u]-=eps*der
                                    energy_temp=functional(X_temp)
                                    if (energy_temp<energy):
                                        ismax=1
                                        eps0Cont[i]=eps
                                        break
                            else:
                                for ite in range(10):
                                    eps*=1.2
                                    X_temp=X.copy()
                                    X_temp[1][t][i][n][d][u]-=eps*der
                                    energy_temp=functional(X_temp)
                                    if (energy_temp>=energy):
                                        eps/=1.2
                                        break
                                eps0Cont[i]=eps
                                ismax=1


                            # Now we have 'the best' eps
                            if (ismax==1):
                                X[1][t][i][n][d][u]-=eps*der
                                energy=functional(X)

            elif (Types[i]==0):
                Ntrans=ModulesList[i].Ntrans
                for n in range(Ntrans):
                    for d in range(dim):
                        print('k={}, t={}  i={}  n={}  d={} eps={}  energy= {}'.format(k,t,i,n,d,eps,energy))
                        if (eps0Cont[i]>1):
                                eps0Cont[i]=1
                        eps=eps0Cont[i]
                        eps1=eps
                        ismax=0
                        X_temp=X.copy()
                        delta=0.1*np.abs(X_temp[1][t][i][n][d])
                        if(delta==0):
                            delta=0.1
                        X_temp[1][t][i][n][d]+=delta
                        energy_diff=functional(X_temp)
                        der=(energy_diff-energy)/delta
                        X_temp=X.copy()
                        X_temp[1][t][i][n][d]-=eps*der
                        energy_temp=functional(X_temp)
                        if(energy_temp>energy):
                            for ite in range(10):
                                eps*=0.8
                                X_temp=X.copy()
                                X_temp[1][t][i][n][d]-=eps*der
                                energy_temp=functional(X_temp)
                                if (energy_temp<energy):
                                    ismax=1
                                    eps0Cont[i]=eps
                                    break
                        else:
                            for ite in range(10):
                                eps*=1.2
                                X_temp=X.copy()
                                X_temp[1][t][i][n][d]-=eps*der
                                energy_temp=functional(X_temp)
                                if (energy_temp>=energy):
                                    eps/=1.2
                                    break
                            eps0Cont[i]=eps
                            ismax=1


                        # Now we have 'the best' eps
                        if (ismax==1):
                            X[1][t][i][n][d]-=eps*der
                            energy=functional(X)



            elif (Types[i]==3 or Types[i]==2):
                if (Types[i]==3):
                    Nrot=ModulesList[i].NRotation
                else:
                      Nrot=ModulesList[i].NScaling
                for n in range(Nrot):
                        print('k={} t={}  i={}  n={}   energy= {}'.format(k,t,i,n,energy))
                        if (eps0Cont[i]>1):
                                eps0Cont[i]=1
                        eps=eps0Cont[i]
                        eps1=eps
                        ismax=0
                        X_temp=X.copy()
                        X_temp[1][t][i][n]+=delta
                        energy_diff=functional(X_temp)
                        der=(energy_diff-energy)/delta
                        X_temp=X.copy()
                        X_temp[1][t][i][n]-=eps*der
                        energy_temp=functional(X_temp)
                        if(energy_temp>energy):
                            for ite in range(10):
                                eps*=0.8
                                X_temp=X.copy()
                                X_temp[1][t][i][n]-=eps*der
                                energy_temp=functional(X_temp)
                                if (energy_temp<energy):
                                    ismax=1
                                    eps0Cont[i]=eps
                                    break
                        else:
                            for ite in range(10):
                                eps*=1.2
                                X_temp=X.copy()
                                X_temp[1][t][i][n]-=eps*der
                                energy_temp=functional(X_temp)
                                if (energy_temp>=energy):
                                    eps/=1.2
                                    break
                            eps0Cont[i]=eps
                            ismax=1


                        # Now we have 'the best' eps
                        if(ismax==1):
                            X[1][t][i][n]-=eps*der
                            energy=functional(X)



    for i in range(NbMod):
        if (Types[i]==0):

            Ntrans=ModulesList[i].Ntrans
            for n in range(Ntrans):
                for d in range(dim):
                    if (eps0GD[i]>1):
                        eps0GD[i]=1
                    eps=eps0GD[i]
                    print('k={} i={}  n={}  d={} eps={} energy= {}'.format(k,i,n,d,eps,energy))
                    eps1=eps
                    ismax=0
                    X_temp=X.copy()
                    delta=0.1*functional.Module.ModulesList[i].KernelClass.scale
                    X_temp[0][i][n][d]+=delta
                    energy_diff=functional(X_temp)
                    der=(energy_diff-energy)/delta
                    X_temp=X.copy()
                    X_temp[0][i][n][d]-=eps*der
                    energy_temp=functional(X_temp)
                    if(energy_temp>energy):
                        for ite in range(10):
                            eps*=0.8
                            X_temp=X.copy()
                            X_temp[0][i][n][d]-=eps*der
                            energy_temp=functional(X_temp)
                            if (energy_temp<energy):
                                ismax=1
                                eps0GD[i]=eps
                                break
                    else:
                        for ite in range(10):
                            eps*=1.2
                            X_temp=X.copy()
                            X_temp[0][i][n][d]-=eps*der
                            energy_temp=functional(X_temp)
                            if (energy_temp>energy):
                                eps/=1.2
                                break
                        eps0GD[i]=eps
                        ismax=1

                    # Now we have 'the best' eps
                    if (ismax==1):
                        X[0][i][n][d]-=eps*der
                        energy=functional(X)

        elif (Types[i]==3 or Types[i]==2 or Types[i]==1):
            if (Types[i]==3):
                Nrot=ModulesList[i].NRotation
            elif (Types[i]==3):
                Nrot=ModulesList[i].NScaling
            elif (Types[i]==1):
                Nrot=ModulesList[i].NAffine
            for n in range(Nrot):
                for d in range(dim):
                    print('k={} i={}  n={}  d={}  energy= {}'.format(k,i,n,d,energy))
                    eps=eps0GD[i]
                    eps1=eps
                    ismax=0
                    X_temp=X.copy()

                    X_temp[0][i][n][d]+=delta
                    energy_diff=functional(X_temp)
                    der=(energy_diff-energy)/delta
                    X_temp=X.copy()
                    X_temp[0][i][n][d]-=eps*der
                    energy_temp=functional(X_temp)
                    if(energy_temp>energy):
                        for ite in range(10):
                            eps*=0.8
                            X_temp=X.copy()
                            X_temp[0][i][n][d]-=eps*der
                            energy_temp=functional(X_temp)
                            if (energy_temp<energy):
                                ismax=1
                                eps0GD[i]=eps
                                break
                    else:
                        for ite in range(10):
                            eps*=1.2
                            X_temp=X.copy()
                            X_temp[0][i][n][d]-=eps*der
                            energy_temp=functional(X_temp)
                            if (energy_temp>energy):
                                eps/=1.2
                                break
                        eps0GD[i]=eps
                        ismax=1

                    # Now we have 'the best' eps
                    if (ismax==1):
                        X[0][i][n][d]-=eps*der
                        energy=functional(X)











#%%  see result
I_t=Shoot_mixt(vector_fields_list,X[0],X[1])

vector_fields_list_tot=Mix_vect_field(vector_fields_list,X[0],X[1])
grid_points=compute_grid_deformation_list(vector_fields_list_tot, 1/nb_time_point_int, template.space.points().T)

for t in range(nb_time_point_int+1):
    I_t[t].show('time {}'.format(t+1))
    #grid=grid_points[t].reshape(2, space.shape[0], space.shape[1]).copy()
    #plot_grid(grid, 2)
#

#%% See the non modular part

I_t=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vector_fields_list, template)

for t in range(nb_time_point_int+1):
    I_t[t].show('time {}'.format(t+1))
#
#%% See modular part
vect_field_mod_list=ComputeModularVectorFields(vector_fields_list,X[0],X[1])
vect_field_mod_inter=odl.ProductSpace(template.space.tangent_bundle,nb_time_point_int +1).element([ [vect_field_mod_list[i][u].interpolation(template.space.points().T) for u in range(dim)] for i in range(nb_time_point_int+1)])
I_t=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_mod_inter, template)

for t in range(nb_time_point_int+1):
    I_t[t].show('time {}'.format(t+1))
#

#%% See deformation

GD_t=functional.ComputetrajectoryGD(X)

vect_field_list=odl.ProductSpace(functional.Module.DomainField.tangent_bundle,nb_time_point_int).element()

for i in range(nb_time_point_int):
    vect_field_list[i]=functional.Module.ComputeField(GD_t[i],X[1][i]).copy()

I_t=functional.Shoot(X)
#grid_points=compute_grid_deformation_list(vect_field_list, 1/nb_time_point_int, I0.space.points().T)
DirRot=functional.Module.ModulesList[1].DirectionsVec
for t in range(nb_time_point_int+1):
    I_t[t].show('t= {}'.format(t))
#    grid=grid_points[t].reshape(2, 128, 128).copy()
#    plot_grid(grid, 2)
    TP=functional.Module.ModulesList[1].ComputeToolPoints(GD_t[t][1])
    for u in range(Ntrans):
        plt.plot(GD_t[t][0][u][0], GD_t[t][0][u][1],'xb')
        plt.quiver(GD_t[t][0][u][0], GD_t[t][0][u][1],X[1][t][0][u][0],X[1][t][0][u][1],color='g')
#    for u in range(NRotation):
#        plt.plot(GD_t[t][1][u][0], GD_t[t][1][u][1],'ob')
#        for v in range(dim+1):
#            plt.quiver(TP[u][v][0],TP[u][v][1],X[1][t][1][u]*DirRot[v][0],X[1][t][1][u]*DirRot[v][1],color='g')
#%%


def compute_grid_deformation(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size

    grid_points = initial_grid

    for t in range(nb_time_points):
        velocity = np.empty_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            velocity[i, ...] = vi.interpolation(grid_points)
        grid_points += time_step*velocity

    return grid_points


def compute_grid_deformation_list(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size
    grid_list=[]
    grid_points=initial_grid.copy()
    grid_list.append(initial_grid)

    for t in range(nb_time_points):
        velocity = np.empty_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            velocity[i, ...] = vi.interpolation(grid_points)
        grid_points += time_step*velocity
        grid_list.append(grid_points.copy())

    return grid_list


def plot_grid(grid, skip):
    for i in range(0, grid.shape[1], skip):
        plt.plot(grid[0, i, :], grid[1, i, :], 'r', linewidth=0.5)

    for i in range(0, grid.shape[2], skip):
        plt.plot(grid[0, :, i], grid[1, :, i], 'r', linewidth=0.5)




#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#DomainField=space
#
#GDspace=odl.ProductSpace(odl.space.rn(dim),Ntrans)
#Contspace=odl.ProductSpace(odl.space.rn(dim),Ntrans)
#
#vector_field=translation.ComputeField([[0,0],[0,1]],[[0,1],[0,2]])
#eva=translation.ComputeFieldevaluate([[0,0],[0,1]],[[0,1],[0,2]])
#eva(odl.space.rn(dim).element([0,0]))
#
#vector_field=translation.ComputeFieldDer([[-10,-10],[10,10]],[[0,1],[0,1]])([[1,0],[1,0]])
##vector_field.show()
#
#speed=translation.ComputeFieldDerevaluate([[-1,-1],[1,1]],[[0,1],[0,1]])([[[0,1],[0,1]],[0,1]])
#
#X=odl.ProductSpace(GDspace,DomainField.tangent_bundle).element([[[0,0],[0,1]],vector_field])
#
#speed=translation.ApplyVectorField(X[0],X[1])
#
#GD=GDspace.element([[-10,-10],[10,10]])
#Cont=Contspace.element([[0,1],[0,1]])
#appli=translation.ApplyModule(translation,GD,Cont)
#appli(GD)
#
#energy=translation.Cost(GD,Cont)
#energy=translation.CostDerGD(GD,Cont)(GD)
#energy=translation.CostDerCont(GD,Cont)(Cont)
#
#
#mod=[translation,translation,translation]
#GDspace=odl.ProductSpace(*[mod[i].GDspace for i in range(len(mod))])
#mod_new=Compound(mod)
#vector_field=mod_new.ComputeField(GD,Cont)
#eva=mod_new.ComputeFieldEvaluate(GD,Cont)([0,0])
#vector_field=mod_new.ComputeFieldDer(GD,Cont)(GD)
#eva=mod_new.ComputeFieldDerEvaluate(GD,Cont)([GD,[0,0]])
#appli=mod_new.ApplyVectorField(GD,vector_field)
#
#energy=mod_new.Cost(GD,Cont)
#energy=mod_new.CostDerGD(GD,Cont)(GD)
#energy=mod_new.CostDerCont(GD,Cont)(Cont)
#
#
#o=GDspace.element([[0,0]])
#h=Contspace.element([[0,5]])
#vector_field=DomainField.tangent_bundle.zero()
#Kernel=kernel
#
#
#mg = DomainField.meshgrid
#for i in range(Ntrans):
#    kern = Kernel([mgu - ou for mgu, ou in zip(mg, o[i])])
#    vector_field += DomainField.tangent_bundle.element([kern * hu for hu in h[i]])
#
#
#
#x=odl.space.rn(dim).element([0,0])
#speed=odl.space.rn(dim).zero()
#for i in range(Ntrans):
#    a=Kernel(o[i]-x)
#    speed+=a*h[i]
#
#
#
#
#class GaussianKernel(object):
#    def __init__(self,scale):
#        self.scale=scale
#
#    def Eval(self,x):
#        scaled = [xi ** 2 / (2 * self.scale ** 2) for xi in x]
#        return np.exp(-sum(scaled))
#
#    @property
#    def derivative(self):
#        ker=self
#        class ComputeDer(object):
#            def __init__(self,x0):
#                self.x0=x0
#            def Eval(self,dx):
#                a=ker.Eval(self.x0)
#                b=[-xi*dxi/( ker.scale ** 2) for xi, dxi in zip(self.x0,dx) ]
#                return a*sum(b)
#        return ComputeDer
#
#
#Kernel=GaussianKernel(2)
#Kernel.Eval([0,0])
#Kernel.derivative([0,0]).Eval([0,0])
#
#KernelEval=Kernel.Eval
#
#
#
#
#
#
#%%

template=I0
# Create a product space for displacement field
disp_field_space = space.tangent_bundle

# Define a displacement field that bends the template a bit towards the
# upper left. We use a list of 2 functions and discretize it using the
# disp_field_space.element() method.
sigma = 2
h=[10,0]
disp_func = [
    lambda x: h[0]* np.exp(-(x[0] ** 2 + x[1] ** 2) / (2 * sigma ** 2)),
    lambda x: h[1] * np.exp(-(x[0] ** 2 + x[1] ** 2) / (2 * sigma ** 2))]

disp_field = disp_field_space.element(disp_func)


# Initialize the deformation operator with fixed template
deform_op = odl.deform.LinDeformFixedTempl(template)

# Apply the deformation operator to get the deformed template.
deformed_template = deform_op(disp_field)

template.show()
deformed_template.show()
proj_data = forward_op(deformed_template)
data_image=[deformed_template]