#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:49:20 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:42:31 2017

@author: barbara
"""

"""
Gradient descent when the module is FromFile
different step size for centre and theta
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
from DeformationModulesODL.deform import FromFile
from DeformationModulesODL.deform import FromFileV5
from DeformationModulesODL.deform import TemporalAttachmentModulesGeom
from DeformationModulesODL.deform import TemporalAttachmentModulesGeomAtlas

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
#
##%%

space_mod = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[128, 128],
    dtype='float32', interp='linear')

#%% Define Module
scale_rotation=5
kernelrot=Kernel.GaussianKernel(scale_rotation)
rotation=LocalRotation.LocalRotation(space_mod, 1, kernelrot)

#Module=DeformationModuleAbstract.Compound([translation,rotation])
Module=DeformationModuleAbstract.Compound([rotation])


#%% Define data
space=space_mod
nb_data=10

#define template
template=space.zero()
points=space.points()


for i in range(len(points)):
    pt=points[i]
    if(pt[1]> -3 and (3*pt[1])<(3.0*pt[0]+3) and (3*pt[1])<(-3.0*pt[0]+3) ):
        template[i]=1


#template=space.element(scipy.ndimage.filters.gaussian_filter(template.asarray(),fac_smooth))
#template.show()


def Rtheta(theta,points):
    # theta is the angle, in rad
    # input = list of points, for ex given by space.points() or
    # np.array(vect_field).T
    #output = list of points of same size, rotated of an angle theta

    points_rot=np.empty_like(points).T
    points_rot[0]=np.cos(theta)*points.T[0].copy() - np.sin(theta)*points.T[1].copy()
    points_rot[1]=np.sin(theta)*points.T[0].copy() + np.cos(theta)*points.T[1].copy()

    return points_rot.T.copy()
    points.T[0]

mini=-10
maxi=10
def Rot_image(theta,centre,template):
    I1=space.element()
    for i in range(len(points)):
        pt=points[i]
        if(pt[0]>mini and pt[0] < maxi and pt[1]>mini and pt[1]<maxi):
            pt_rot_inv=Rtheta(-theta,pt-centre).copy()
            I1[i]=template.interpolation([[pt_rot_inv[0]+centre[0]],[pt_rot_inv[1]+centre[1]]])
        else:
            I1[i]=template[i]

    return I1

fac_smooth=2
if False:
    theta_list=[0 , 10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    centre=[0,0]
    data_list=[]
    for i in range(nb_data):
        data_list.append(space.element(scipy.ndimage.filters.gaussian_filter(Rot_image(theta_list[i],centre,template).asarray(),fac_smooth)).copy())
    #I1.show()

    for i in range(nb_data):
        np.savetxt('/home/bgris/data/triangles/image_{}'.format(i),data_list[i])

    for i in range(nb_data):
        data_list[i].show('{}'.format(i))
#

data_list=[]
for i in range(nb_data):
    data_list.append(space.element(scipy.ndimage.filters.gaussian_filter(np.loadtxt('/home/bgris/data/triangles/image_{}'.format(i)),fac_smooth)).copy())
#

#%% Define functional
lam=0.0001
nb_time_point_int=10
forward_op=odl.IdentityOperator(space)
lamb0=1e-5
lamb1=1e-5
import scipy
data_time_points=np.array([1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
forward_operators_list=[]
for i in range(nb_data):
    forward_operators_list.append([forward_op])



Norm=odl.solvers.L2NormSquared(forward_op.range)

# The parameter for kernel function
sigma = scale_rotation

# Give kernel function
def kernel_lddmm(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

domainlddmmfunc=odl.ProductSpace(space.real_space.tangent_bundle,nb_time_point_int+1)

functional_mod_list=[]
functional_lddmm_list=[]

for i in range(nb_data):
    functional_mod_list.append(TemporalAttachmentModulesGeomAtlas.FunctionalModulesGeomAtlas(lamb0, nb_time_point_int, [data_list[i]], data_time_points, forward_operators_list[i],Norm, Module))
    functional_lddmm_list.append(odl.deform.TemporalAttachmentLDDMMGeomAtlas(nb_time_point_int, [data_list[i]],
                            data_time_points, forward_operators_list[i],Norm, kernel_lddmm,
                            domain=None))
#
#
##%%
#def Mix_vect_field(vect_field_list,GD_init,Cont):
#    space_pts=template.space.points()
#    GD=GD_init.copy()
#    vect_field_list_tot=vect_field_list.copy()
#    for i in range(nb_time_point_int):
#        vect_field_list_mod=functional_mod.Module.ComputeField(GD,Cont[i]).copy()
#        vect_field_list_mod_interp=template.space.tangent_bundle.element([vect_field_list_mod[u].interpolation(space_pts.T) for u in range(dim)]).copy()
#        vect_field_list_tot[i]+=vect_field_list_mod_interp.copy()
#        GD+=(1/nb_time_point_int)*functional_mod.Module.ApplyVectorField(GD,vect_field_list_tot[i]).copy()
#
#    return vect_field_list_tot
#
#def Shoot_mixt(vect_field_list,GD,Cont):
#    vect_field_list_mod=Mix_vect_field(vect_field_list,GD,Cont).copy()
#    I=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_list_mod, template)
#    return I
#
#def attach_tot(vect_field_list,GD,Cont):
#    I=Shoot_mixt(vect_field_list,GD,Cont)
#    return Norm(forward_operators[0](I[nb_time_point_int])-data[0])
#
#
#
#def grad_attach_vector_field(GD,Cont):
#    vect_field_list_tot=vect_field_list(GD,Cont)
#    grad=energy_op_lddmm.gradient(vect_field_list_tot).copy()
#    return grad
##
#
#def grad_attach_vector_fieldL2(GD,Cont):
#    vect_field_list_tot=vect_field_list(GD,Cont)
#    grad=energy_op_lddmm.gradientL2(vect_field_list_tot).copy()
#    return grad
##
#
#mini=0
#maxi=1
#rec_space=template.space
#time_itvs=nb_time_point_int
#
#
#
#def plot_result(name,image_N0):
#    plt.figure()
#    rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
#    rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
#    rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
#    rec_result = rec_space.element(image_N0[time_itvs])
#    ##%%
#    # Plot the results of interest
#    plt.figure(2, figsize=(24, 24))
#    #plt.clf()
#
#    plt.subplot(3, 3, 1)
#    plt.imshow(np.rot90(template), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.axis('off')
#    #plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
#    plt.colorbar()
#    #plt.title('Trajectory from EllipseMvt with DeformationModulesODL/deform/vect_field_ellipse')
#
#    plt.subplot(3, 3, 2)
#    plt.imshow(np.rot90(rec_result_1), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4))
#
#    plt.subplot(3, 3, 3)
#    plt.imshow(np.rot90(rec_result_2), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    #grid=grid_points[time_itvs // 4].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    #plot_grid(grid, 2)
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))
#
#    plt.subplot(3, 3, 4)
#    plt.imshow(np.rot90(rec_result_3), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    #grid=grid_points[time_itvs // 4*2].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    #plot_grid(grid, 2)
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))
#
#    plt.subplot(3, 3, 5)
#    plt.imshow(np.rot90(rec_result), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs ))
#
#    plt.subplot(3, 3, 6)
#    plt.imshow(np.rot90(ground_truth[0]), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('Ground truth')
#
#
#
#    plt.savefig(name, bbox_inches='tight')
#
#
#def plot_result_moduleV5(name,image_N0,GD_list):
#    plt.figure()
#    rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
#    rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
#    rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
#    rec_result = rec_space.element(image_N0[time_itvs])
#    ##%%
#    # Plot the results of interest
#    plt.figure(2, figsize=(24, 24))
#    #plt.clf()
#
#    plt.subplot(3, 3, 1)
#    plt.imshow(np.rot90(template), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.axis('off')
#    #plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
#    plt.colorbar()
#    #plt.title('Trajectory from EllipseMvt with DeformationModulesODL/deform/vect_field_ellipse')
#
#    plt.subplot(3, 3, 2)
#    plt.imshow(np.rot90(rec_result_1), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.plot(GD_list[time_itvs // 4][0][2][0],GD_list[time_itvs // 4][0][2][1],'o')
#    plt.plot(GD_list[time_itvs // 4][0][3][0],GD_list[time_itvs // 4][0][3][1],'x')
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4))
#
#    plt.subplot(3, 3, 3)
#    plt.imshow(np.rot90(rec_result_2), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.plot(GD_list[time_itvs // 4 *2][0][2][0],GD_list[time_itvs // 4][0][2][1],'o')
#    plt.plot(GD_list[time_itvs // 4 *2][0][3][0],GD_list[time_itvs // 4][0][3][1],'x')
#    #grid=grid_points[time_itvs // 4].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    #plot_grid(grid, 2)
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))
#
#    plt.subplot(3, 3, 4)
#    plt.imshow(np.rot90(rec_result_3), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.plot(GD_list[time_itvs // 4 *3][0][2][0],GD_list[time_itvs // 4][0][2][1],'o')
#    plt.plot(GD_list[time_itvs // 4 *3][0][3][0],GD_list[time_itvs // 4][0][3][1],'x')
#    #grid=grid_points[time_itvs // 4*2].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    #plot_grid(grid, 2)
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))
#
#    plt.subplot(3, 3, 5)
#    plt.imshow(np.rot90(rec_result), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.plot(GD_list[time_itvs ][0][2][0],GD_list[time_itvs // 4][0][2][1],'o')
#    plt.plot(GD_list[time_itvs  ][0][3][0],GD_list[time_itvs // 4][0][3][1],'x')
#
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))
#
#    plt.subplot(3, 3, 6)
#    plt.imshow(np.rot90(ground_truth[0]), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('Ground truth')
#
#
#
#    plt.savefig(name, bbox_inches='tight')
##
#%%
dim=2

def vect_field_list(GD_init,Cont):
    space_pts=space.points()
    GD=GD_init.copy()
    vect_field_list_tot=[]
    for i in range(nb_time_point_int+1):
        vect_field_mod=functional_mod_list[0].Module.ComputeField(GD,Cont[i]).copy()
        vect_field_list_interp=space.tangent_bundle.element([vect_field_mod[u].interpolation(space_pts.T) for u in range(dim)]).copy()
        GD+=(1/nb_time_point_int)*functional_mod_list[0].Module.ApplyVectorField(GD,vect_field_list_interp).copy()
        vect_field_list_tot.append(vect_field_list_interp)

    return odl.ProductSpace(template.space.tangent_bundle,nb_time_point_int+1).element(vect_field_list_tot)




def attach_mod(GD,Cont,template,i):
    vect_field_list_mod=vect_field_list(GD,Cont)
    I=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_list_mod, template).copy()
    return Norm(forward_operators_list[i][0](I[nb_time_point_int])-data_list[i])
#



def ComputeGD_list(GD,Cont):
    GD_list=[]
    GD_list.append(GD.copy())
    for i in range(nb_time_point_int):
        GD_temp=GD_list[i].copy()
        vect_field=functional_mod_list[0].Module.ComputeField(GD_temp,Cont[i]).copy()
        GD_temp+=(1/nb_time_point_int)*functional_mod_list[0].Module.ApplyVectorField(GD_temp,vect_field)
        GD_list.append(GD_temp.copy())

    return GD_list.copy()



def grad_attach_vector_fieldL2(GD,Cont,template,i):
    vect_field_list_tot=vect_field_list(GD,Cont)
    grad=functional_lddmm_list[i].gradientL2([template,vect_field_list_tot]).copy()
    return grad
#


#

#%%

niter=500
eps = 0.01

Cont_list=[]
for i in range(nb_data):
    Cont_list.append(odl.ProductSpace(Module.Contspace,nb_time_point_int+1).zero())

X_list=[space.zero(),Module.GDspace.element([[[0,0]]]), Cont_list]


##%%
dim=2
attachment_term=0
for i in range(nb_data):
    attachment_term+=attach_mod(X_list[1],X_list[2][i],X_list[0],i)


print(" Initial , attachment term : {}".format(attachment_term))

ModulesList=Module.ModulesList
NbMod=len(ModulesList)
delta=0.1
epsmax=0.5
deltamin=0.1
deltaCont=0.1
deltaGD=0.2
# 0=SumTranslations, 1=affine, 2=scaling, 3=rotation, 4 = ellipsemvt
Types=[3]
#energy=functional(X)+1
inv_N=1/nb_time_point_int
epsContmax=1
epsGDmax=1
epsCont=0.1
epsGD=0.0
epstemplate=0.01

eps_vect_field=0.1
cont=1
space_pts=template.space.points()


#%%
import copy
energy=attachment_term
attachment_term=0
for i in range(nb_data):
    attachment_term+=attach_mod(X_list[1],X_list[2][i],X_list[0],i)

print('Initial  energy = {} '.format(attachment_term))
for k in range(niter):
    gradGD=Module.GDspace.zero()
    gradCont=odl.ProductSpace(odl.ProductSpace(Module.Contspace,nb_time_point_int+1),nb_data).zero()
    gradtemplate=space.zero()

    for idata in range(nb_data):

        GD_idata=ComputeGD_list(X_list[1],X_list[2][idata]).copy()
        gradlddmm_idata=grad_attach_vector_fieldL2(X_list[1],X_list[2][idata],X_list[0],idata)
        grad_template_idata=gradlddmm_idata[0]
        grad_vect_field_idata=gradlddmm_idata[1]
        gradtemplate+=grad_template_idata.copy()

        for i in range(NbMod):

            basisCont=ModulesList[i].basisCont.copy()
            dimCont=len(basisCont)
            #print('i = {}'.format(i))
            for iter_cont in range(dimCont):
                for t in range(nb_time_point_int):
                    GD_idata_temp=copy.deepcopy(GD_idata)
                    Cont_idata=copy.deepcopy(X_list[2][idata])
                    Cont_idata[t][i]+=deltaCont*basisCont[iter_cont]
                    GD_diff=ComputeGD_list(X_list[1],Cont_idata).copy()
                    temp=0
                    #print('t = {}'.format(t))
                    for s in range(nb_time_point_int):
                        # vec_temp is the derivative of the generated vector field at s with respect to h[t][iter_cont]
                        vec_temp=( Module.ComputeField(GD_diff[s], Cont_idata[s]).copy()-Module.ComputeField(GD_idata[s], X_list[2][idata][s]).copy() )/deltaCont

                        # It is necessary to interpolate in order to do the inner product
                        vec_temp_interp=space.tangent_bundle.element(vec_temp).copy()

                        temp+=inv_N*grad_vect_field_idata[s].inner(vec_temp_interp)
                        #print('s = {}'.format(s))

                    #ATTENTION : WE SUPPOSE THAT THE DERIVATIVE OF THE COST
                    # WITH RESPECT TO GD IS NULL
                    # Not here because we want to minimize only attachment term
                    #temp+=lamb0*Module.CostGradCont(GD[t], X[1][t])[iter_cont]
                    gradCont[idata][t][i][iter_cont]+=temp
        for i in range(NbMod):
            basisGD=ModulesList[i].basisGD.copy()
            dimGD=len(basisGD)

            for iter_gd in range(dimGD):
                GD_init_idata=copy.deepcopy(X_list[1])
                GD_init_idata[i]+=deltaGD*basisGD[iter_gd]
                GD_diff_idata=ComputeGD_list(GD_init_idata,X_list[2][idata]).copy()
                temp=0
                for s in range(nb_time_point_int):
                    # vec_temp is the derivative of the generated vector field at s with respect to GD[iter_cont]
                    vec_temp=( Module.ComputeField(GD_diff_idata[s], X_list[2][idata][s]).copy()-Module.ComputeField(GD_idata[s], X_list[2][idata][s]).copy() )/deltaGD

                    # It is necessary to interpolate in order to do the inner product
                    vec_temp_interp=space.tangent_bundle.element(vec_temp).copy()
                            #[vec_temp[u].interpolation(space_pts.T) for u in range(dim)]).copy()

                    temp+=inv_N*grad_vect_field_idata[s].inner(vec_temp_interp)
                gradGD[i]+=temp*basisGD[iter_gd]

    for ite in range(20):
        X_list_temp=copy.deepcopy(X_list)
        X_list_temp[0]-=epstemplate*gradtemplate
        X_list_temp[1]-=epsGD*gradGD
        X_list_temp[2]-=epsCont*gradCont
        energy_mod0=0
        for idata in range(nb_data):
            energy_mod0+=attach_mod(X_list_temp[1],X_list_temp[2][idata],X_list_temp[0],idata)

        X_list_temp=copy.deepcopy(X_list)
        X_list_temp[0]-=0.8*epstemplate*gradtemplate
        X_list_temp[1]-=epsGD*gradGD
        X_list_temp[2]-=epsCont*gradCont
        energy_mod1=0
        for idata in range(nb_data):
            energy_mod1+=attach_mod(X_list_temp[1],X_list_temp[2][idata],X_list_temp[0],idata)

        X_list_temp=copy.deepcopy(X_list)
        X_list_temp[0]-=epstemplate*gradtemplate
        X_list_temp[1]-=0.8*epsGD*gradGD
        X_list_temp[2]-=epsCont*gradCont
        energy_mod2=0
        for idata in range(nb_data):
            energy_mod2+=attach_mod(X_list_temp[1],X_list_temp[2][idata],X_list_temp[0],idata)

        X_list_temp=copy.deepcopy(X_list)
        X_list_temp[0]-=epstemplate*gradtemplate
        X_list_temp[1]-=epsGD*gradGD
        X_list_temp[2]-=0.8*epsCont*gradCont
        energy_mod3=0
        for idata in range(nb_data):
            energy_mod3+=attach_mod(X_list_temp[1],X_list_temp[2][idata],X_list_temp[0],idata)



        print('energy0 = {}, energy1 = {}, energy2 = {}, energy3 = {} '.format(energy_mod0,energy_mod1,energy_mod2,energy_mod3) )
        if (energy_mod0 <= energy_mod1 and energy_mod0 <= energy_mod2 and energy_mod0 <= energy_mod3):
            X_list_temp=copy.deepcopy(X_list)
            X_list_temp[0]-=epstemplate*gradtemplate
            X_list_temp[1]-=epsGD*gradGD
            X_list_temp[2]-=epsCont*gradCont
            energy_temp=energy_mod0
        elif (energy_mod1 <= energy_mod0 and energy_mod1 <= energy_mod2 and energy_mod1 <= energy_mod3 ):
            X_list_temp=copy.deepcopy(X_list)
            X_list_temp[0]-=0.8*epstemplate*gradtemplate
            X_list_temp[1]-=epsGD*gradGD
            X_list_temp[2]-=epsCont*gradCont
            energy_temp=energy_mod1
            epstemplate*=0.8
        elif (energy_mod2 <= energy_mod0 and energy_mod2 <= energy_mod1 and energy_mod2 <= energy_mod3 ):
            X_list_temp=copy.deepcopy(X_list)
            X_list_temp[0]-=epstemplate*gradtemplate
            X_list_temp[1]-=0.8*epsGD*gradGD
            X_list_temp[2]-=epsCont*gradCont
            energy_temp=energy_mod2
            epsGD*=0.8
        else:
            X_list_temp=copy.deepcopy(X_list)
            X_list_temp[0]-=epstemplate*gradtemplate
            X_list_temp[1]-=epsGD*gradGD
            X_list_temp[2]-=0.8*epsCont*gradCont
            energy_temp=energy_mod3
            epsCont*=0.8



        if (energy_temp < energy):
            X_list=copy.deepcopy(X_list_temp)
            energy=energy_temp
            print('k={} , energy = {} , ite={}'.format(k,energy_temp,ite))
            #print('GD =  {}'.format(X[0]))
            epsCont*=1.2
            epsGD*=1.2
            epstemplate*=1.2
            break
        else:
           epsCont*=0.8
           epsGD*=0.8
           epstemplate*=0.8

    if (ite==19):
        print('No possible to descent')
        break

    print('epsGD= {} ,epsCont= {},  epsGDtemplate ={}'.format(epsGD,epsCont,epstemplate))
    #vector_fields_list=((1-eps_vect_field*lamb1)*vector_fields_list-eps_vect_field*grad_vect_field ).copy()
#

#%%
#I_t=Shoot_mixt(vector_fields_list,X[0],X[1])
vect_field_list_mod=vect_field_list(X[0],X[1])
I_t=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_list_mod, template)

name='/home/bgris/Results/DeformationModules/testEstimation/Rotation/EstimatedTrajectory_FromFileV6_from_vect_field_V5_Rotation_SheppLogan'
#name+= '_sigma_2_INDIRECT_num_angle_10_initial_epsCont_0_01_fixed_GD'
name+= '_sigma_lddmm_1_sigma_cp_0_5_nbtrans_30_expectedvalue_not_optimized_GD'
image_N0=I_t
plot_result(name,image_N0)


#%%
GD_list=ComputeGD_list(X[0],X[1])

points=space.points()

for k in range(nb_time_point_int):
    v0=Module.ComputeField(GD_list[k],X[1][k])
    plt.figure()
    plt.quiver(points.T[0][::20],points.T[1][::20],v0[0][::20],v0[1][::20])
    plt.axis('equal')
    plt.title('Vector field at time {}'.format(k))
#
name_mod=name+'plotGD'
plot_result_moduleV5(name_mod,image_N0,GD_list)

for i in range(nb_time_point_int+1):
    #vect_field_list_mod[i][0].show('{}'.format(i))
    I_t[i].show('{}'.format(i))
    plt.plot(GD_list[i][0][2][0],GD_list[i][0][2][1],'o')
    plt.plot(GD_list[i][0][3][0],GD_list[i][0][3][1],'x')

#    plt.plot(GD_init[0][2][0],GD_init[0][2][1],'o')
#    plt.plot(GD_init[0][3][0],GD_init[0][3][1],'x')
