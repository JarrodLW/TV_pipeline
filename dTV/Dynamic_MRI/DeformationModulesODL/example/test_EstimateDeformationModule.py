#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:15:37 2017

@author: barbara
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
#
#%%

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

forward_op=odl.IdentityOperator(space)
proj_data = forward_op(I1)


space_mod = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256],
    dtype='float32', interp='nearest')

#%% Define Module
NEllipse=1

kernelelli=Kernel.GaussianKernel(2)
Name='DeformationModulesODL/deform/vect_field_ellipses'

#elli=FromFile.FromFile(space_mod, Name, kernelelli)
elli=EllipseMvt.EllipseMvt(space_mod, Name, kernelelli)

#Module=DeformationModuleAbstract.Compound([translation,rotation])
Module=DeformationModuleAbstract.Compound([elli])

#GD_init=Module.GDspace.zero()
GD_init=Module.GDspace.element([[[-0.2, 0.4],0]])
Cont_init=Module.Contspace.zero()

#Module.ComputeFieldDer(GD_init, Cont_init)(GD_init)
#%% Define functional
lam=0.0001
nb_time_point_int=10

lamb0=1e-7
lamb1=1e-2
import scipy
data_time_points=np.array([1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
data=data_space.element([proj_data])
forward_operators=[forward_op]


Norm=odl.solvers.L2NormSquared(forward_op.range)

functional_mod = TemporalAttachmentModulesGeom.FunctionalModulesGeom(lamb0, nb_time_point_int, template00, data, data_time_points, forward_operators,Norm, Module)



# The parameter for kernel function
sigma = 2

# Give kernel function
def kernel_lddmm(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


# Define energy operator
energy_op_lddmm=odl.deform.TemporalAttachmentLDDMMGeom(nb_time_point_int, template00 ,data,
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

def grad_attach_vector_fieldL2(vect_field_list,GD,Cont):
    vect_field_list_tot=Mix_vect_field(vect_field_list,GD,Cont)
    grad=energy_op_lddmm.gradientL2(vect_field_list_tot).copy()
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



#%%
GD_init=Module.GDspace.zero()
#GD_init=Module.GDspace.element([[0, 0]])
Cont_init=odl.ProductSpace(Module.Contspace,nb_time_point_int+1).zero()

niter=500
eps = 0.01

X=functional_mod.domain.element([GD_init,Cont_init].copy())

vector_fields_list_init=energy_op_lddmm.domain.zero()
vector_fields_list=vector_fields_list_init.copy()
#%%
dim=2
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
deltaCont=0.1
deltaGD=0.2
# 0=SumTranslations, 1=affine, 2=scaling, 3=rotation, 4 = ellipsemvt
Types=[4]
#energy=functional(X)+1
inv_N=1/nb_time_point_int
epsContmax=1
epsGDmax=1
epsCont=0.01
epsGD=0.00
eps_vect_field=0.01
cont=1
space_pts=template.space.points()

GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()

energy=attach_tot(vector_fields_list,X[0],X[1])
Reg_mod=functional_mod.ComputeReg(X)
energy_mod=Reg_mod+energy
print('Initial  energy = {} '.format(energy_mod))
for k in range(niter):
    gradGD=functional_mod.Module.GDspace.zero()
    gradCont=odl.ProductSpace(functional_mod.Module.Contspace,nb_time_point_int+1).zero()

    #energy=attach_tot(vector_fields_list,X[0],X[1])
    #Reg_mod=functional_mod.ComputeReg(X)
    #energy_mod=Reg_mod+energy
    GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()
    #print('k={}  before vect field attachment term = {}, reg_mod={}'.format(k,energy,Reg_mod))
    # gradient with respect to vector field
    grad_vect_field=grad_attach_vector_field(vector_fields_list,X[0],X[1])
    # (1-lamb1) because of the gradient of the regularity term

    #GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()

    #energy=attach_tot(vector_fields_list,X[0],X[1])
    #Reg_mod=functional_mod.ComputeReg(X)
    #energy_mod=Reg_mod+energy
    #print('      k={}  after vect field  attachment term = {}, reg_mod={}'.format(k,energy,Reg_mod))

    for i in range(NbMod):

        basisCont=ModulesList[i].basisCont.copy()
        dimCont=len(basisCont)
        #print('i = {}'.format(i))
        for iter_cont in range(dimCont):
            for t in range(nb_time_point_int):
                X_temp=X.copy()
                X_temp[1][t][i]+=deltaCont*basisCont[iter_cont]
                GD_diff=ComputeGD_mixt(vector_fields_list,X[0],X_temp[1]).copy()
                temp=0
                #print('t = {}'.format(t))
                for s in range(nb_time_point_int):
                    # vec_temp is the derivative of the generated vector field at s with respect to h[t][iter_cont]
                    vec_temp=( Module.ComputeField(GD_diff[s], X_temp[1][s]).copy()-Module.ComputeField(GD[s], X[1][s]).copy() )/deltaCont
                    
                    # It is necessary to interpolate in order to do the inner product
                    vec_temp_interp=space.tangent_bundle.element(vec_temp).copy()
                    
                    temp+=inv_N*grad_vect_field[s].inner(vec_temp_interp)
                    #print('s = {}'.format(s))
     
                #ATTENTION : WE SUPPOSE THAT THE DERIVATIVE OF THE COST
                # WITH RESPECT TO GD IS NULL
                temp+=Module.CostGradCont(GD[t], X[1][t])[iter_cont]
                gradCont[t][i][iter_cont]+=temp
    for i in range(NbMod):
        basisGD=ModulesList[i].basisGD.copy()
        dimGD=len(basisGD)

        for iter_gd in range(dimGD):
            X_temp=X.copy()
            X_temp[0][i]+=deltaGD*basisGD[iter_gd]
            GD_diff=ComputeGD_mixt(vector_fields_list,X_temp[0],X_temp[1]).copy()
            temp=0
            for s in range(nb_time_point_int):
                # vec_temp is the derivative of the generated vector field at s with respect to GD[iter_cont]
                vec_temp=( Module.ComputeField(GD_diff[s], X_temp[1][s]).copy()-Module.ComputeField(GD[s], X[1][s]).copy() )/deltaGD
                 
                # It is necessary to interpolate in order to do the inner product
                vec_temp_interp=space.tangent_bundle.element(vec_temp).copy()
                        #[vec_temp[u].interpolation(space_pts.T) for u in range(dim)]).copy()
                
                temp+=inv_N*grad_vect_field[s].inner(vec_temp_interp)
            gradGD[i]+=temp*basisGD[iter_gd]

    for ite in range(10):
        X_temp=X.copy()
        X_temp[1]-=epsCont*gradCont.copy()
        X_temp[0]-=epsGD*gradGD
        energy=attach_tot(vector_fields_list,X_temp[0],X_temp[1])
        Reg_mod=functional_mod.ComputeReg(X_temp)
        energy_mod0=Reg_mod+energy

        X_temp=X.copy()
        X_temp[1]-=0.8*epsCont*gradCont.copy()
        X_temp[0]-=epsGD*gradGD
        energy=attach_tot(vector_fields_list,X_temp[0],X_temp[1])
        Reg_mod=functional_mod.ComputeReg(X_temp)
        energy_mod1=Reg_mod+energy

        X_temp=X.copy()
        X_temp[1]-=epsCont*gradCont.copy()
        X_temp[0]=0.8*epsGD*gradGD
        energy=attach_tot(vector_fields_list,X_temp[0],X_temp[1])
        Reg_mod=functional_mod.ComputeReg(X_temp)
        energy_mod2=Reg_mod+energy
        print('energy0 = {}, energy1 = {}, energy2 = {} '.format(energy_mod0,energy_mod1,energy_mod2) )
        if (energy_mod0 <= energy_mod1 and energy_mod0 <= energy_mod2):
            X_temp=X.copy()
            X_temp[1]-=epsCont*gradCont.copy()
            X_temp[0]-=epsGD*gradGD
            energy_mod_temp=energy_mod0
        elif (energy_mod1 <= energy_mod0 and energy_mod1 <= energy_mod2):
            X_temp=X.copy()
            X_temp[1]-=0.8*epsCont*gradCont.copy()
            X_temp[0]-=epsGD*gradGD
            energy_mod_temp=energy_mod1   
            epsCont*=0.8
        else:
            X_temp=X.copy()
            X_temp[1]-=epsCont*gradCont.copy()
            X_temp[0]-=0.8*epsGD*gradGD
            energy_mod_temp=energy_mod2
            epsGD*=0.8
                
        if (energy_mod_temp < energy_mod):
            X=X_temp.copy()
            energy_mod=energy_mod_temp
            print('k={} , energy = {} '.format(k,energy_mod_temp))
            epsGD*=1.2
            epsCont*=1.2
            break
        else:
           epsGD*=0.8
           epsCont*=0.8
        
    if (ite==9):
        print('No possible to descent')
        break
    #X[1]-=epsCont*gradCont.copy()
    #    print(X[1][0])
    #    #X[0]-=epsGD*gradGD
    #    print([X[0]])
    print('epsGD= {} , epsCont = {}'.format(epsGD,epsCont))
    #vector_fields_list=((1-eps_vect_field*lamb1)*vector_fields_list-eps_vect_field*grad_vect_field ).copy()
#

#%%


I_t=Shoot_mixt(vector_fields_list,X[0],X[1])

vector_fields_list_tot=Mix_vect_field(vector_fields_list,X[0],X[1])
grid_points=compute_grid_deformation_list(vector_fields_list_tot, 1/nb_time_point_int, template.space.points().T)

for t in range(nb_time_point_int+1):
    #t=nb_time_point_int
    I_t[t].show('Mixed strategy time {}'.format(t+1))
    grid=grid_points[t].reshape(2, space.shape[0], space.shape[1]).copy()
    plot_grid(grid, 5)
#










