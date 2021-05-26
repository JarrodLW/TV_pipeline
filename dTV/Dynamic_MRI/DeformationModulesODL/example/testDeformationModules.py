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
from DeformationModulesODL.deform import TemporalAttachmentModulesGeom

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




I1name = '/home/bgris/Downloads/pictures/j.png'
I0name = '/home/bgris/Downloads/pictures/v.png'
I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)

# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=I0.shape,
    dtype='float32', interp='linear')

# Create the ground truth as the given image
ground_truth =space.element(I1)


# Create the template as the given image
template = space.element(I0)





I0=space.element(I0)
I1=space.element(I1)
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
forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')







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
NRotation=2
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

#Module=DeformationModuleAbstract.Compound([translation,rotation])
Module=DeformationModuleAbstract.Compound([translation])
#ModuleF=translationF
#Module=affine
#Module=DeformationModuleAbstract.Compound([translation,translation])


#%% Define functional
lam=0.01
nb_time_point_int=10
template=I0


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

data_time_points=np.array([1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
data=data_space.element([proj_data])
forward_operators=[forward_op]
data_image=[I1]


Norm=odl.solvers.L2NormSquared(forward_op.range)



functional = TemporalAttachmentModulesGeom.FunctionalModulesGeom(lam, nb_time_point_int, template, data, data_time_points, forward_operators,Norm, Module)
#functionalF = TemporalAttachmentModulesGeom.FunctionalModulesGeom(lam, nb_time_point_int, template, data, data_time_points, forward_operators,Norm, ModuleF)
#GD_init=Module.GDspace.element([[[0,0]],[[0,-5],[0,5]]])
GD_init=Module.GDspace.element([[[-10,15]]])
#GD_init=Module.GDspace.element([[-4,0], [-2,0], [0,0], [2,0], [4,0],[-4,2], [-2,2], [0,2], [2,2], [4,2],[-4,4], [-2,4], [0,4], [2,4], [4,4],[-4,-2], [-2,-2], [0,-2], [2,-2], [4,-2],[-4,-4], [-2,-4], [0,-4], [2,-4], [4,-4]])
Cont_init=odl.ProductSpace(Module.Contspace,nb_time_point_int+1).zero()
#%% Initial parameter
mini=-16
#maxi=16
#nbsq=20
#
#
#
#ec=(maxi-mini)/nbsq
#GD_init=Module.GDspace.element()
#cont=0
#for i in range(nbsq):
#    for j in range(nbsq):
#      GD_init[cont]=odl.rn(2).element([mini+(i+0.5)*ec, mini+(j+0.5)*ec])
#      cont+=1
#
#GD_init=Module.GDspace.element([[0,-7],[0,0],[0,7]])
#GD_init=Module.GDspace.element([[0,-7.5],[0,-4.5],[0,-1.5],[0,1.5],[0,4.5],[0,7.5]])
#GD_init=Module.GDspace.element([[0,0]])
#Cont=Module.basisCont[0]+2*Module.basisCont[11]
GD_init=Module.GDspace.element([[[0,5],[0,-5]],[[0,0]]])
#GD_init=Module.GDspace.element([[[0,0]]])
#%%

import random
Cont=Module.basisCont[0]+2*Module.basisCont[1]
Cont=Module.Contspace.zero()
for i in range(len(Module.basisCont)):
    Cont+=(random.gauss(0,1)*Module.basisCont[i]).copy()

#%%
vectField=Module.ComputeField(GD_init,Cont)
vectFieldF=ModuleF.ComputeField(GD_init,Cont)


#%%
grad=functional.gradient([GD_init,Cont_init])
#%%
energy=functional([GD_init,Cont_init])
energy=functionalF([GD_init,Cont_init])

#%%
A=Module.ComputeFieldDer(GD_init,Cont)(GD_init)
B=ModuleF.ComputeFieldDer(GD_init,Cont)(GD_init)
#%%

import timeit

start = timeit.default_timer()
#energy=functional([GD_init,Cont_init])
for i in range(220):
    traj=functional([GD_init,Cont_init])
    print(i)


#vectField=Module.ComputeField(GD_init,Cont)
#speed=Module.ApplyVectorField(GD_init,vectField)
#Module.Cost(GD_init,Cont)
#A=Module.ComputeFieldDer(GD_init,Cont)(GD_init)
#op=Module.ApplyModule(ModuleF,GD_init,Cont)
#A=op(GD_init)
#Module.CostDerGD(GD_init,Cont)(GD_init)
#Module.CostGradGD(GD_init,Cont)
end = timeit.default_timer()
print(end - start)


#%%


import timeit

start = timeit.default_timer()
#energyF=functionalF([GD_init,Cont_init])
#vectField=ModuleF.ComputeField(GD_init,Cont)
traj=functionalF.ComputetrajectoryGD([GD_init,Cont_init])
#speed=ModuleF.ApplyVectorField(GD_init,vectField)
#ModuleF.Cost(GD_init,Cont)

#B=ModuleF.ComputeFieldDer(GD_init,Cont)(GD_init)
#op=ModuleF.ApplyModule(ModuleF,GD_init,Cont)
#B=op(GD_init)
#ModuleF.CostDerGD(GD_init,Cont)(GD_init)
#ModuleF.CostGradCont(GD_init,Cont)
end = timeit.default_timer()
print(end - start)




#%%
speed0=vectField[0].interpolation(np.array(GD_init).T)
speed1=vectField[1].interpolation(np.array(GD_init).T)

speed = Module.GDspace.element(np.array([speed0,speed1]).T)


speed = Module.GDspace.element(np.array([vectField[i].interpolation(np.array(GD_init).T) for i in range(2)]).T)
#%%

grad=functional.gradient([GD_init,Cont_init])

#%% Naive Gradient descent : gradient computed by finite differences
# TRANSLATION
#functional=functionalF
niter=200
eps = 0.1
X=functional.domain.element([GD_init,Cont_init].copy())
attachment_term=functional(X)
energy=attachment_term
print(" Initial , attachment term : {}".format(attachment_term))
gradGD=functional.Module.GDspace.element()
gradCont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).element()

d_GD=functional.Module.GDspace.zero()
d_Cont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).zero()

delta=0.001
#energy=functional(X)+1
cont=1
for k in range(niter):

    if(cont==1):
        #Computation of the gradient by finite differences
        for t in range(nb_time_point_int+1):
            for n in range(Ntrans):
                for d in range(dim):
                    X_temp=X.copy()
                    X_temp[1][t][n][d]+=delta
                    energy_der=functional(X_temp)
                    #print('t={}  n={}  d={}  energy_der={}'.format(t,n,d,energy_der))
                    gradCont[t][n][d]=(energy_der-energy)/delta

        for n in range(Ntrans):
            for d in range(dim):
                X_temp=X.copy()
                X_temp[0][n][d]+=delta
                energy_der=functional(X_temp)
                #print('n={}  d={}  energy_der={}'.format(n,d,energy_der))
                gradGD[n][d]=(energy_der-energy)/delta
        #print(gradGD)
        grad=functional.domain.element([gradGD,gradCont])

    X_temp= (X- eps *grad).copy()
    #print(X[0])
    energytemp=functional(X_temp)
    if energytemp< energy:
        X= X_temp.copy()
        energy = energytemp
        print(" iter : {}  ,  energy : {}".format(k,energy))
        cont=1
        eps = eps*1.2
    else:
        eps = eps/2
        print(" iter : {}  ,  epsilon : {}".format(k,eps))
        cont=0
#



#%% Naive Gradient descent : gradient computed by finite differences
#AFFINE
#functional=functionalF
niter=100
eps = 0.001
X=functional.domain.element([GD_init,Cont_init].copy())
attachment_term=functional(X)
energy=attachment_term
print(" Initial , attachment term : {}".format(attachment_term))
gradGD=functional.Module.GDspace.element()
gradCont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).element()

d_GD=functional.Module.GDspace.zero()
d_Cont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).zero()

delta=0.1
#energy=functional(X)+1
cont=1
for k in range(niter):

    if(cont==1):
        #Computation of the gradient by finite differences
        for t in range(nb_time_point_int+1):
            for n in range(NAffine):
                for d in range(dim+1):
                    for u in range(dim):
                        X_temp=X.copy()
                        X_temp[1][t][n][d][u]+=delta
                        energy_der=functional(X_temp)
                    #print('t={}  n={}  d={}  energy_der={}'.format(t,n,d,energy_der))
                        gradCont[t][n][d][u]=(energy_der-energy)/delta

        for n in range(NAffine):
            for d in range(dim):
                X_temp=X.copy()
                X_temp[0][n][d]+=delta
                energy_der=functional(X_temp)
                #print('n={}  d={}  energy_der={}'.format(n,d,energy_der))
                gradGD[n][d]=(energy_der-energy)/delta
        #print(gradGD)
        grad=functional.domain.element([gradGD,gradCont])

    X_temp= (X- eps *grad).copy()
    print(X[0])
    energytemp=functional(X_temp)
    if energytemp< energy:
        X= X_temp.copy()
        energy = energytemp
        print(" iter : {}  ,  energy : {}".format(k,energy))
        cont=1
        eps = eps*1.2
    else:
        eps = eps*0.8
        print(" iter : {}  ,  epsilon : {}".format(k,eps))
        cont=0
#



#%% Naive Gradient descent : gradient computed by finite differences
# Descent for all times simultaneously
#Combination of modules
#functional=functionalF
niter=10
eps = 0.01
#GD_init=Module.GDspace.element([GD_init_trans])
#Cont_init=odl.ProductSpace(Module.Contspace,nb_time_point_int+1).zero()

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
delta=0.1
epsmax=10
# 0=SumTranslations, 1=affine, 2=scaling, 3=rotation
Types=[0]
#energy=functional(X)+1
eps0Cont=[1,1]
eps0GD=[1,1]
cont=1
for k in range(niter):

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
                        print('k={} i={}  n={}  d={}  u={} eps={}  energy= {}'.format(k,i,n,d,u,eps,energy))
                        for t in range(nb_time_point_int):
                            X_temp_diff=X.copy()
                            delta=0.1*np.abs(X_temp_diff[1][t][i][n][d][u])
                            if (delta==0):
                                delta=0.1
                            X_temp_diff[1][t][i][n][d][u]+=delta
                            energy_diff=functional(X_temp_diff)
                            der[t]=(energy_diff-energy)/delta
                            X_temp[1][t][i][n][d][u]-=eps*der[t]
                        energy_temp=functional(X_temp)
                        if(energy_temp>energy):
                            for ite in range(10):
                                eps*=0.8
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n][d][u]-=eps*der[t]
                                energy_temp=functional(X_temp)
                                if (energy_temp<energy):
                                    ismax=1
                                    eps0Cont[i]=eps
                                    break
                        else:
                            for ite in range(10):
                                eps*=1.2
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n][d][u]-=eps*der[t]
                                energy_temp=functional(X_temp)
                                if (energy_temp>=energy):
                                    eps/=1.2
                                    break
                            eps0Cont[i]=eps
                            ismax=1


                        # Now we have 'the best' eps
                        if (ismax==1):
                            for t in range(nb_time_point_int):
                                X[1][t][i][n][d][u]-=eps*der[t]
                            energy=functional(X)

        elif (Types[i]==0):
                Ntrans=ModulesList[i].Ntrans
                for n in range(Ntrans):
                    for d in range(dim):
                        if eps0Cont[i]>epsmax:
                            eps0Cont[i]=epsmax
                        eps=eps0Cont[i]
                        print('k={} i={}  n={}  d={} eps={}  energy= {}'.format(k,i,n,d,eps,energy))
                        X_temp=X.copy()
                        ismax=0
                        der=np.empty(nb_time_point_int)
                        for t in range(nb_time_point_int):
                            X_temp_diff=X.copy()
                            delta=np.abs(X_temp_diff[1][t][i][n][d])
                            if (delta==0):
                                delta=0.1
                            X_temp_diff[1][t][i][n][d]+=delta
                            energy_diff=functional(X_temp_diff)
                            der[t]=(energy_diff-energy)/delta
                            X_temp[1][t][i][n][d]-=eps*der[t]
                        energy_temp=functional(X_temp)
                        if(energy_temp>energy):
                            for ite in range(10):
                                eps*=0.8
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n][d]-=eps*der[t]
                                energy_temp=functional(X_temp)
                                if (energy_temp<energy):
                                    ismax=1
                                    eps0Cont[i]=eps
                                    break
                        else:
                            for ite in range(10):
                                eps*=1.2
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n][d]-=eps*der[t]
                                energy_temp=functional(X_temp)
                                if (energy_temp>=energy):
                                    eps/=1.2
                                    break
                            eps0Cont[i]=eps
                            ismax=1


                        # Now we have 'the best' eps
                        if (ismax==1):
                            for t in range(nb_time_point_int):
                                X[1][t][i][n][d]-=eps*der[t]
                            energy=functional(X)



        elif (Types[i]==3 or Types[i]==2):
                if (Types[i]==3):
                    Nrot=ModulesList[i].NRotation
                else:
                      Nrot=ModulesList[i].NScaling
                for n in range(Nrot):
                        print('k={}  i={}  n={}   energy= {}'.format(k,i,n,energy))
                        X_temp=X.copy()
                        if (eps0Cont[i]>epsmax):
                            eps0Cont[i]=epsmax
                        eps=eps0Cont[i]
                        ismax=0
                        der=np.empty(nb_time_point_int)
                        for t in range(nb_time_point_int):
                            X_temp_diff=X.copy()
                            delta=np.abs(X_temp_diff[1][t][i][n])
                            if (delta==0):
                                delta=0.1
                            X_temp_diff[1][t][i][n]+=delta
                            energy_diff=functional(X_temp_diff)
                            der[t]=(energy_diff-energy)/delta
                            X_temp[1][t][i][n]-=eps*der[t]
                        energy_temp=functional(X_temp)
                        if(energy_temp>energy):
                            for ite in range(10):
                                eps*=0.8
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n]-=eps*der[t]
                                energy_temp=functional(X_temp)
                                if (energy_temp<energy):
                                    ismax=1
                                    eps0Cont[i]=eps
                                    break
                        else:
                            for ite in range(10):
                                eps*=1.2
                                X_temp=X.copy()
                                for t in range(nb_time_point_int):
                                    X_temp[1][t][i][n]-=eps*der[t]
                                energy_temp=functional(X_temp)
                                if (energy_temp>=energy):
                                    eps/=1.2
                                    break
                            eps0Cont[i]=eps
                            ismax=1


                        # Now we have 'the best' eps
                        if(ismax==1):
                            for t in range(nb_time_point_int):
                                X[1][t][i][n]-=eps*der[t]
                            energy=functional(X)



    for i in range(NbMod):
        if (Types[i]==0):

            Ntrans=ModulesList[i].Ntrans
            for n in range(Ntrans):
                for d in range(dim):
                    if eps0GD[i]>epsmax:
                        eps0GD[i]=epsmax
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












#%%

def kernelFun(x):
    sigma = 2.0
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

def kernelOpFun(x):
    return kernel.Eval(x)
#%% Gradient descent
functional=functionalF
niter=100
eps = 0.001
X=functional.domain.element([GD_init,Cont_init].copy())
attachment_term=functional(X)
print(" Initial , attachment term : {}".format(attachment_term))

energy=functional(X)+1
for k in range(niter):
    grad=functional.gradient(X)
    X= (X- eps *grad).copy()
    energytemp=functional(X)
    if energytemp< energy:
        X= (X- eps *grad).copy()
        energy = energytemp
        print(" iter : {}  ,  energy : {}".format(k,energy))
    else:
        eps = eps/2
        print(" iter : {}  ,  epsilon : {}".format(k,eps))
#
#%%  see result
I_t=template
GD_t=functional.ComputetrajectoryGD(X)
I_t.show('time {}'.format(0))
for t in range(nb_time_point_int+1):
    vect_field=-Module.ComputeField(GD_t[t],X[1][t]).copy()
    deform_op = odl.deform.LinDeformFixedTempl(I_t)
    I_t = deform_op(vect_field)
    I_t.show('time {}'.format(t+1))
#

#%%

I_t = functional.Shoot(X)
I_t[5].show()

data.show()
((data[0]-I_t[0])**2).show('Initial difference')
((data[0]-I_t[5])**2).show('Final difference')
#%%
data_image[0].show()

((template-data_image[0])**2).show('Initial difference')
((I_t-data_image[0])**2).show('Final difference')


((forward_op(template-data_image[0]))**2).show('Initial difference')
((forward_op(I_t-data_image[0]))**2).show('Final difference')

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