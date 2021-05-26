#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:31:10 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:09:53 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:46:41 2017

@author: bgris
"""

""" test for simulated 4D cardiac data """

import odl
import numpy as np
from matplotlib import pylab as plt

from deform import Kernel
from deform import DeformationModuleAbstract
from deform import SumTranslations
from deform import UnconstrainedAffine
from deform import LocalScaling
from deform import TemporalAttachmentModulesGeom

#%% Load data as a list of images
a=120
b=256
c=256

space = odl.uniform_discr(
    min_pt=[-70, -127, -127], max_pt=[70, 127, 127], shape=[a,b,c],
    dtype='float32', interp='nearest')


data_list=[]
index_list=[0,16]
for i in range(len(index_list)):
    filename='/home/bgris/odl/examples/CardiacPhantom/SPECT_Torso_act_'+ str(index_list[i]+1) + '.bin'
    A = np.fromfile(filename, dtype='float32')
    A = A.reshape([a,b,c])
    data_list.append(space.element(A))
Ndata=len(index_list)
#data_list[0].show(indices=np.s_[:,space.shape[1] // 2,:])
#data_list[0].show(indices=np.s_[space.shape[0] // 2,:,:])

#data_list[1].show(indices=np.s_[:,space.shape[1] // 2,:])


#%% Extract from the images only the important part
a1=120
b1=100
c1=100
space_tronc= odl.uniform_discr(
    min_pt=[-70, -50, -60], max_pt=[70, 50, 60], shape=[a1,b1,c1],
    dtype='float32', interp='nearest')

template_tronc=space_tronc.element(
                 data_list[0].interpolation(space_tronc.points().T).reshape(a1,b1,c1))

data_tronc=space_tronc.element(
                 data_list[1].interpolation(space_tronc.points().T).reshape(a1,b1,c1))
#data_list[0].show(indices=np.s_[:,:,space_tronc.shape[0] // 2])
#%% Parameter for matching

space_mod = odl.uniform_discr(
    min_pt=[-70, -127, -127], max_pt=[70, 127, 127], shape=[a,b,c],
    dtype='float32', interp='nearest')
#space_mod=space_tronc
#%% Define Module

dim=3
NAffine=1
Ntrans=1
kernel=Kernel.GaussianKernel(30)


translation=SumTranslations.SumTranslations(space_mod, Ntrans, kernel)
affine=UnconstrainedAffine.UnconstrainedAffine(space_mod, NAffine, kernel)
Module=affine

dim=3
NScaling=1


scaling=LocalScaling.LocalScaling(space_mod, NScaling, kernel)



Module=affine

Module=DeformationModuleAbstract.Compound([affine])

#%% Define functional

forward_op=odl.IdentityOperator(space_tronc)
nb_time_point_int=5

lam=0.001
nb_time_point_int=5
template=template_tronc

data_time_points=np.array([1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
data=data_space.element([data_tronc])
forward_operators=[forward_op]
data_image=[data_list[1]]


Norm=odl.solvers.L2NormSquared(forward_op.range)



functional = TemporalAttachmentModulesGeom.FunctionalModulesGeom(lam, nb_time_point_int, template, data, data_time_points, forward_operators,Norm, Module)

Cont_init=odl.ProductSpace(Module.Contspace,nb_time_point_int+1).zero()

GD_init=Module.GDspace.element([[[0,0,0]]])

#%%



import timeit

start = timeit.default_timer()
energy=functional([GD_init,Cont_init])

end = timeit.default_timer()
print(end - start)


#%% Naive Gradient descent : gradient computed by finite differences
#AFFINE
#functional=functionalF
niter=50

eps = 1e-4
X=functional.domain.element([GD_init,Cont_init].copy())
attachment_term=functional(X)
energy=attachment_term
print(" Initial , attachment term : {}".format(attachment_term))
gradGD=functional.Module.GDspace.element()
gradCont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).element()

d_GD=functional.Module.GDspace.zero()
d_Cont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).zero()

delta=1
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
    #print(X[0])
    energytemp=functional(X_temp)
    if energytemp< energy:
        X= X_temp.copy()
        energy = energytemp
        print(" iter : {}  ,  energy : {}".format(k,energy))
        cont=1
        eps = eps*1.2
    else:
        eps = eps*0.8
        print(" iter : {}  ,  epsilon : {} ,  energytemp : {}".format(k,eps,energytemp))
        cont=0
#


#%% Naive Gradient descent : gradient computed by finite differences
# And descent direction by direction
#AFFINE
#functional=functionalF
niter=2

eps = 0.00000001
X=functional.domain.element([GD_init,Cont_init].copy())
attachment_term=functional(X)
energy=attachment_term
print(" Initial , attachment term : {}".format(attachment_term))
gradGD=functional.Module.GDspace.element()
gradCont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).element()

d_GD=functional.Module.GDspace.zero()
d_Cont=odl.ProductSpace(functional.Module.Contspace,nb_time_point_int+1).zero()
eps0Cont=10000
eps0GD=0.1
delta=1
#energy=functional(X)+1
cont=1
for k in range(niter):

    if(cont==1):
        #Computation of the gradient by finite differences
        for t in range(nb_time_point_int+1):
            for n in range(NAffine):
                for d in range(dim+1):
                    for u in range(dim):
                        print('t={} k={}  n={}  d={}  energy= {}'.format(t,k,n,d,energy))
                        eps=eps0Cont

                        ismax=0
                        X_temp=X.copy()
                        X_temp[1][t][n][d][u]+=delta
                        energy_diff=functional(X_temp)
                        der=(energy_diff-energy)/delta
                        X_temp=X.copy()
                        X_temp[1][t][n][d][u]-=eps*der
                        energy_temp=functional(X_temp)
                        if(energy_temp>energy):
                            for ite in range(10):
                                eps*=0.8
                                X_temp=X.copy()
                                X_temp[1][t][n][d][u]-=eps*der
                                energy_temp=functional(X_temp)
                                if (energy_temp<energy):
                                    ismax=1
                                    eps0Cont=eps
                                    break
                        else:
                            for ite in range(10):
                                eps*=1.2
                                X_temp=X.copy()
                                X_temp[1][t][n][d][u]-=eps*der
                                energy_temp=functional(X_temp)
                                if (energy_temp>=energy):
                                    eps/=1.2
                                    break
                            eps0Cont=eps
                            ismax=1


                        # Now we have 'the best' eps
                        if (ismax==1):
                            X[1][t][n][d][u]-=eps*der
                            energy=functional(X)



        for n in range(NAffine):
            for d in range(dim):
                print('k={}  n={}  d={}  energy= {}'.format(k,n,d,energy))
                eps=eps0GD
                eps1=eps
                ismax=0
                X_temp=X.copy()

                X_temp[0][n][d]+=delta
                energy_diff=functional(X_temp)
                der=(energy_diff-energy)/delta
                X_temp=X.copy()
                X_temp[0][n][d]-=eps*der
                energy_temp=functional(X_temp)
                if(energy_temp>energy):
                    for ite in range(10):
                        eps*=0.8
                        X_temp=X.copy()
                        X_temp[0][n][d]-=eps*der
                        energy_temp=functional(X_temp)
                        if (energy_temp<energy):
                            ismax=1
                            eps0GD=eps
                            break
                else:
                    for ite in range(10):
                        eps*=1.2
                        X_temp=X.copy()
                        X_temp[0][n][d]-=eps*der
                        energy_temp=functional(X_temp)
                        if (energy_temp>energy):
                            eps/=1.2
                            break
                    eps0GD[i]=eps
                    ismax=1

                # Now we have 'the best' eps
                if (ismax==1):
                    X[0][n][d]-=eps*der
                    energy=functional(X)




#%% See result

Registration=odl.deform.ShootTemplateFromVectorFields(vector_fields_list, template)
index_time_data=[0,5,10]
for i in range(len(index_list)):
    data[i].show('Ground truth time{}'.format(i),indices=np.s_[ space.shape[0] // 2,:, :])
    Registration[index_time_data[i]].show('Result time{}'.format(i),indices=np.s_[ space.shape[0] // 2,:, :])
    ((data[i]-template)**2).show('Initial Difference time{}'.format(i),indices=np.s_[ space.shape[0] // 2,:, :])
    ((data[i]-Registration[index_time_data[i]])**2).show('Final ifference,  time{}'.format(i),indices=np.s_[ space.shape[0] // 2,:, :])

#%%
for i in range(nb_time_point_int):
    I_t[i].show('Result time{}'.format(i),indices=np.s_[:,:,space_tronc.shape[2] // 2])

