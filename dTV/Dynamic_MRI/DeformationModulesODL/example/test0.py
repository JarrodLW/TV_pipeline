##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Mon May  8 11:02:49 2017
#
#@author: bgris
#"""
#
#
#
#functional = TemporalAttachmentModulesGeom.FunctionalModulesGeom(lam, nb_time_point_int, template, data, data_time_points, forward_operators,Norm, Module)
#
#
##energy_der=functional(X_temp)
#
#for j in range(10):
#
#    for i in range(5):
#        if (i>2):
#            break
#        print('i={}'.format(i))
#
#    print('j={}'.format(j))
#
#
#
##%%
#image_domain=template.space
#from odl.discr import DiscreteLp, Gradient, Divergence
#grad_op = Gradient(domain=image_domain, method='forward',
#                   pad_mode='symmetric')
## Create the divergence op
#div_op = -grad_op.adjoint
#template.space.element(1+div_op(vector_fields_list[0]))
#
#%%

import odl
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')

phantom=odl.phantom.cuboid(space)
phantom=odl.phantom.defrise(space)

ellipses0 = [[1, 0.5, 0.5, 0.0, 0.0, 0.0],[1.0, 0.15, 0.3, 0.0, 0.0, 0.0]]
phantom0=odl.phantom.geometric.ellipsoid_phantom(space,ellipses0)

ellipses1 = [[1, 0.5, 0.5, 0.0, 0.0, 0.0],[1.0, 0.3, 0.15, 0.0, 0.0, 0.0]]
phantom1=odl.phantom.geometric.ellipsoid_phantom(space,ellipses1)

phantom=odl.phantom.shepp_logan(space)

#%%
ellipses= [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.02, .1100, .3100, 0.2200, 0.0000, -18],
            [-.02, .1600, .4100, -.2200, 0.0000, 18],
            [0.01, .2100, .2500, 0.0000, 0.3500, 0],
            [0.01, .0460, .0460, 0.0000, 0.1000, 0],
            [0.01, .0460, .0460, 0.0000, -.1000, 0],
            [0.01, .0460, .0230, -.0800, -.6050, 0],
            [0.01, .0230, .0230, 0.0000, -.6060, 0],
            [0.01, .0230, .0460, 0.0600, -.6050, 0]]


ellipses1= [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.02, .1100, .3100, 0.2200, 0.0000, -18],
            [-.02, .1600, .4100, -.2200, 0.0000, 18],
            [0.01, .2100, .2500, 0.0000, 0.3500, 0],
            [0.01, .0460, .0460, 0.0000, 0.1000, 0],
            [0.01, .0460, .0460, 0.0000, -.1000, 0],
            [0.01, .0460, .0230, -.06500, -0.6550, 45],
            [0.01, .0230, .0230, 0.0000, -.6060, 0],
            [0.01, .0230, .0460, 0.0400, -.5550, 45]]




phantom=odl.phantom.geometric.ellipsoid_phantom(space,ellipses)
phantom.show(clim=[1 , 1.1])


phantom1=odl.phantom.geometric.ellipsoid_phantom(space,ellipses1)
phantom1.show(clim=[1 , 1.1])

#%%
#space = odl.uniform_discr(
#    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
#    dtype='float32', interp='linear')
#
#template= odl.phantom.shepp_logan(space)
#template.show(clim=[1,1.1])
#
#NRotation=1
#space_mod = odl.uniform_discr(
#    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
#    dtype='float32', interp='nearest')
#
#kernelrot=Kernel.GaussianKernel(2)
#rotation=LocalRotation.LocalRotation(space_mod, NRotation, kernelrot)
#
#GD=rotation.GDspace.element([[-0.0500, -9.5]])
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
#I1.show(clim=[1 , 1.1])
#
##%% Define functional
#lam=0.01
#nb_time_point_int=10
#forward_op=odl.IdentityOperator(space)
#
#lamb0=1e-7
#lamb1=1e-4
#
#data_time_points=np.array([1])
#data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
#data=data_space.element([proj_data])
#forward_operators=[forward_op]
#data_image=[I1]
#
#
#Norm=odl.solvers.L2NormSquared(forward_op.range)
#
#functional_mod = TemporalAttachmentModulesGeom.FunctionalModulesGeom(lamb0, nb_time_point_int, template, data, data_time_points, forward_operators,Norm, Module)
#
#
#


grad=functional.gradient(vector_fields_list)
##%%
vector_fields_list_init=energy_op.domain.zero()
vector_fields_list=vector_fields_list_init.copy()
attachment_term=energy_op(vector_fields_list)
print(" Initial ,  attachment term : {}".format(attachment_term))

for k in range(niter):
    grad=functional.gradient(vector_fields_list)
    vector_fields_list= (vector_fields_list- eps *grad).copy()
    attachment_term=energy_op(vector_fields_list)
    print(" iter : {}  ,  attachment term : {}".format(k,attachment_term))


#%%



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


space_mod = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256],
    dtype='float32', interp='nearest')


dim=2
NRotation=1

kernelrot=Kernel.GaussianKernel(1.5)
rotation=LocalRotation.LocalRotation(space_mod, NRotation, kernelrot)


GD=rotation.GDspace.element([[0,0]])
Cont=rotation.Contspace.element([10])

dGD=rotation.GDspace.element([[0,1]])
#rotation.ComputeFieldDer(GD,Cont)

rotation.ComputeFieldDer(GD,Cont)(dGD).show()

























