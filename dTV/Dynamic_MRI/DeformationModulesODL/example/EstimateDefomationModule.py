#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:59:39 2017

@author: bgris
"""

import odl
from odl.deform.linearized import _linear_deform
from odl.discr import DiscreteLp, Gradient, Divergence
from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
from odl.space import ProductSpace
import numpy as np
import scipy

def padded_ft_op(space, padded_size):
    """Create zero-padding fft setting

    Parameters
    ----------
    space : the space needs to do FT
    padding_size : the percent for zero padding
    """
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = FourierTransform(
        padded_op.range, halfcomplex=False, shift=shifts, impl='numpy')

    return ft_op * padded_op

def fitting_kernel(space, kernel):

    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return discretized_kernel

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

for i in range(nb_ellipses):
    images_ellipses[i].show('{}'.format(i))


# The parameter for kernel function
sigma = 2.0

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

def energy(source_list, target_list,kernel, forward_op,norm, X, lamalpha, lamv):
    dim=forward_op.domain.ndim
    space=source_list[0].space
    padded_size = 2 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    vect_field_list=X[0]
    # alpha_list is a list of size nb_vect_fields
    # for each k, alpha_list[k] is a list of nb_data scalar
    alpha_list=X[1]
    energy0=0
    energy1=0
    energy2=0
    nb_data=len(source_list)
    nb_vect_fields=len(vect_field_list)
    for i in range(nb_data):
        vect_field_temp=space.tangent_bundle.zero()
        for k in range(nb_vect_fields):
            vect_field_temp-=(alpha_list[k][i]*vect_field_list[k]).copy()
            energy0+=(alpha_list[k][i]-1)**2
            if (i==0):
                temp=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(vect_field_list[k]) * ft_kernel_fitting).copy()
                energy1+=temp.inner(vect_field_list[k])
        temp=_linear_deform(source_list[i],vect_field_temp).copy()
        energy2+=norm(forward_op(temp)-target_list[i])
    print("energy alpha = {}, energy V = {}, energy attach = {}".format(energy0, energy1, energy2))
    energy=lamalpha*energy0 + lamv*energy1+ energy2
    return energy

def energy_gradient(source_list, target_list,kernel, forward_op,norm, X, lamalpha, lamv):
    vect_field_list=X[0]
    # alpha_list is a list of size nb_vect_fields
    # for each k, alpha_list[k] is a list of nb_data scalar
    alpha_list=X[1]

    dim=forward_op.domain.ndim
    space=source_list[0].space
    padded_size = 2 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    grad_op = Gradient(domain=space, method='forward', pad_mode='symmetric')

    nb_data=len(source_list)
    nb_vect_fields=len(vect_field_list)
    list_vect_field_data=[]
    for n in range(nb_data):
        temp=space.tangent_bundle.zero()
        for k in range(nb_vect_fields):
            temp+= alpha_list[k][n]*vect_field_list[k].copy()
        list_vect_field_data.append(temp.copy())

    grad_vect=[]
    grad_alpha=[]

    for k in range(nb_vect_fields):
        temp_grad_vect=2*lamv* vect_field_list[k].copy()
        temp_grad_alpha=[]
        for n in range(nb_data):
            temp_grad_alpha.append(2*lamalpha*(alpha_list[k][n]-1))

            grad_S=(norm*(forward_op - target_list[n])).gradient(_linear_deform(source_list[n],-list_vect_field_data[n]).copy()).copy()
            grad_source_n=grad_op(source_list[n]).copy()
            tmp=space.tangent_bundle.element([
                    _linear_deform(grad_source_n[d],-list_vect_field_data[n]).copy() for d in range(dim)].copy())

            temp_grad_alpha[n]-=grad_S.inner(space.element(sum(tmp[d]*vect_field_list[k][d] for d in range(dim)).copy()))

            for d in range(dim):
                tmp[d]*=grad_S.copy()
            tmp3=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(tmp) * ft_kernel_fitting).copy()
            temp_grad_vect-=alpha_list[k][n]*tmp3.copy()

        grad_alpha.append(temp_grad_alpha.copy())
        grad_vect.append(temp_grad_vect.copy())

    return [grad_vect,grad_alpha]
#
#%%

forward_op = odl.IdentityOperator(space)

nb_data=4

source_list=[]
target_list=[]

for i in range(nb_data):
    source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[i].copy(),3)))
    target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[i+1].copy(),3)))

#source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[3],3)))
#target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[4],3)))

norm = odl.solvers.L2NormSquared(forward_op.range)
import random
X_init=[[],[]]
nb_vect_fields=1
for k in range(nb_vect_fields):
    X_init[0].append(space.tangent_bundle.zero())
    temp=[]
    for k in range(nb_data):
        temp.append(1)
    X_init[1].append(temp.copy())

#energy(source_list, target_list,kernel, forward_op,norm, X)
#grad=energy_gradient(source_list, target_list,kernel, forward_op,norm, X)



#%% Gradient descent
lamalpha=1e-5
lamv=1e-5
X=X_init.copy()
ener=energy(source_list, target_list,kernel, forward_op,norm, X, lamalpha, lamv)
print('Initial energy = {}'.format(ener))
niter=100
eps=0.02
for i in range(niter):
    grad=energy_gradient(source_list, target_list,kernel, forward_op,norm, X, lamalpha, lamv)
    X[0]=[X[0][k] - eps*grad[0][k] for k in range(nb_vect_fields)].copy()
    X[1]=[[X[1][k][n] - eps*grad[1][k][n] for n in range(nb_data)] for k in range(nb_vect_fields)]
    ener=energy(source_list, target_list,kernel, forward_op,norm, X, lamalpha, lamv)
    print('Iter = {},  energy = {}'.format(i,ener))
#
#%%

for n in range(nb_data):
    space.element(source_list[n]).show('Source {}'.format(n))
    space.element(target_list[n]).show('Target {}'.format(n))

for n in range(nb_data):
    temp=_linear_deform(source_list[n],space.tangent_bundle.element(sum(-X[1][k][n]*X[0][k] for k in range(nb_vect_fields)))).copy()
    space.element(temp).show('Transported source {}'.format(n))
#

#%% Save vector field estimated

np.savetxt('/home/barbara/DeformationModulesODL/deform/vect_field_ellipses',X[0][0])
   
vect_field_load=space.tangent_bundle.element(np.loadtxt('/home/barbara/DeformationModulesODL/deform/vect_field_ellipses')).copy()














