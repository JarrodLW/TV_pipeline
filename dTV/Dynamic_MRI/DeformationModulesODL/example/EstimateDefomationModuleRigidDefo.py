#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:03:48 2017

@author: barbara
"""

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

## For ellipse mouvement
#a_list=[0.2,0.4,0.6,0.8,1,0.2,0.4,0.6,0.8,1]
#b_list=[1,0.8,0.6,0.4,0.2,0.2,0.4,0.6,0.8,1]
#c0_list=0.1*np.array([-0.5, 0.2, 0,0.3,-0.5,0,0,0.1,0.3,-0.2 ])
#c1_list=0.1*np.array([0.1,-0.5,-0.2,0.4,0,0,0,-0.1,-0.1,0.2])
#theta=10*np.array([np.pi, 0.2*np.pi, -0.1*np.pi, 0.3*np.pi, 0, 0, 0,0.1*np.pi,-0.2*np.pi,0])
#fac=0.3
#nb_ellipses=len(a_list)
#images_ellipses_source=[]
#images_ellipses_target=[]
#for i in range(nb_ellipses):
#    ellipses=[[1,fac* a_list[i], fac*(b_list[i]+0.2), c0_list[i], c1_list[i], theta[i]]]
#    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
#    ellipses=[[1,fac* (a_list[i]+0.2), fac*(b_list[i]), c0_list[i], c1_list[i], theta[i]]]
#    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())

#for i in range(nb_ellipses):
#    images_ellipses_source[i].show('source {}'.format(i))
#    images_ellipses_target[i].show('target {}'.format(i))


# For rotation mouvement
a_list=[0.2,0.4,0.6,0.8,1,0.2,0.4,0.6,0.8,1]
b_list=[1,0.8,0.6,0.4,0.2,0.2,0.4,0.6,0.8,1]
c0_list=0.1*np.array([-0.5, 0.2, 0,0.3,-0.5,0,0,0.1,0.3,-0.2 ])
c1_list=0.1*np.array([0.1,-0.5,-0.2,0.4,0,0,0,-0.1,-0.1,0.2])
theta=10*np.array([np.pi, 0.2*np.pi, -0.1*np.pi, 0.3*np.pi, 0, 0, 0,0.1*np.pi,-0.2*np.pi,0])
fac=0.3
nb_ellipses=len(a_list)
images_ellipses_source=[]
images_ellipses_target=[]
for i in range(nb_ellipses):
    ellipses=[[1,fac* a_list[i], fac*(b_list[i]+0.2), c0_list[i], c1_list[i], theta[i]]]
    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
    ellipses=[[1,fac* a_list[i], fac*(b_list[i]+0.2), c0_list[i], c1_list[i], theta[i]+15]]
    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())





# The parameter for kernel function
sigma = 2.0

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))



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
def Rprimetheta(theta,points):
    # theta is the angle, in rad
    # input = list of points, for ex given by space.points() or 
    # np.array(vect_field).T
    #output = list of points of same size, rotated of an angle theta
    
    points_rot=np.empty_like(points).T
    points_rot[0]=-np.sin(theta)*points.T[0].copy() - np.cos(theta)*points.T[1].copy()
    points_rot[1]=np.cos(theta)*points.T[0].copy() - np.sin(theta)*points.T[1].copy()
    
    return points_rot.T.copy()
    
    
def SmallDefRigid(source, centre,angle,vect_field,alpha):
    dim=forward_op.domain.ndim
    space=source_list[0].space
    points=space.points()
    padded_size = 2 * space.shape[0]
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    padded_space=padded_op.range
    vect_field_padded=padded_space.tangent_bundle.element([padded_op(vect_field[u]) for u in range(dim)])
    
    points_temp=points.copy() - centre.copy() 
    
    # points rotated with angle - theta[i] 
    points_temp = Rtheta(-angle, points_temp).copy()    
    # Interpolation of the vector field on points_temp
    vect_field_temp=np.array(
            [vect_field_padded[k].interpolation(points_temp.T) for k in range(dim)]).T.copy()
    
    # Rotation of the vector field with angle theta[i]
    vect_field_temp=space.tangent_bundle.element(Rtheta(angle,vect_field_temp).T)
            
    # Shift of the vector fieldwith translation of vector center_list[i
    #vect_field_temp=space.tangent_bundle.element(
    #        [vect_field_temp[k] + centre[k] for k in range(dim)]).copy()
    
    vect_field_temp*=(-alpha)
    
    temp=_linear_deform(source,vect_field_temp).copy()
    
    return temp
    
    
def energyRigid(source_list, target_list,kernel, forward_op,norm, X, lamalpha, lamv):
    # X is a list containing lists of a vector field v (here we force nb_vect_fields=1),
    # alpha_i centers c_i , angles theta_i 
    
    dim=forward_op.domain.ndim
    space=source_list[0].space
    points=space.points()
    padded_size = 2 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    padded_space=padded_op.range
    vect_field=padded_space.tangent_bundle.element([padded_op(X[0][u]) for u in range(dim)])
    # alpha_list is a list of size nb_vect_fields
    # for each k, alpha_list[k] is a list of nb_data scalar
    alpha_list=X[1]
    center_list=X[2]
    angle_list=X[3]
    energy0=0
    energy1=0
    energy2=0
    nb_data=len(source_list)
    for i in range(nb_data):
        
        # points shifted with translation of vector - center_list[i]
        points_temp=points.copy() - center_list[i].copy() 
        
        # points rotated with angle - theta[i] 
        points_temp = Rtheta(-angle_list[i], points_temp).copy()
        
        # Interpolation of the vector field on points_temp
        vect_field_temp=np.array(
                [vect_field[k].interpolation(points_temp.T) for k in range(dim)]).T.copy()
        
        # Rotation of the vector field with angle theta[i]
        vect_field_temp=space.tangent_bundle.element(Rtheta(angle_list[i],vect_field_temp).T)
                
        # Shift of the vector fieldwith translation of vector center_list[i
        #vect_field_temp=space.tangent_bundle.element(
        #        [vect_field_temp[k] + center_list[i][k] for k in range(dim)]).copy()
        
        vect_field_temp*=(-alpha_list[i])
        
        energy0+=(alpha_list[i]-1)**2
        if (i==0):
            temp=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(vect_field_temp) * ft_kernel_fitting).copy()
            energy1+=temp.inner(vect_field_temp)
        temp=_linear_deform(source_list[i],vect_field_temp).copy()
        energy2+=norm(forward_op(temp)-target_list[i])
    print("energy alpha = {}, energy V = {}, energy attach = {}".format(energy0, energy1, energy2))
    energy=lamalpha*energy0 + lamv*energy1+ energy2
    return energy

def energyRigid_gradient(source_list, target_list,kernel, forward_op,norm, X, lamalpha, lamv):
    
    space=source_list[0].space
    points=space.points()
    padded_size = 2 * space.shape[0]
    dim=forward_op.domain.ndim
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    padded_space=padded_op.range
    vect_field=X[0]
    vect_field_padded=padded_space.tangent_bundle.element([padded_op(X[0][u]) for u in range(dim)])
    # alpha_list is a list of size nb_vect_fields
    # for each k, alpha_list[k] is a list of nb_data scalar
    alpha_list=X[1]
    center_list=X[2]
    angle_list=X[3]
    
    
    
    padded_size = 2 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    grad_op = Gradient(domain=space, method='forward', pad_mode='symmetric')

    nb_data=len(source_list)
    list_vect_field_data=[]
    for i in range(nb_data):
        
        # points shifted with translation of vector - center_list[i]
        points_temp=points.copy() - center_list[i].copy() 
        
        # points rotated with angle - theta[i] 
        points_temp = Rtheta(-angle_list[i], points_temp).copy()
        
        # Interpolation of the vector field on points_temp
        vect_field_temp=np.array(
                [vect_field_padded[k].interpolation(points_temp.T) for k in range(dim)]).T.copy()
        
        # Rotation of the vector field with angle theta[i]
        vect_field_temp=space.tangent_bundle.element(Rtheta(angle_list[i],vect_field_temp).T)
                
        # Shift of the vector fieldwith translation of vector center_list[i]
        #vect_field_temp=space.tangent_bundle.element(
        #        [vect_field_temp[k] + center_list[i][k] for k in range(dim)]).copy()
        
        
        list_vect_field_data.append(vect_field_temp.copy())

    grad_vect=space.tangent_bundle.element()
    grad_alpha=[]
    grad_center=[]
    grad_angle=[]

    temp_grad_vect=2*lamv* space.tangent_bundle.element([vect_field_padded[u].interpolation(points.T) for u in range(dim)])
    temp_grad_alpha=[]
    temp_grad_center=[]
    temp_grad_angle=[]
    
    for i in range(nb_data):
        grad_S_i=(norm*(forward_op - target_list[i])).gradient(_linear_deform(source_list[i],-alpha_list[i]*list_vect_field_data[i]).copy()).copy()
        grad_source_i=grad_op(source_list[i]).copy()
        grad_source_i_depl=space.tangent_bundle.element([
                _linear_deform(grad_source_i[d],-alpha_list[i]*list_vect_field_data[i]).copy() for d in range(dim)].copy())
        
        points_dec=Rtheta(-angle_list[i], points).copy() + center_list[i]
        tmp=grad_source_i_depl.copy()
        for u in range(dim):
            tmp[u]*= grad_S_i
            
        tmp_padded=padded_space.tangent_bundle.element([padded_op(tmp[v]) for v in range(dim)])
            
        # for vect    
        tmp_dec=np.array(
                [tmp_padded[u].interpolation(points_dec.T) for u in range(dim)]).T.copy()
        
        tmp_dec=space.tangent_bundle.element(Rtheta(-angle_list[i],tmp_dec).T)
        tmp3=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(tmp_dec) * ft_kernel_fitting).copy()
        
        temp_grad_vect-=alpha_list[i]*tmp3.copy()        
        
        #for alpha
        temp_grad_alpha.append(2*lamalpha*(alpha_list[i]-1))
        temp_grad_alpha[i]-= tmp.inner(list_vect_field_data[i])
        
        
        #for angle 
        points_temp0=points.copy() - center_list[i].copy()
        points_temp=Rtheta(-angle_list[i], points_temp0).copy()
        vect_field_temp=np.array([vect_field_padded[u].interpolation(points_temp.T) for u in range(dim)]).T.copy()
        vect_field_temp=space.tangent_bundle.element(
                Rprimetheta(angle_list[i],vect_field_temp).T).copy()
        
        #points_temp2=Rtheta(-angle_list[i],points_temp0).copy()
        points_temp3=Rprimetheta(-angle_list[i],points_temp0).copy()
        vect_field_temp2=space.tangent_bundle.element()
        # we save vect_field_temp3_list in order to use it for centers
        vect_field_temp3_list=[]
        for u in range(dim):
            grad_vect_field_u=grad_op(vect_field[u])
            grad_vect_field_u=padded_space.tangent_bundle.element([padded_op(grad_vect_field_u[v]) for v in range(dim)])
            vect_field_temp3_list.append(
                    space.tangent_bundle.element(
                            [grad_vect_field_u[d].interpolation(points_temp.T) for d in range(dim)]).copy())
            vect_field_temp2[u]=sum(vect_field_temp3_list[u][d]*space.element(points_temp3.T[d]) for d in range(dim)).copy()
            
        
        vect_field_temp2=space.tangent_bundle.element(
                Rtheta(angle_list[i],np.array(vect_field_temp2).T).T)
        
        vect_field_temp-=vect_field_temp2.copy()
        temp_grad_angle.append(-alpha_list[i]*vect_field_temp.inner(tmp))
        
        
        #for centers
        #image_one=space.one()
        #temp_grad_center.append([-alpha_list[i]*image_one.inner(tmp[u].copy()) for u in range(dim)])
        
        temp_grad_center.append(np.empty_like(center_list[i]))
        
        vect_field_temp4=np.array(
                [np.cos(angle_list[i])* vect_field_temp3_list[u][0] - 
                 np.sin(angle_list[i])* vect_field_temp3_list[u][1] for u in range(dim)]).T.copy()
    
        vect_field_temp4=space.tangent_bundle.element(
                Rtheta(angle_list[i],vect_field_temp4).T).copy()

        temp_grad_center[i][0]+=alpha_list[i]*tmp.inner(vect_field_temp4)       
            
        vect_field_temp5=np.array(
                [np.sin(angle_list[i])* vect_field_temp3_list[u][0] - 
                 np.cos(angle_list[i])* vect_field_temp3_list[u][1] for u in range(dim)]).T.copy()
    
        vect_field_temp5=space.tangent_bundle.element(
                Rtheta(angle_list[i],vect_field_temp5).T).copy()

        temp_grad_center[i][1]+=alpha_list[i]*tmp.inner(vect_field_temp4)       
        
        
    grad_alpha=temp_grad_alpha.copy()
    grad_vect=temp_grad_vect.copy()
    grad_angle=temp_grad_angle.copy()
    grad_center=temp_grad_center.copy()

    return [grad_vect,grad_alpha,grad_center,grad_angle]
#
#%%

forward_op = odl.IdentityOperator(space)

nb_data=10

source_list=[]
target_list=[]

for i in range(nb_data):
    source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses_source[i].copy(),3)))
    target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses_target[i].copy(),3)))

#source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[3],3)))
#target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[4],3)))

norm = odl.solvers.L2NormSquared(forward_op.range)
#import random
X_init=[[],[],[],[]]
nb_vect_fields=1

X_init[0]=space.tangent_bundle.zero()
tempalpha=[]
tempcenter=[]
tempangle=[]
for k in range(nb_data):
    tempalpha.append(1)
    tempcenter.append(np.array([0,0]))
    tempangle.append(0)
X_init[1]=tempalpha.copy()
X_init[2]=tempcenter.copy()
X_init[3]=tempangle.copy()

#energy(source_list, target_list,kernel, forward_op,norm, X)
#grad=energy_gradient(source_list, target_list,kernel, forward_op,norm, X)



#%% Gradient descent
lamalpha=1e-5
lamv=1e-5
X=X_init.copy()
ener=energyRigid(source_list, target_list,kernel, forward_op,norm, X, lamalpha, lamv)
print('Initial energy = {}'.format(ener))
niter=200
eps=0.02
eps0=eps
eps1=eps
eps2=eps
eps3=eps
for i in range(niter):
    grad=energyRigid_gradient(source_list, target_list,kernel, forward_op,norm, X, lamalpha, lamv)
    X_temp=X.copy()
    X_temp[0]=X[0] - eps0*grad[0].copy()
    X_temp[1]=np.array(X[1]) - eps1*np.array(grad[1])
    X_temp[2]=[np.array(X[2][i])- eps2*np.array(grad[2][i]) for i in range(nb_data)]
    X_temp[3]=[np.array(X[3][i]) - eps3*np.array(grad[3][i]) for i in range(nb_data)]
    
    ener_temp=energyRigid(source_list, target_list,kernel, forward_op,norm, X_temp, lamalpha, lamv)
    if (ener_temp<ener):
        X=X_temp.copy()
        ener=ener_temp
        print('Iter = {},  energy = {}, theta={}, eps0={} ,  eps1={} ,  eps2={} ,  eps3={} , '.format(i,ener,X[3],eps0,eps1,eps2,eps3))
        eps0*=1.2
        eps1*=1.2
        eps2*=1.2
        eps3*=1.2
    else:
        X_temp0=X.copy()
        X_temp0[0]=X[0] - 0.5*eps0*grad[0].copy()
        X_temp0[1]=np.array(X[1]) - eps1*np.array(grad[1])
        X_temp0[2]=[np.array(X[2][i])- eps2*np.array(grad[2][i]) for i in range(nb_data)]
        X_temp0[3]=[np.array(X[3][i]) - eps3*np.array(grad[3][i]) for i in range(nb_data)]
        ener_temp0=energyRigid(source_list, target_list,kernel, forward_op,norm, X_temp0, lamalpha, lamv)
        
        X_temp1=X.copy()
        X_temp1[0]=X[0] - eps0*grad[0].copy()
        X_temp1[1]=np.array(X[1]) - 0.5*eps1*np.array(grad[1])
        X_temp1[2]=[np.array(X[2][i])- eps2*np.array(grad[2][i]) for i in range(nb_data)]
        X_temp1[3]=[np.array(X[3][i]) - eps3*np.array(grad[3][i]) for i in range(nb_data)]
        ener_temp1=energyRigid(source_list, target_list,kernel, forward_op,norm, X_temp1, lamalpha, lamv)
        
        X_temp2=X.copy()
        X_temp2[0]=X[0] - eps0*grad[0].copy()
        X_temp2[1]=np.array(X[1]) - eps1*np.array(grad[1])
        X_temp2[2]=[np.array(X[2][i])- 0.5*eps2*np.array(grad[2][i]) for i in range(nb_data)]
        X_temp2[3]=[np.array(X[3][i]) - eps3*np.array(grad[3][i]) for i in range(nb_data)]
        ener_temp2=energyRigid(source_list, target_list,kernel, forward_op,norm, X_temp2, lamalpha, lamv)
        
        X_temp3=X.copy()
        X_temp3[0]=X[0] - 0.5*eps0*grad[0].copy()
        X_temp3[1]=np.array(X[1]) - eps1*np.array(grad[1])
        X_temp3[2]=[np.array(X[2][i])- eps2*np.array(grad[2][i]) for i in range(nb_data)]
        X_temp3[3]=[np.array(X[3][i]) - 0.5*eps3*np.array(grad[3][i]) for i in range(nb_data)]
        ener_temp3=energyRigid(source_list, target_list,kernel, forward_op,norm, X_temp3, lamalpha, lamv)
        
        if (ener_temp0 < ener_temp1 and ener_temp0 < ener_temp2 and ener_temp0 < ener_temp3):
            X_temp=X_temp0
            eps0*=0.5
            ener_temp=ener_temp0
        else:
            if(ener_temp1 < ener_temp0 and ener_temp1 < ener_temp2 and ener_temp1 < ener_temp3):
                X_temp=X_temp1
                eps1*=0.5
                ener_temp=ener_temp1
            else:
                if(ener_temp2 < ener_temp0 and ener_temp2 < ener_temp1 and ener_temp2 < ener_temp3):
                    X_temp=X_temp2
                    eps2*=0.5
                    ener_temp=ener_temp2
                else:
                    X_temp=X_temp3
                    eps3*=0.5
                    ener_temp=ener_temp3
       
        if (ener_temp<ener):
            X=X_temp.copy()
            ener=ener_temp
            print('Iter = {},  energy = {}, theta={}, eps0={} ,  eps1={} ,  eps2={} ,  eps3={} , '.format(i,ener,X[3],eps0,eps1,eps2,eps3))
            eps0*=1.2
            eps1*=1.2
            eps2*=1.2
            eps3*=1.2

        else:
            eps0*=0.5
            eps1*=0.5
            eps2*=0.5
            eps3*=0.5
            
        
        print('Iter = {},  eps = {}'.format(i,eps))
#
#%%

for n in range(nb_data):
    space.element(source_list[n]).show('Source {}'.format(n))
    space.element(target_list[n]).show('Target {}'.format(n))

for n in range(nb_data):
    temp=SmallDefRigid(source_list[n], X[2][n],X[3][n],X[0],X[1][n])
    (space.element(temp)-space.element(target_list[n])).show('Transported source {}'.format(n))
#

#%% Save vector field estimated

np.savetxt('/home/barbara/DeformationModulesODL/deform/vect_field_rotation_Rigid',X[0])

np.savetxt('/home/barbara/DeformationModulesODL/deform/vect_field_ellipses_Rigid',X[0])
   
vect_field=space.tangent_bundle.element(np.loadtxt('/home/barbara/DeformationModulesODL/deform/vect_field_rotation_Rigid')).copy()

import matplotlib.pyplot as plt
points=space.points()
v=X[0]
v=vect_field.copy()
plt.figure()
plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
plt.axis('equal')
plt.title('Rotated')









