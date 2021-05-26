#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:54:57 2017

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
#
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
#    ellipses=[[1,fac* a_list[i], fac*(b_list[i]+0.2), c0_list[0], c1_list[0], theta[i]]]
#    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
#    ellipses=[[1,fac* (a_list[i]+0.2), fac*(b_list[i]), c0_list[0], c1_list[0], theta[i]]]
#    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())

#for i in range(nb_ellipses):
#    images_ellipses_source[i].show('source {}'.format(i))
#    images_ellipses_target[i].show('target {}'.format(i))
#

# For rotation mouvement
a_list=[0.2,0.4,0.6,0.8,1,0.2,0.4,0.6,0.8,1]
b_list=[1,0.8,0.6,0.4,0.2,0.2,0.4,0.6,0.8,1]
c0_list=0.0*np.array([-0.5, 0.2, 0,0.3,-0.5,0,0,0.1,0.3,-0.2 ])
c1_list=0.0*np.array([0.1,-0.5,-0.2,0.4,0,0,0,-0.1,-0.1,0.2])
theta_init=50*np.array([0, 0.2*np.pi, -0.1*np.pi, 0.3*np.pi, 0,  -0.25*np.pi,  0.5*np.pi,0.1*np.pi,-0.2*np.pi,0])
fac=0.3
nb_ellipses=len(a_list)
images_ellipses_source=[]
images_ellipses_target=[]
for i in range(nb_ellipses):
    ellipses=[[1,fac* a_list[0], fac*(b_list[0]), c0_list[0], c1_list[0], theta_init[i]]]
    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
    ellipses=[[1,fac* a_list[0], fac*(b_list[0]), c0_list[0], c1_list[0], theta_init[i]+5]]
    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())




# For rotation mouvement, with a fixed surrounding ellipse to force locality
a_list=[0.2,0.4,0.6,0.8,1,0.2,0.4,0.6,0.8,1]
b_list=[1,0.8,0.6,0.4,0.2,0.2,0.4,0.6,0.8,1]
c0_list=0.0*np.array([-0.5, 0.2, 0,0.3,-0.5,0,0,0.1,0.3,-0.2 ])
c1_list=0.0*np.array([0.1,-0.5,-0.2,0.4,0,0,0,-0.1,-0.1,0.2])
theta_init=50*np.array([0, 0.2*np.pi, -0.1*np.pi, 0.3*np.pi, 0,  -0.25*np.pi,  0.5*np.pi,0.1*np.pi,-0.2*np.pi,0])
fac=0.3
nb_ellipses=len(a_list)
images_ellipses_source=[]
images_ellipses_target=[]
for i in range(nb_ellipses):
    ellipses=[[1,fac* a_list[0], fac*(b_list[0]), c0_list[0], c1_list[0], theta_init[i]],
          [1,2*fac* 0.2, 2*fac*1, c0_list[0], c1_list[0], theta_init[i]]]
    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
    ellipses=[[1,fac* 0.2, fac*1, c0_list[0], c1_list[0], theta_init[i]+5],
          [1,2*fac* 0.2, 2*fac*1, c0_list[0], c1_list[0], theta_init[i]]]
    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())





def fun_u_theta(theta):

    return np.array([np.cos(theta), np.sin(theta)]).copy()

def fun_v_theta(theta):

    return np.array([-np.sin(theta) , np.cos(theta)]).copy()


def fun_f(z):
    no=sum(z[i]**2 for i in range(len(z)))
    return (z/no).copy()

def fun_f_diff (z, deltaz):
    no=sum(z[i]**2 for i in range(len(z)))
    u=deltaz.copy()
    u-=sum([z[i]*deltaz[i] for i in range(len(z))])*(2/np.sqrt(no))*z
    return (u/no).copy()

def fun_g(z):
    no=sum(z[i]**2 for i in range(len(z)))
    return (z/np.sqrt(no)).copy()

def fun_g_diff(z,i):
    no=sum(z[i]**2 for i in range(len(z)))
    u=np.zeros_like(z).copy()
    u[i]=(1/np.sqrt(no))
    u= (u- (z[i] /(np.sqrt(no)**3))*z).copy()

    return u.copy()

def fun_g_perp(z):
    u=fun_g(z).copy()
    v=np.empty_like(u)
    v[0]=-u[1]
    v[1]=u[0]

    return v.copy()

def fun_g_perp_diff(z,i):
    u=fun_g_diff(z,i).copy()
    v=np.empty_like(u)
    v[0]=-u[1]
    v[1]=u[0]

    return v.copy()



def fun_alpha(b,theta,points):
    u_theta=fun_u_theta(theta).copy()

    points_dec=(points-b).copy()
    I=space.element(sum(points_dec.T[i]*u_theta[i] for i in range(len(u_theta))))

    return I

def fun_beta(b,theta,points):
    v_theta=fun_v_theta(theta).copy()

    points_dec=(points-b).copy()
    I=space.element(sum(points_dec.T[i]*v_theta[i] for i in range(len(v_theta))))

    return I



def fun_alpha_diff(b,theta,points,diff_ind):
    #diff_ind[0] determins if we differentiate wrt b or theta
    #diff_ind[1] determins wrt which component we differentiate(not important for theta)
    I=space.zero()
    if (diff_ind[0]==0):
        # derivate wrt b
        I=space.zero()
        if(diff_ind[0]==0):
            I=(I- np.cos(theta)).copy()
        else:
            I=(I- np.sin(theta)).copy()
    else:
       theta_rot=theta + 0.5*np.pi
       I=fun_alpha(b,theta_rot,points).copy()

    return I.copy()



def fun_beta_diff(b,theta,points,diff_ind):
    #diff_ind[0] determins if we differentiate wrt b or theta
    #diff_ind[1] determins wrt which component we differentiate(not important for theta)
    I=space.zero()
    if (diff_ind[0]==0):
        # derivate wrt b
        I=space.zero()
        if(diff_ind[0]==0):
            I=(I+ np.sin(theta)).copy()
        else:
            I=(I- np.cos(theta)).copy()
    else:
       theta_rot=theta+0.5*np.pi
       I=fun_beta(b,theta_rot,points).copy()

    return I.copy()
#



b=np.array([0.0,1.0])
theta=0

alph=fun_alpha(b,theta,space.points())
alph_diff=fun_alpha_diff(b,theta,space.points(),[0,0])
#alph.show('alpha')
#alph_diff.show('alpha diff')

v=[]
v.append(space.zero())
v.append(-0.5*space.one())
h=1
#
#def Compute_4_Vectors(theta):
#    vectors=[]
#    vectors.append(fun_g(a-c).copy())
#    vectors.append(fun_g_perp(a-c).copy())
#    vectors.append(fun_g(b-d).copy())
#    vectors.append(fun_g_perp(b-d).copy())
#
#    return vectors.copy()
##

#%%
def ComputeVectorFieldV2(b,theta,h,v):
    # a,b,c,d are the 4 points defining the localization (= geometrical descriptor)
    # h is the scalar control
    # v=[v11,v12,v21,v22] is made of the 4 functions on Omega defining the vector fields (fixed once estimated)
    points=space.points()

    # points shifted with translation of vector - Bary
    points_temp=points.copy() - b.copy()

    #alpha and beta are the coordinate of the points of Omega in the basis of vectors
    # (a-c)/|a-c| = fun_g(a-c) and (b-d)/|b-d| = fun_g(b-d)
    alph=fun_alpha(b,theta,points_temp)
    bet=fun_beta(b,theta,points_temp)

    # defining reference points = alpha e_1 + beta e_2 (with e_1=[1,0] and
    # e_2 = [0,1])
    points_ref=np.empty_like(points)
    points_ref.T[0]=np.reshape(np.asarray(alph),points.T[0].shape)
    points_ref.T[1]=np.reshape(np.asarray(bet),points.T[1].shape)

    padded_size = 3 * space.shape[0]
    #padded_ft_fit_op = padded_ft_op(space, padded_size)
    padded_op = ResizingOperator(
    space, ran_shp=[padded_size for _ in range(space.ndim)])
    padded_space=padded_op.range
    if v[0].space==space:
        v_padded=[padded_space.element(padded_op(v[u])) for u in range(len(v))]
    else:
        v_padded=v.copy()

    # v_interp is made of the interpolation of functions of v on
    # the reference points points_depl_origin
    v_interp=[]

    for i in range(len(v)):
        v_interp.append(
                space.element(v_padded[i].interpolation(points_ref.T)))

    vectors=[fun_u_theta(theta), fun_v_theta(theta)]
    vect_field=space.tangent_bundle.zero()


    for i in range(len(vectors)):
        vect_field_temp=space.tangent_bundle.zero()
        for j in range(len(vectors[i])):
            vect_field_temp[j]+=vectors[i][j]
            vect_field_temp[j]*=v_interp[i].copy()

        vect_field+=vect_field_temp.copy()

    return h*vect_field.copy()
#



#%%
def energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X, lamh, lamv, lamo):
    # X is a list containing (it this order) v which is a list of 4 functions defining the vector field,
    # h_list which is a list of scalar controls, b_listwhich is a list of points and
    # h_list which is a list of angles


    dim=forward_op.domain.ndim
    space=source_list[0].space

    padded_size = 3 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    h_list=X[1]
    b_list=X[2]
    theta_list=X[3]

    # List of the 4 functions defining the vector field
    v_list=X[0]

    # padding of functions so that the interpolation works even for points
    # slightly out of Omega
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    padded_space=padded_op.range
    v_ext=[padded_space.element(padded_op(v_list[u])) for u in range(len(v_list))]

    energy0=0
    energy1=0
    energy2=0
    energy3=0
    nb_data=len(source_list)
    for i in range(nb_data):

        energy0+=(h_list[i]**2-1)**2

        if (i==0):
            v_temp=space.tangent_bundle.zero()
            for j in range(len(v_list)):
                v_temp[0]=v_list[j].copy()
                temp=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(v_temp) * ft_kernel_fitting).copy()
                energy1+=temp.inner(v_temp)


        energy3+=sum([b_list[i][j]**2 for j in range(len(b_list[i]))])

        vect_field_i=ComputeVectorFieldV2( b_list[i],theta_list[i], h_list[i],v_ext).copy()


        temp=_linear_deform(source_list[i],-vect_field_i).copy()
        energy2+=norm(forward_op(temp)-target_list[i])

    print("energy alpha = {}, energy V = {}, norm b= {}, energy attach = {} ".format(lamh*energy0, lamv*energy1, lamo*energy3, energy2))
    energy=lamh*energy0 + lamv*energy1+ energy2 + lamo* energy3

    return energy
#
#%%
#source_list=source_list[1:3]
#target_list=target_list[1:3]
lamh=1
lamo=1
lamv=1
h_list=[1,1]
b_list=[b,b]
theta_list=[0,0]
v=[]
v.append(space.zero())
v.append(space.zero())
X=[v,h_list,b_list, theta_list]
#%%
def energyV2_gradient(source_list, target_list, kernel, forward_op,norm, X, lamh, lamv, lamo):
    # X is a list containing (it this order) v which is a list of 4 functions defining the vector field,
    # h_list which is a list of scalar controls, b_list which is a list of points and
    # theta_list which is a list of angles


    dim=forward_op.domain.ndim
    space=source_list[0].space

    padded_size = 3 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    h_list=X[1]
    b_list=X[2]
    theta_list=X[3]

    # List of the 4 functions defining the vector field
    v_list=X[0]

    # padding of functions so that the interpolation works even for points
    # slightly out of Omega
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    padded_space=padded_op.range
    v_ext=[padded_space.element(padded_op(v_list[u])) for u in range(len(v_list))]

    points=space.points()

    grad_op = Gradient(domain=space, method='forward', pad_mode='symmetric')

    nb_data=len(source_list)
    list_vect_field_data=[]

    for i in range(nb_data):

        vect_field_i=ComputeVectorFieldV2(b_list[i], theta_list[i], h_list[i],v_ext).copy()

        list_vect_field_data.append(vect_field_i.copy())

    grad_v=[2*lamv*v_list[j] for j in range(len(v_list))]
    grad_h=[]
    grad_b=np.zeros_like(b_list)
    grad_theta=np.zeros_like(theta_list)


    for i in range(nb_data):
        # Gradient of function : I \mapsto norm(T(I) -target[i] )  taken in source[i] transforme by vect_field_data[i]
        grad_S_i=(norm*(forward_op - target_list[i])).gradient(_linear_deform(source_list[i],-list_vect_field_data[i]).copy()).copy()
        grad_source_i=grad_op(source_list[i]).copy()
        grad_source_i_depl=space.tangent_bundle.element([
                _linear_deform(grad_source_i[d],-list_vect_field_data[i]).copy() for d in range(dim)].copy())

        tmp=grad_source_i_depl.copy()
        for u in range(dim):
            tmp[u]*= grad_S_i

        tmp_padded=padded_space.tangent_bundle.element([padded_op(tmp[vv]) for vv in range(dim)])

        vectors_i=[fun_u_theta(theta_list[i]), fun_v_theta(theta_list[i])]

        ######## for vect ########
        # the determinent of the matrix (fun_g (a-c), fun_g(b-d)) is needed in the following
        deter_i=1


        # points corresponding to T_o ^{-1} (\Omega)
        points_inv=np.empty_like(points)
        points_inv.T[0]=points.T[0]*vectors_i[0][0] + points.T[1]*vectors_i[1][0] + b_list[i][0]
        points_inv.T[1]=points.T[0]*vectors_i[0][1] + points.T[1]*vectors_i[1][1]  + b_list[i][1]

        tmp_dec=space.tangent_bundle.element([tmp_padded[u].interpolation(points_inv.T) for u in range(dim)]).copy()
        tmp1=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(tmp_dec) * ft_kernel_fitting).copy()
        tmp1*=(deter_i * h_list[i])

        for j in range(len(v_list)):
            grad_v[j]-=sum([vectors_i[j][u]*tmp1[u] for u in range(dim)])



        ######## for h ########
        grad_h.append(4*lamh*(h_list[i]**2-1)*h_list[i])
        grad_h[i]-= tmp.inner(list_vect_field_data[i])


        ######## for points b ########

        vect_field_derivates=[grad_op(list_vect_field_data[i][uu]) for uu in range(dim)]

        # points shifted with translation of vector - b
        points_temp=points.copy() - b_list[i].copy()
        alph=fun_alpha( b_list[i], theta_list[i], points_temp)
        bet=fun_beta( b_list[i], theta_list[i], points_temp)

        # defining reference points = alpha e_1 + beta e_2 (with e_1=[1,0] and
        # e_2 = [0,1])
        points_ref=np.empty_like(points)
        points_ref.T[0]=np.reshape(np.asarray(alph),points.T[0].shape)
        points_ref.T[1]=np.reshape(np.asarray(bet),points.T[1].shape)

        # v_interp is made of the interpolation of functions of v on
        # the reference points points_depl_origin
        v_interp=[]

        for uu in range(len(v_ext)):
            v_interp.append(
                    space.element(v_ext[uu].interpolation(points_ref.T)))


        for d in range(dim):
            alph_der=fun_alpha_diff( b_list[i], theta_list[i], points, [0,d]).copy()
            bet_der=fun_beta_diff( b_list[i], theta_list[i],  points, [0,d]).copy()

            vect_field_a=-space.tangent_bundle.element([vect_field_derivates[uu][0]*alph_der + vect_field_derivates[uu][1]*bet_der for uu in range(dim)]).copy()

            grad_b[i][d]+=tmp.inner(vect_field_a)
            grad_b[i][d]+=2*b_list[i][u]



        ######## for angles theta ########

        alph_der=fun_alpha_diff( b_list[i], theta_list[i], points, [1,0]).copy()
        bet_der=fun_beta_diff( b_list[i], theta_list[i],  points, [1,0]).copy()

        vect_field_theta=-space.tangent_bundle.element([vect_field_derivates[uu][0]*alph_der + vect_field_derivates[uu][1]*bet_der for uu in range(dim)]).copy()

        grad_theta[i]+=tmp.inner(vect_field_theta)

        theta_rot=theta_list[i]+ 0.5*np.pi
        u_der=fun_u_theta(theta_rot)
        v_der=fun_v_theta(theta_rot)

        grad_theta[i]-= tmp.inner(space.tangent_bundle.element([h_list[i]*v_interp[0]*u_der[uu] for uu in range(dim)]).copy())
        grad_theta[i]-= tmp.inner(space.tangent_bundle.element([h_list[i]*v_interp[1]*v_der[uu] for uu in range(dim)]).copy())

    return [grad_v, grad_h, grad_b, grad_theta]
#
#%%

forward_op = odl.IdentityOperator(space)

nb_data=10

source_list=[]
target_list=[]

for i in range(nb_data):
    source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses_source[i].copy(),1.5)))
    target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses_target[i].copy(),1.5)))

#source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[3],3)))
#target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[4],3)))
#%%
norm = odl.solvers.L2NormSquared(forward_op.range)
#import random
X_init=[[],[],[],[],[],[]]
nb_vect_fields=1

X_init[0]=[space.zero() for uu in range(2)]
temph=[]
tempb=[]
temptheta=[]
tempd=[]
for k in range(nb_data):
    temph.append(1)
    tempb.append(np.array([0.0,0.0]))
    temptheta.append(0.0)

X_init[1]=temph.copy()
X_init[2]=tempb.copy()
X_init[3]=temptheta.copy()
#X_init[3]=[0.5*theta_init[u]*np.pi/180 for u in range(len(theta_init))].copy()

#energy(source_list, target_list,kernel, forward_op,norm, X)
#grad=energy_gradient(source_list, target_list,kernel, forward_op,norm, X)


# The parameter for kernel function
sigma = 0.5

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))
#


#%% Gradient descent
lamh=1e-5
lamv=1*1e-1
lamo=1e-5
X=X_init.copy()
ener=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X, lamh, lamv, lamo)
print('Initial energy = {}'.format(ener))
niter=200
eps=0.02
eps0=eps
eps1=eps
eps2=eps
eps3=eps
#%%
cont=0
for i in range(niter):
    if (cont==0):
        grad=energyV2_gradient(source_list, target_list,kernel, forward_op,norm, X, lamh, lamv, lamo)

    X_temp=X.copy()
    X_temp[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(2)].copy()
    X_temp[1]=[X[1][uu] - eps1*grad[1][uu] for uu in range(nb_data)]
    X_temp[2]=[np.array(X[2][uu])- eps2*np.array(grad[2][uu]) for uu in range(nb_data)]
    X_temp[3]=[X[3][uu] - eps1*grad[3][uu] for uu in range(nb_data)]

    ener_temp=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp, lamh, lamv, lamo)

    if (ener_temp<ener):
        X=X_temp.copy()
        ener=ener_temp
        print('Iter = {},  energy = {},  eps0={} ,  eps1={} ,  eps2={}  '.format(i,ener,eps0,eps1,eps2))
        eps0*=1.2
        eps1*=1.2
        eps2*=1.2
        eps3*=1.2
        cont=0
    else:
        X_temp0=X.copy()
        X_temp0[0]=[X[0][uu] - 0.5*eps0*grad[0][uu] for uu in range(2)].copy()
        X_temp0[1]=[X[1][uu] - eps1*grad[1][uu] for uu in range(nb_data)]
        X_temp0[2]=[np.array(X[2][uu])- eps2*np.array(grad[2][uu]) for uu in range(nb_data)]
        X_temp0[3]=[X[3][uu] - eps3*grad[3][uu] for uu in range(nb_data)]
        ener_temp0=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp0, lamh, lamv, lamo)

        X_temp1=X.copy()
        X_temp1[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(2)].copy()
        X_temp1[1]=[X[1][uu] - 0.5*eps1*grad[1][uu] for uu in range(nb_data)]
        X_temp1[2]=[np.array(X[2][uu])- eps2*np.array(grad[2][uu]) for uu in range(nb_data)]
        X_temp1[3]=[X[3][uu] - eps3*grad[3][uu] for uu in range(nb_data)]
        ener_temp1=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp1, lamh, lamv, lamo)

        X_temp2=X.copy()
        X_temp2[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(2)].copy()
        X_temp2[1]=[X[1][uu] - eps1*grad[1][uu] for uu in range(nb_data)]
        X_temp2[2]=[np.array(X[2][uu])- 0.5*eps2*np.array(grad[2][uu]) for uu in range(nb_data)]
        X_temp2[3]=[X[3][uu] - eps3*grad[3][uu] for uu in range(nb_data)]
        ener_temp2=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp2, lamh, lamv, lamo)

        X_temp3=X.copy()
        X_temp3[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(2)].copy()
        X_temp3[1]=[X[1][uu] - eps1*grad[1][uu] for uu in range(nb_data)]
        X_temp3[2]=[np.array(X[2][uu])- eps2*np.array(grad[2][uu]) for uu in range(nb_data)]
        X_temp3[3]=[X[3][uu] - 0.5*eps3*grad[3][uu] for uu in range(nb_data)]
        ener_temp3=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp3, lamh, lamv, lamo)

        if (ener_temp0 < ener_temp1 and ener_temp0 < ener_temp2 and ener_temp0 < ener_temp3):
            X_temp=X_temp0.copy()
            eps0*=0.5
            ener_temp=ener_temp0
        else:
            if(ener_temp1 < ener_temp0 and ener_temp1 < ener_temp2 and ener_temp1 < ener_temp3):
                X_temp=X_temp1.copy()
                eps1*=0.5
                ener_temp=ener_temp1
            else:
                if(ener_temp1 < ener_temp0 and ener_temp1 < ener_temp2 and ener_temp1 < ener_temp3):
                    X_temp=X_temp2.copy()
                    eps2*=0.5
                    ener_temp=ener_temp2
                else:
                    X_temp=X_temp3.copy()
                    eps3*=0.5
                    ener_temp=ener_temp3

        if (ener_temp<ener):
            X=X_temp.copy()
            ener=ener_temp
            print('Iter = {},  energy = {},  eps0={} ,  eps1={} ,  eps2={} , eps3={}'.format(i,ener,eps0,eps1,eps2,eps3))
            eps0*=1.2
            eps1*=1.2
            eps2*=1.2
            eps3*=1.2
            cont=0

        else:
            cont=1
            eps0*=0.5
            eps1*=0.5
            eps2*=0.5
            eps3*=0.5


        print('Iter = {},  eps = {}'.format(i,eps))
#
#%%
nb_datamax=nb_data

import matplotlib.pyplot as plt
for n in range(nb_datamax):
    space.element(source_list[n]).show('Source {}'.format(n))
    #space.element( source_list[n] - target_list[n]).show('Initial difference {}'.format(n))
    vect_field_n=ComputeVectorFieldV2(X[2][n],X[3][n],X[1][n],X[0])
    temp=_linear_deform(source_list[n],-vect_field_n).copy()
    (space.element(temp)-space.element(target_list[n])).show('Transported source {}'.format(n))
    #(space.element(temp)).show('Transported source {}'.format(n))
    points=space.points()
    v=vect_field_n.copy()
#    plt.figure()
#    plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
#    plt.axis('equal')
#    plt.title('v n {}'.format(n))
    #vect_field_n.show('Vector field {}'.format(n))
#
#%%
import matplotlib.pyplot as plt
n=1
source_list[n] .show('source {}'.format(n))
vect_field_n=ComputeVectorFieldV2(X[2][n],X[3][n],X[1][n],X[0])
temp=_linear_deform(source_list[n],-vect_field_n).copy()
#(space.element(temp)-space.element(target_list[n])).show('Transported source {}'.format(n))
(space.element(temp)).show('Transported source {}'.format(n))
points=space.points()
v=vect_field_n.copy()
plt.figure()
plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
plt.axis('equal')
plt.title('v n {}'.format(n))

#%% plot reference
import matplotlib.pyplot as plt
plt.figure()
points=space.points()
plt.quiver(points.T[0][::20],points.T[1][::20],X[0][0][::20],X[0][1][::20])
plt.axis('equal')
plt.title('Reference')


#%%

for i in range(2):
    X[0][i].show('{}'.format(i))
#
#%% Save vector field estimated
#np.savetxt('/home/barbara/DeformationModulesODL/deform/vect_field_rotation_Rigid',X[0])

np.savetxt('/home/bgris/DeformationModulesODL/deform/vect_field_rotation_mvt_V3_sigma_2_lamv_1e__3_lamh_1e__5_lamo_1e__5',space.tangent_bundle.element(X[0]))

vect_field=space.tangent_bundle.element(np.loadtxt('/home/bgris/DeformationModulesODL/deform/vect_field_V3_sigma_2_lamv_1e__3_lamh_1e__5_lamo_1e__5')).copy()

import matplotlib.pyplot as plt
points=space.points()
#v=X[0]
v=vect_field_n.copy()
plt.figure()
plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
plt.axis('equal')
plt.title('v n')

#%%
for u in range(2):
    X[0][u].show('{}'.format(u))
#

#%%

n=0
ellipses=[[1,fac* 0.2, fac*1, c0_list[0], c1_list[0], theta_init[n]],
          [1,0.1, 0.1, -0.3, 0.3, theta_init[i]],
          [1,0.1, 0.1, -0.5, -0.5, theta_init[i]],
          [1,0.1, 0.1, 0.5, 0.5, theta_init[i]],
          [1,0.1, 0.1, 0.5, -0.5, theta_init[i]]]

ellipses=[[1,fac* 0.2, fac*1, c0_list[0], c1_list[0], theta_init[n]+5],
          [1,2*fac* 0.2, 2*fac*1, c0_list[0], c1_list[0], theta_init[n]]]

source_test=odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy()
source_test.show()
source_list[n] .show('source {}'.format(n))
vect_field_n=ComputeVectorFieldV2(X[2][n],X[3][n],X[1][n],X[0])
temp=_linear_deform(source_test,-vect_field_n).copy()
space.element(temp).show()


