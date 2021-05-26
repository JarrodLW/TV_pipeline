#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:23:27 2017

@author: barbara
"""

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

# For ellipse mouvement
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
    ellipses=[[1,fac* a_list[i], fac*(b_list[i]+0.2), c0_list[0], c1_list[0], theta[i]]]
    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
    ellipses=[[1,fac* (a_list[i]+0.2), fac*(b_list[i]), c0_list[0], c1_list[0], theta[i]]]
    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())

##for i in range(nb_ellipses):
##    images_ellipses_source[i].show('source {}'.format(i))
##    images_ellipses_target[i].show('target {}'.format(i))


## For rotation mouvement
#a_list=[0.2,0.4,0.6,0.8,1,0.2,0.4,0.6,0.8,1]
#b_list=[1,0.8,0.6,0.4,0.2,0.2,0.4,0.6,0.8,1]
#c0_list=0.0*np.array([-0.5, 0.2, 0,0.3,-0.5,0,0,0.1,0.3,-0.2 ])
#c1_list=0.0*np.array([0.1,-0.5,-0.2,0.4,0,0,0,-0.1,-0.1,0.2])
#theta=10*np.array([0, 0.2*np.pi, -0.1*np.pi, 0.3*np.pi, 0, 0, 0,0.1*np.pi,-0.2*np.pi,0])
#fac=0.3
#nb_ellipses=len(a_list)
#images_ellipses_source=[]
#images_ellipses_target=[]
#for i in range(nb_ellipses):
#    ellipses=[[1,fac* a_list[0], fac*(b_list[0]+0.2), c0_list[i], c1_list[i], theta[i]]]
#    images_ellipses_source.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())
#    ellipses=[[1,fac* a_list[0], fac*(b_list[0]+0.2), c0_list[i], c1_list[i], theta[i]+15]]
#    images_ellipses_target.append(odl.phantom.geometric.ellipsoid_phantom(space,ellipses).copy())





# The parameter for kernel function
sigma = 0.5

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))





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



def fun_alpha(a,b,c,d,points):
    u=(fun_f(fun_g(a-c) + fun_g(b-d)) + fun_f(fun_g(a-c) - fun_g(b-d))).copy()
    bary=(a+b+c+d)/4
    points_dec=(points-bary).copy()
    I=space.element(sum(points_dec.T[i]*u[i] for i in range(len(u))))

    return I

def fun_beta(a,b,c,d,points):
    u=(fun_f(fun_g(a-c) + fun_g(b-d)) - fun_f(fun_g(a-c) - fun_g(b-d))).copy()
    bary=(a+b+c+d)/4
    points_dec=(points-bary).copy()
    I=space.element(sum(points_dec.T[i]*u[i] for i in range(len(u))))

    return I



def fun_alpha_diff(a,b,c,d,points,diff_ind):
    #diff_ind[0] determins if we differentiate wrt a,b,c or d
    #diff_ind[1] determins wrt which component we differentiate
    u=diff_ind[1]
    alpha=0
    direc=a
    eps0=0
    eps1=0
    if (diff_ind[0]==0):
        # differentiate wrt a[u]
        direc=a-c
        eps0=1
        eps1=1

    if (diff_ind[0]==1):
        # differentiate wrt b[u]
        direc=b-d
        eps0=1
        eps1=-1

    if (diff_ind[0]==2):
        # differentiate wrt c[u]
        direc=a-c
        eps0=-1
        eps1=1

    if (diff_ind[0]==3):
        # differentiate wrt d[u]
        direc=b-d
        eps0=-1
        eps1=-1


    z_0=fun_g(a-c) + fun_g(b-d)
    z_1=fun_g(a-c) - fun_g(b-d)
    alpha-= 0.25*fun_f(z_0)[u]
    alpha-= 0.25*fun_f(z_1)[u]
    diff_g=eps1*fun_g_diff(direc,u)
    diff=eps0*fun_f_diff(z_0,diff_g) + eps0*fun_f_diff(z_1,diff_g)
    I=space.element(sum([space.element(points.T[i]*diff[i]) for i in range(len(a))]))
    I+=alpha
    return I.copy()


def fun_beta_diff(a,b,c,d,points,diff_ind):
    #diff_ind[0] determins if we differentiate wrt a,b,c or d
    #diff_ind[1] determins wrt which component we differentiate
    u=diff_ind[1]
    alpha=0
    eps0=0
    eps1=0
    if (diff_ind[0]==0):
        # differentiate wrt a[u]
        direc=a-c
        eps0=1
        eps1=1

    if (diff_ind[0]==1):
        # differentiate wrt b[u]
        direc=b-d
        eps0=1
        eps1=-1

    if (diff_ind[0]==2):
        # differentiate wrt c[u]
        direc=a-c
        eps0=-1
        eps1=1

    if (diff_ind[0]==3):
        # differentiate wrt d[u]
        direc=b-d
        eps0=-1
        eps1=-1


    z_0=fun_g(a-c) + fun_g(b-d)
    z_1=fun_g(a-c) - fun_g(b-d)
    alpha-= 0.25*fun_f(z_0)[u]
    alpha-= 0.25*fun_f(z_1)[u]
    diff_g=eps1*fun_g_diff(direc,u)
    diff=eps0*fun_f_diff(z_0,diff_g) - eps0*fun_f_diff(z_1,diff_g)
    I=space.element(sum([space.element(points.T[i]*diff[i]) for i in range(len(a))]))
    I+=alpha
    return I.copy()
#



a=np.array([1.0,0.0])
b=np.array([0.0,1.0])
c=np.array([-1.0,0.0])
d=np.array([0.0,-1.0])

alph=fun_alpha(a,b,c,d,space.points())
alph_diff=fun_alpha_diff(a,b,c,d,space.points(),[0,0])
#alph.show('alpha')
#alph_diff.show('alpha diff')

v=[]
v.append(space.zero())
v.append(-0.5*space.one())
v.append(space.zero())
v.append(space.one())
h=1

def Compute_4_Vectors(a,b,c,d):
    vectors=[]
    vectors.append(fun_g(a-c).copy())
    vectors.append(fun_g_perp(a-c).copy())
    vectors.append(fun_g(b-d).copy())
    vectors.append(fun_g_perp(b-d).copy())

    return vectors.copy()
#

#%%
def ComputeVectorFieldV2(a,b,c,d,h,v):
    # a,b,c,d are the 4 points defining the localization (= geometrical descriptor)
    # h is the scalar control
    # v=[v11,v12,v21,v22] is made of the 4 functions on Omega defining the vector fields (fixed once estimated)
    points=space.points()
    # barycentre of a,b,c,d
    Bary=0.25*(a+b+c+d).copy()

    # points shifted with translation of vector - Bary
    points_temp=points.copy() - Bary.copy()

    #alpha and beta are the coordinate of the points of Omega in the basis of vectors
    # (a-c)/|a-c| = fun_g(a-c) and (b-d)/|b-d| = fun_g(b-d)
    alph=fun_alpha(a,b,c,d,points_temp)
    bet=fun_beta(a,b,c,d,points_temp)

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

    vectors=Compute_4_Vectors(a,b,c,d)
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
    # h_list which is a list of scalar controls and 4 lists (a_list, b_list, c_list, d_list) of points which are
    # the geometrical descriptors

    dim=forward_op.domain.ndim
    space=source_list[0].space

    padded_size = 3 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    # h_list is a list of size nb_vect_fields
    # for each k, h_list[k] is a list of nb_data scalar
    h_list=X[1]
    a_list=X[2]
    b_list=X[3]
    c_list=X[4]
    d_list=X[5]

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

        Bary_i=0.25*(a_list[i]+b_list[i]+c_list[i]+d_list[i]).copy()
        energy3+=sum([Bary_i[j]**2 for j in range(len(Bary_i))])

        vect_field_i=ComputeVectorFieldV2(a_list[i], b_list[i], c_list[i], d_list[i], h_list[i],v_ext).copy()


        temp=_linear_deform(source_list[i],-vect_field_i).copy()
        energy2+=norm(forward_op(temp)-target_list[i])

    print("energy alpha = {}, energy V = {}, norm Bary= {}, energy attach = {} ".format(lamh*energy0, lamv*energy1, lamo*energy3, energy2))
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
a_list=[a,a]
b_list=[b,b]
c_list=[c,c]
d_list=[d,d]
v=[]
v.append(space.zero())
v.append(space.zero())
v.append(space.zero())
v.append(space.zero())
X=[v,h_list,a_list,b_list,c_list,d_list]
#%%
def energyV2_gradient(source_list, target_list, kernel, forward_op,norm, X, lamh, lamv, lamo):
    # X is a list containing (it this order) v which is a list of 4 functions defining the vector field,
    # h_list which is a list of scalar controls and 4 lists (a_list, b_list, c_list, d_list) of points which are
    # the geometrical descriptors

    space=source_list[0].space
    points=space.points()
    dim=forward_op.domain.ndim
    space=source_list[0].space

    padded_size = 3 * space.shape[0]
    padded_ft_fit_op = padded_ft_op(space, padded_size)
    vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(space, kernel)
    ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

    # h_list is a list of size nb_vect_fields
    # for each k, h_list[k] is a list of nb_data scalar
    h_list=X[1]
    a_list=X[2]
    b_list=X[3]
    c_list=X[4]
    d_list=X[5]

    # List of the 4 functions defining the vector field
    v_list=X[0]

    # padding of functions so that the interpolation works even for points
    # slightly out of Omega
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    padded_space=padded_op.range
    v_ext=[padded_space.element(padded_op(v_list[u])) for u in range(len(v_list))]



    grad_op = Gradient(domain=space, method='forward', pad_mode='symmetric')

    nb_data=len(source_list)
    list_vect_field_data=[]

    for i in range(nb_data):

        vect_field_i=ComputeVectorFieldV2(a_list[i], b_list[i], c_list[i], d_list[i], h_list[i],v_ext).copy()

        list_vect_field_data.append(vect_field_i.copy())

    grad_v=[2*lamv*v_list[j] for j in range(len(v_list))]
    grad_h=[]
    grad_a=np.zeros_like(a_list)
    grad_b=np.zeros_like(b_list)
    grad_c=np.zeros_like(c_list)
    grad_d=np.zeros_like(d_list)


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

        vectors_i=Compute_4_Vectors(a_list[i], b_list[i], c_list[i], d_list[i])


        ######## for vect ########
        # the determinent of the matrix (fun_g (a-c), fun_g(b-d)) is needed in the following
        deter_i=vectors_i[0][0]*vectors_i[2][1] - vectors_i[2][0]*vectors_i[2][0]

        Bary_i=0.25*(a_list[i]+b_list[i]+c_list[i]+d_list[i]).copy()

        # points corresponding to T_o ^{-1} (\Omega)
        points_inv=np.empty_like(points)
        points_inv.T[0]=points.T[0]*vectors_i[0][0] + points.T[1]*vectors_i[2][0] + Bary_i[0]
        points_inv.T[1]=points.T[0]*vectors_i[0][1] + points.T[1]*vectors_i[2][1]  + Bary_i[1]

        tmp_dec=space.tangent_bundle.element([tmp_padded[u].interpolation(points_inv.T) for u in range(dim)]).copy()
        tmp1=(2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(tmp_dec) * ft_kernel_fitting).copy()
        tmp1*=(deter_i * h_list[i])

        for j in range(len(v_list)):
            grad_v[j]-=sum([vectors_i[j][u]*tmp1[u] for u in range(dim)])



        ######## for h ########
        grad_h.append(4*lamh*(h_list[i]**2-1)*h_list[i])
        grad_h[i]-= tmp.inner(list_vect_field_data[i])


        ######## for points a,b,c,d ########

        vect_field_derivates=[grad_op(list_vect_field_data[i][uu]) for uu in range(dim)]

        # points shifted with translation of vector - Bary
        points_temp=points.copy() - Bary_i.copy()
        alph=fun_alpha(a_list[i], b_list[i], c_list[i], d_list[i],points_temp)
        bet=fun_beta(a_list[i], b_list[i], c_list[i], d_list[i],points_temp)

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

        diff_g_ac=[fun_g_diff(a_list[i]-c_list[i],uu) for uu in range(dim)]
        diff_g_bd=[fun_g_diff(b_list[i]-d_list[i],uu) for uu in range(dim)]
        diff_g_perp_ac=[fun_g_perp_diff(a_list[i]-c_list[i],uu) for uu in range(dim)]
        diff_g_perp_bd=[fun_g_perp_diff(b_list[i]-d_list[i],uu) for uu in range(dim)]

        ## for a ##
        for d in range(dim):
            alph_der=fun_alpha_diff(a_list[i], b_list[i], c_list[i], d_list[i], points, [0,d]).copy()
            bet_der=fun_beta_diff(a_list[i], b_list[i], c_list[i], d_list[i], points, [0,d]).copy()

            vect_field_a=-space.tangent_bundle.element([vect_field_derivates[uu][0]*alph_der + vect_field_derivates[uu][1]*bet_der for uu in range(dim)]).copy()

            vect_field_a -= space.tangent_bundle.element([h_list[i]*v_interp[0]*diff_g_ac[d][uu] for uu in range(dim)]).copy()
            vect_field_a -= space.tangent_bundle.element([h_list[i]*v_interp[1]*diff_g_perp_ac[d][uu] for uu in range(dim)]).copy()

            grad_a[i][d]+=tmp.inner(vect_field_a)
            grad_a[i][d]+=0.5*lamo*Bary_i[d]

        ## for b ##
        for d in range(dim):
            alph_der=fun_alpha_diff(a_list[i], b_list[i], c_list[i], d_list[i], points, [1,d]).copy()
            bet_der=fun_beta_diff(a_list[i], b_list[i], c_list[i], d_list[i], points, [1,d]).copy()

            vect_field_b=-space.tangent_bundle.element([vect_field_derivates[uu][0]*alph_der + vect_field_derivates[uu][1]*bet_der for uu in range(dim)]).copy()

            vect_field_b -= space.tangent_bundle.element([h_list[i]*v_interp[2]*diff_g_bd[d][uu] for uu in range(dim)]).copy()
            vect_field_b -= space.tangent_bundle.element([h_list[i]*v_interp[3]*diff_g_perp_bd[d][uu] for uu in range(dim)]).copy()

            grad_b[i][d]+=tmp.inner(vect_field_b)
            grad_b[i][d]+=0.5*lamo*Bary_i[d]

        ## for c ##
        for d in range(dim):
            alph_der=fun_alpha_diff(a_list[i], b_list[i], c_list[i], d_list[i], points, [2,d]).copy()
            bet_der=fun_beta_diff(a_list[i], b_list[i], c_list[i], d_list[i], points, [2,d]).copy()

            vect_field_c=-space.tangent_bundle.element([vect_field_derivates[uu][0]*alph_der + vect_field_derivates[uu][1]*bet_der for uu in range(dim)]).copy()

            vect_field_c += space.tangent_bundle.element([h_list[i]*v_interp[0]*diff_g_ac[d][uu] for uu in range(dim)]).copy()
            vect_field_c += space.tangent_bundle.element([h_list[i]*v_interp[1]*diff_g_perp_ac[d][uu] for uu in range(dim)]).copy()

            grad_c[i][d]+=tmp.inner(vect_field_c)
            grad_c[i][d]+=0.5*lamo*Bary_i[d]

        ## for d ##
        for d in range(dim):
            alph_der=fun_alpha_diff(a_list[i], b_list[i], c_list[i], d_list[i], points, [3,d]).copy()
            bet_der=fun_beta_diff(a_list[i], b_list[i], c_list[i], d_list[i], points, [3,d]).copy()

            vect_field_d=-space.tangent_bundle.element([vect_field_derivates[uu][0]*alph_der + vect_field_derivates[uu][1]*bet_der for uu in range(dim)]).copy()

            vect_field_d += space.tangent_bundle.element([h_list[i]*v_interp[2]*diff_g_bd[d][uu] for uu in range(dim)]).copy()
            vect_field_d += space.tangent_bundle.element([h_list[i]*v_interp[3]*diff_g_perp_bd[d][uu] for uu in range(dim)]).copy()

            grad_d[i][d]+=tmp.inner(vect_field_d)
            grad_d[i][d]+=0.5*lamo*Bary_i[d]

    return [grad_v, grad_h, grad_a, grad_b, grad_c, grad_d]
#
#%%

forward_op = odl.IdentityOperator(space)

nb_data=5

source_list=[]
target_list=[]

for i in range(nb_data):
    source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses_source[i].copy(),3)))
    target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses_target[i].copy(),3)))

#source_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[3],3)))
#target_list.append(space.element(scipy.ndimage.filters.gaussian_filter(images_ellipses[4],3)))

norm = odl.solvers.L2NormSquared(forward_op.range)
#import random
X_init=[[],[],[],[],[],[]]
nb_vect_fields=1

X_init[0]=[space.zero() for uu in range(4)]
temph=[]
tempa=[]
tempb=[]
tempc=[]
tempd=[]
for k in range(nb_data):
    temph.append(1)
    tempa.append(np.array([1,0]))
    tempb.append(np.array([0,1]))
    tempc.append(np.array([-1,0]))
    tempd.append(np.array([0,-1]))

X_init[1]=temph.copy()
X_init[2]=tempa.copy()
X_init[3]=tempb.copy()
X_init[4]=tempc.copy()
X_init[5]=tempd.copy()

#energy(source_list, target_list,kernel, forward_op,norm, X)
#grad=energy_gradient(source_list, target_list,kernel, forward_op,norm, X)



#%% Gradient descent
lamh=1e-5
lamv=1e1
lamo=1e-5
X=X_init.copy()
ener=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X, lamh, lamv, lamo)
print('Initial energy = {}'.format(ener))
niter=200
eps=0.02
eps0=eps
eps1=eps
eps2=eps
#eps3=eps
cont=0
for i in range(niter):
    if (cont==0):
        grad=energyV2_gradient(source_list, target_list,kernel, forward_op,norm, X, lamh, lamv, lamo)

    X_temp=X.copy()
    X_temp[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(4)].copy()
    X_temp[1]=[X[1][uu] - eps1*grad[1][uu] for uu in range(nb_data)]
    for k in range(4):
        X_temp[k+2]=[np.array(X[k+2][uu])- eps2*np.array(grad[k+2][uu]) for uu in range(nb_data)]

    ener_temp=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp, lamh, lamv, lamo)

    if (ener_temp<ener):
        X=X_temp.copy()
        ener=ener_temp
        print('Iter = {},  energy = {},  eps0={} ,  eps1={} ,  eps2={}  '.format(i,ener,eps0,eps1,eps2))
        eps0*=1.2
        eps1*=1.2
        eps2*=1.2
        cont=0
        #eps3*=1.2
    else:
        X_temp0=X.copy()
        X_temp0[0]=[X[0][uu] - 0.5*eps0*grad[0][uu] for uu in range(4)].copy()
        X_temp0[1]=[X[1][uu] - eps1*grad[1][uu] for uu in range(nb_data)]
        for k in range(4):
            X_temp0[k+2]=[np.array(X[k+2][uu])- eps2*np.array(grad[k+2][uu]) for uu in range(nb_data)]
        ener_temp0=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp0, lamh, lamv, lamo)

        X_temp1=X.copy()
        X_temp1[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(4)].copy()
        X_temp1[1]=[X[1][uu] - 0.5*eps1*grad[1][uu] for uu in range(nb_data)]
        for k in range(4):
            X_temp1[k+2]=[np.array(X[k+2][uu])- eps2*np.array(grad[k+2][uu]) for uu in range(nb_data)]
        ener_temp1=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp1, lamh, lamv, lamo)

        X_temp2=X.copy()
        X_temp2[0]=[X[0][uu] - eps0*grad[0][uu] for uu in range(4)].copy()
        X_temp2[1]=[X[1][uu] - eps1*grad[1][uu] for uu in range(nb_data)]
        for k in range(4):
            X_temp2[k+2]=[np.array(X[k+2][uu])- 0.5*eps2*np.array(grad[k+2][uu]) for uu in range(nb_data)]
        ener_temp2=energyEstimateV2(source_list, target_list,kernel, forward_op,norm, X_temp2, lamh, lamv, lamo)

        if (ener_temp0 < ener_temp1 and ener_temp0 < ener_temp2):
            X_temp=X_temp0.copy()
            eps0*=0.5
            ener_temp=ener_temp0
        else:
            if(ener_temp1 < ener_temp0 and ener_temp1 < ener_temp2):
                X_temp=X_temp1.copy()
                eps1*=0.5
                ener_temp=ener_temp1
            else:
                X_temp=X_temp2.copy()
                eps2*=0.5
                ener_temp=ener_temp2

        if (ener_temp<ener):
            X=X_temp.copy()
            ener=ener_temp
            print('Iter = {},  energy = {},  eps0={} ,  eps1={} ,  eps2={} , '.format(i,ener,eps0,eps1,eps2))
            eps0*=1.2
            eps1*=1.2
            eps2*=1.2
            cont=0

        else:
            cont=1
            eps0*=0.5
            eps1*=0.5
            eps2*=0.5


        print('Iter = {},  eps = {}'.format(i,eps))
#
#%%
nb_datamax=nb_data

for n in range(nb_datamax):
    #space.element(source_list[n]).show('Source {}'.format(n))
    space.element( source_list[n] - target_list[n]).show('Initial difference {}'.format(n))

for n in range(nb_data):
    vect_field_n=ComputeVectorFieldV2(X[2][n],X[3][n],X[4][n],X[5][n],X[1][n],X[0])
    temp=_linear_deform(source_list[n],-vect_field_n).copy()
    (space.element(temp)-space.element(target_list[n])).show('Transported source {}'.format(n))
    vect_field_n.show('Vector field {}'.format(n))
#

#%% Save vector field estimated

np.savetxt('/home/barbara/DeformationModulesODL/deform/vect_field_rotation_Rigid',X[0])

np.savetxt('/home/barbara/DeformationModulesODL/deform/vect_field_ellipses_Rigid',X[0])

vect_field=space.tangent_bundle.element(np.loadtxt('/home/barbara/DeformationModulesODL/deform/vect_field_rotation_Rigid')).copy()

import matplotlib.pyplot as plt
points=space.points()
#v=X[0]
v=vect_field_n.copy()
plt.figure()
plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
plt.axis('equal')
plt.title('v n')

#%%
for u in range(4):
    X_temp[0][u].show('{}'.format(u))
#






