#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:33:17 2017

@author: bgris
"""



import numpy as np
import matplotlib.pyplot as plt
import odl
from odl.deform.linearized import _linear_deform
from odl.discr import DiscreteLp, Gradient, Divergence
from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
from odl.space import ProductSpace
import numpy as np
import scipy



if False:
    path='/home/bgris/data/SheppLoganRotationSmallDef/'
    name='vectfield_smalldef_sigma_0_3'
    for i in range(nb_data):
        name_i=path + name + '_{}'.format(i)
        np.savetxt(name_i,vectorfield_list_center[i])



#vectorfield_list_center_save=copy.deepcopy(vectorfield_list_center)



# Initialize random number generator
np.random.seed(123)


# We suppose that we have a list of vector fields vector_field_list

#%%
## Set parameters
# Size of dataset
size = 10


# noise of observation
sigmanoise=0.2

# scale of the kernel
sigma_kernel=0.3
fac=2
xmin=-2.2
xmax=3.2
dx=round((xmax-xmin)/(fac*sigma_kernel))
ymin=-2.0
ymax=2.0
dy=round((ymax-ymin)/(fac*sigma_kernel))
x0=[]
for i in range(dx+1):
    for j in range(dy+1):
        x0.append(xmin +fac*sigma_kernel* i*1.0)
        x0.append(ymin + fac*sigma_kernel*j*1.0)
#Number of translations
nbtrans=round(0.5*len(x0))

def kernel(x):
    scaled = [xi ** 2 / (2 * sigma_kernel ** 2) for xi in x]
    return np.exp(-sum(scaled))

if False:
    space=odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[128,128],
    dtype='float32', interp='linear')
    vectorfield_list_center=[]
    path='/home/bgris/data/SheppLoganRotationSmallDef/'
    name='vectfield_smalldef'
    for i in range(size):
        name_i=path + name + '_{}'.format(i)
        vect_field_load_i_test=space.tangent_bundle.element(np.loadtxt(name_i)).copy()
        vectorfield_list_center.append(vect_field_load_i_test.copy())

if False:
    vectorfield_list_center[0].show()
    plt.plot(x0[::2],x0[1::2],'x')

#
#%% Useful functions

# Returns the rigid transformation of points x and vectors alpha
# by the transformation parametrized by lam, theta,c
def Rigidtransformation(x,alpha, lam, theta,c):
    alpha_def=[]
    x_def=[]
    for i in range(nbtrans):
        x_def.append(x[2*i]*np.cos(theta) - x[2*i + 1]*np.sin(theta) + c[0])
        x_def.append(x[2*i]*np.sin(theta) + x[2*i + 1]*np.cos(theta) + c[1])
        alpha_def.append(lam*(alpha[2*i]*np.cos(theta) - alpha[2*i + 1]*np.sin(theta) ))
        alpha_def.append(lam*(alpha[2*i]*np.sin(theta) + alpha[2*i + 1]*np.cos(theta) ))
    return [x_def.copy(),alpha_def.copy()].copy()

# function taking into account the centres and vectors for the vector field,
# as well as lists of parameters for the rigid deformation (translation vector,
# rotation angle and lambda) and return a list of corresponding vector fields
# (as lists of 2 arrays)
# In order to be used in the statistical study, all parameters are column vectors
def Computevectorfieldslists(x,alpha,lambdalist,thetalist,clist):
    mg = space.meshgrid
    lis=[]
    for k in range(size):
        lam=lambdalist[k]
        theta=thetalist[k]
        c=np.array([clist[2*k], clist[2*k+1]])
        param_def_k=Rigidtransformation(x,alpha, lam, theta,c)
        x_def_k=param_def_k[0].copy()
        alpha_def_k=param_def_k[1].copy()

        i=0
        pt=[x_def_k[2*i],x_def_k[2*i+1]].copy()
        kern0 = alpha_def_k[2*i]*kernel([mgu - ou for mgu, ou in zip(mg, pt)]).copy()
        kern1 = alpha_def_k[2*i+1]*kernel([mgu - ou for mgu, ou in zip(mg, pt)]).copy()
        for i in range(1,nbtrans):
            pt=[x_def_k[2*i],x_def_k[2*i+1]].copy()
            kern0 += alpha_def_k[2*i]*kernel([mgu - ou for mgu, ou in zip(mg, pt)]).copy()
            kern1 += alpha_def_k[2*i+1]*kernel([mgu - ou for mgu, ou in zip(mg, pt)]).copy()
        #lis.append([kern0.copy(),kern1.copy()].copy())
        lis.append(kern0.copy())
        lis.append(kern1.copy())
    return lis
#

#%% Generate data
#
#space = odl.uniform_discr(
#    min_pt=[-16, -16], max_pt=[16, 16], shape=[128,128],
#    dtype='float32', interp='linear')
#
#lambdalist_gen= 1*np.random.randn(size)+1
#thetalist_gen=np.random.uniform(0,2*np.pi,size)
#clist_gen=0.5*np.random.randn(2*size)

#x_gen=[-1,1,1,0.5]
#alpha_gen=[-1,0,1,0]

#list_vectfields_gen=Computevectorfieldslists(x_gen,alpha_gen,lambdalist_gen,thetalist_gen,clist_gen)

list_vectfields_gen=[]
for i in range(size):
    list_vectfields_gen.append(vectorfield_list_center[i][0].asarray().copy())
    list_vectfields_gen.append(vectorfield_list_center[i][1].asarray().copy())


#x0=[-1,1,-1,-1,0,-1,0,1,1,-1,1,1,2,-1,2,1]
#%%


import pymc3 as pm


basic_model = pm.Model()
#
#x0=[-1,0,1,0]
with basic_model:

    mu_lambda=pm.Normal('mu_lambda',mu=1,sd=1)
    sigma_lambda=pm.HalfCauchy('sigma_lambda', beta=1)

    # Priors for unknown model parameters
    lambdalist_model=pm.Normal('lambda',mu=mu_lambda,sd=sigma_lambda, shape=size)
    thetalist_model=pm.Uniform('theta',lower=0,upper=2*np.pi,shape=size)
    clist_model=pm.Normal('c',mu=0,sd=1,shape=2*size)

    #x_model=pm.Normal('x',mu=x0,sd=5,shape=2*nbtrans)
    alpha_model=pm.Normal('alpha',mu=0,sd=5,shape=2*nbtrans)

    #mu = pm.Deterministic('mu', pm.backends.ndarray([loc_trans(sigma_kernel, np.array([T[2*i]+ centre[0], T[2*i+1]+ centre[1]])) for i in range(size)]))
    #mu=np.array([loc_trans(sigma_kernel, np.array([T[2*i]+ centre[0], T[2*i+1]+ centre[1]])) for i in range(size)])
    #mu=loc_trans(sigma_kernel, np.array([T[2*0]+ centre[0], T[2*0+1]+ centre[1]]))
    # Likelihood (sampling distribution) of observations
    #Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y[0])
    Y_obs=pm.Normal('Y_obs', mu=Computevectorfieldslists(
            x0,alpha_model,lambdalist_model,thetalist_model,clist_model),
            sd=sigmanoise, observed=list_vectfields_gen)
#
#x0=[-1,0,1,0]
#with basic_model:
#
#    # Priors for unknown model parameters
#    lambdalist_model=pm.Normal(mu=1,sd=1, shape=size)
#    thetalist_model=pm.Uniform(lower=0,upper=2*np.pi,shape=size)
#    clist_model=pm.Normal(mu=0,sd=1,shape=2*size)
#
#    #x_model=pm.Normal('x',mu=x0,sd=5,shape=2*nbtrans)
#    alpha_model=pm.Normal('alpha',mu=0,sd=5,shape=2*nbtrans)
#
#    #mu = pm.Deterministic('mu', pm.backends.ndarray([loc_trans(sigma_kernel, np.array([T[2*i]+ centre[0], T[2*i+1]+ centre[1]])) for i in range(size)]))
#    #mu=np.array([loc_trans(sigma_kernel, np.array([T[2*i]+ centre[0], T[2*i+1]+ centre[1]])) for i in range(size)])
#    #mu=loc_trans(sigma_kernel, np.array([T[2*0]+ centre[0], T[2*0+1]+ centre[1]]))
#    # Likelihood (sampling distribution) of observations
#    #Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y[0])
#    Y_obs=pm.Normal('Y_obs', mu=Computevectorfieldslists(
#            x0,alpha_model,lambdalist_model,thetalist_model,clist_model),
#            sd=sigmanoise, observed=list_vectfields_gen)
##

#%% MAP

map_estimate = pm.find_MAP(model=basic_model)

#%% Collect result and generate corresponding vector fields
lambdalist_est=list(map_estimate.values())[1]
clist_est=list(map_estimate.values())[2]
thetalist_est=list(map_estimate.values())[4]
alpha_est=list(map_estimate.values())[3]
#x_est=list(map_estimate.values())[3]
x_est=x0.copy()
list_vectfields_est=Computevectorfieldslists(x_est,alpha_est,lambdalist_est,thetalist_est,clist_est)


#%% See result
i=0
# plot generated vector fields
for i in range(size):
    #((space.tangent_bundle.element([list_vectfields_gen[2*i],list_vectfields_gen[2*i+1]]) -space.tangent_bundle.element([list_vectfields_est[2*i],list_vectfields_est[2*i+1]]))).show('{}'.format(i))
    ((space.tangent_bundle.element([list_vectfields_est[2*i],list_vectfields_est[2*i+1]]))).show(' est {}'.format(i))
    ((space.tangent_bundle.element([list_vectfields_gen[2*i],list_vectfields_gen[2*i+1]]))).show('gen {}'.format(i))
#

# plot x and alpha
plt.quiver(x_gen[::2],x_gen[1::2],alpha_gen[::2],alpha_gen[1::2],color='r')
plt.quiver(x_est[::2],x_est[1::2],alpha_est[::2],alpha_est[1::2],color='r')



plt.plot(x0[::2],x0[1::2],'o')


#%%
with basic_model:
    #trace = pm.sample(500, step=pm.Metropolis())
    trace = pm.sample(300, tune=100)

#
#%%
lambdalist_est_sim=trace.get_values('lambda')
clist_est_sim=trace.get_values('c')
thetalist_est_sim=trace.get_values('theta')
alphalist_est_sim=trace.get_values('alpha')

lambdalist_est_exp=sum(lambdalist_est_sim)/500
clist_est_exp=sum(clist_est_sim)/500
thetalist_est_exp=sum(thetalist_est_sim)/500
alphalist_est_exp=sum(alphalist_est_sim)/500


x_est=x0.copy()

list_vectfields_est=Computevectorfieldslists(x0,alphalist_est_exp,lambdalist_est_exp,thetalist_est_exp,clist_est_exp)


plt.quiver(x_est[::2],x_est[1::2],alphalist_est_exp[::2],alphalist_est_exp[1::2],color='r')

#%%


param_trsf_i=Rigidtransformation(x_est,alphalist_est_exp, lambdalist_est_exp[i], thetalist_est_exp[i],[clist_est_exp[2*i],clist_est_exp[2*i+1]])
x_trsf_exp_i=param_trsf_i[0]
alpha_trsf_exp_i=param_trsf_i[1]
#%%
i=0
points=space.points()
import matplotlib.pyplot as plt
#v=space.tangent_bundle.element([list_vectfields_gen[2*i],list_vectfields_gen[2*i+1]]).copy()
v=space.tangent_bundle.element([list_vectfields_est[2*i],list_vectfields_est[2*i+1]]).copy()
plt.figure()
plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
plt.axis('equal')
plt.title('v es exp i {}'.format(i))
#


plt.quiver(x_trsf_exp_i[::2],x_trsf_exp_i[1::2],alpha_trsf_exp_i[::2],alpha_trsf_exp_i[1::2],color='r')
#plt.plot(x_trsf_exp[::2],x_trsf_exp[1::2],'o')


i=0
points=space.points()
import matplotlib.pyplot as plt
#v=space.tangent_bundle.element([list_vectfields_gen[2*i],list_vectfields_gen[2*i+1]]).copy()
v=space.tangent_bundle.element([list_vectfields_gen[2*i],list_vectfields_gen[2*i+1]]).copy()
plt.figure()
plt.quiver(points.T[0][::20],points.T[1][::20],v[0][::20],v[1][::20])
plt.axis('equal')
plt.title('v es gen i {}'.format(i))
#


#%%
i=0
vect_field_ref=space.tangent_bundle.element([list_vectfields_est[2*i],list_vectfields_est[2*i+1]]).copy()
vect_field_ref.show('ref')
np.savetxt('/home/bgris/DeformationModulesODL/deform/vect_field_rotation_SheppLogan_V6_sigma_lddmm_0_3_sigma_cp_0_3_nbtrans_80_HiddenVariables_expectedvalue',vect_field_ref)





