#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:36:13 2017

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
from DeformationModulesODL.deform import TemporalAttachmentModulesGeom

import scipy



#%% Generate data



# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[128,128],
    dtype='float32', interp='linear')




ellipses= [[1.50, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.2, .1100, .3100, 0.2200, 0.0000, -18],
            [-.2, .1600, .4100, -.2200, 0.0000, 18],
            [0.1, .2100, .2500, 0.0000, 0.3500, 0],
            [0.1, .0460, .0460, 0.0000, 0.1000, 0],
            [0.1, .0460, .0460, 0.0000, -.1000, 0],
            [0.8, .0460, .0330, -.0800, -.6050, 0],
            [0.8, .0330, .0330, 0.0000, -.6060, 0],
            [0.8, .0330, .0460, 0.0600, -.6050, 0]]






phantom=odl.phantom.geometric.ellipsoid_phantom(space,ellipses)
fac_smooth=0.8
template=space.element(scipy.ndimage.filters.gaussian_filter(phantom.asarray(),fac_smooth))
template.show()

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
#
#%%

maxx=2.0
minx = -2.7
miny=-12
maxy=-7

theta=np.pi/3
centre=np.array([0,-9.6])

points=space.points()

def Rot_image_cache(minx,maxx,miny,maxy,theta,centre,template):
    I1=space.element()
    for i in range(len(points)):
        pt=points[i]
        if(pt[0]>minx and pt[0] < maxx and pt[1]>miny and pt[1]<maxy):
            pt_rot_inv=Rtheta(-theta,pt-centre).copy()
            I1[i]=template.interpolation([[pt_rot_inv[0]+centre[0]],[pt_rot_inv[1]+centre[1]]])
        else:
            I1[i]=template[i]

    return I1
I1=Rot_image_cache(minx,maxx,miny,maxy,theta,centre,template)
#I1.show()

#%%
theta_list=[0 , 15, -20, 30, -50, 45, -10, 20, -30]
theta_list=[0 , 10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
theta_list=np.pi*np.array([0 , 0.1, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
nb_data=len(theta_list)
theta_dec=0.05*np.pi
images_source=[]
images_target=[]
for i in range(nb_data):
    images_source.append(Rot_image_cache(minx,maxx,miny,maxy,theta_list[i],centre,template).copy())
    images_target.append(Rot_image_cache(minx,maxx,miny,maxy,theta_list[i]+theta_dec,centre,template).copy())
#











