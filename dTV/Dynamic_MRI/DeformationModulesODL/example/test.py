#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:23:47 2017

@author: bgris
"""

#
#ModuleF.CostDerGD(GD_init,Cont)(Cont)
#Module.CostGradGD(GD_init,Cont)
#functionalF.gradient([GD_init,Cont_init])

#a=[]
#for i in range(8000):
#    a.append(space.zero())
#

import timeit
#
#start = timeit.default_timer()
#basisContTraj=[]
#for i in range(10):
#    temp=[]
#    for u in range(functional.N+1):
#        #basisContTraj[i][u]=functional.Module.basisCont[i].copy()
#        temp.append([])
#    basisContTraj.append(temp.copy())
#
#
#start = timeit.default_timer()
#for i in range(800):
#    temp=[]
#    for u in range(functional.N+1):
#
#        temp.append(functional.Module.basisCont[i].copy())
#    basisContTraj.append(temp.copy())
##
#end = timeit.default_timer()
#print(end - start)
#%%

import timeit

start = timeit.default_timer()
gamma=functional([GD_init,Cont_init])

end = timeit.default_timer()
print(end - start)