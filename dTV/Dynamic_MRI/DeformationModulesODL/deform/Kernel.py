#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:54:53 2017

@author: bgris
"""


import numpy as np
__all__ = ('GaussianKernel', )

class GaussianKernel(object):
    def __init__(self,scale):
        self.scale=scale


    def Eval(self,x):
        scaled = [xi ** 2 / (2 * self.scale ** 2) for xi in x]
        return np.exp(-sum(scaled))

#    @property
#    def Eval(self):
#        ker = self
#        class ComputeEval(object):
#            def __init__(self):
#                self.scale=ker.scale
#            def Ev(self,x):
#                scaled = [xi ** 2 / (2 * self.scale ** 2) for xi in x]
#                return np.exp(-sum(scaled))
#        return ComputeEval

    @property
    def derivative(self):
        ker=self
        class ComputeDer(object):
            def __init__(self,x0):
                self.x0=x0
            def Eval(self,dx):
                a=ker.Eval(self.x0)
                b=[-xi*dxi/( ker.scale ** 2) for xi, dxi in zip(self.x0,dx) ]
                return a*sum(b)
        return ComputeDer


    def gradient(self,x0):
        #a=self.Eval(x0)
        scaled = [xi ** 2 / (2 * self.scale ** 2) for xi in x0]
        return [-(1/(self.scale ** 2))*np.exp(-sum(scaled))*xi for xi in x0]



