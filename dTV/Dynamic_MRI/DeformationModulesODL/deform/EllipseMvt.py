#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:44:54 2017

@author: barbara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:52:24 2017

@author: bgris
"""


# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

import numpy as np

from odl.discr import DiscreteLp, Gradient, Divergence
from odl.operator import Operator, PointwiseInner
from odl.space import ProductSpace
from odl.deform.linearized import _linear_deform
from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
from odl.space.fspace import FunctionSpace
import odl
from odl.set import LinearSpace, LinearSpaceElement, Set, Field
from odl.deform import TemporalAttachmentLDDMMGeom
from odl.deform import ShootTemplateFromVectorFields
from odl.solvers.functional.functional import Functional
from DeformationModulesODL.deform.DeformationModuleAbstract import DeformationModule
__all__ = ('EllipseMvt', )



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
        padded_op.range, halfcomplex=False, shift=shifts, impl='pyfftw')

    return ft_op * padded_op

def fitting_kernel(space, kernel):

    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return discretized_kernel



class EllipseMvt(DeformationModule):
    def __init__(self,DomainField, Name, Kernel):
        """Initialize a new instance.
        DomainField : space on wich vector fields will be defined
        Name : link to a saved generated vector field from which we build everything
        Kernel : kernel, class that has at least methods Eval and derivative
        """

        self.KernelClass=Kernel
        #self.Kernel=Kernel.Eval
        def kernelOpFun(x):
            return Kernel.Eval(x)
        self.Kernel=kernelOpFun
        self.dim=DomainField.ndim
        vect_field=DomainField.tangent_bundle.element(np.loadtxt(Name)).copy()
        self.vect_field=vect_field.copy()
        
        #Derivative with respect to the geometrical descriptor : 
        # corresponds to (v(x- delta) - v(x)) / delta
        vect_field_der=[]
        
        for i in range(DomainField.ndim):
            delta=2*DomainField.cell_sides[i]
            dec_temp=DomainField.tangent_bundle.zero()
            dec_temp[i]=(delta)*DomainField.one()
            temp=DomainField.tangent_bundle.element(
                    [_linear_deform(vect_field[d],-dec_temp) for d in range(DomainField.ndim)]).copy()
            vect_field_der.append((1/delta)*(temp - vect_field).copy())

        self.vect_field_der=vect_field_der.copy()

        # Each affine deformation is defined by a centre (a point, GD) and
        # dim+1 vectors (the controls)
        GDspace=odl.space.rn(self.dim)
        Contspace=odl.space.rn(1)

        basis=[]
        for d in range(self.dim):
            a=GDspace.zero()
            a[d]=1
            basis.append(a.copy())

        basisGD=basis.copy()

        basisCont=[Contspace.element(1)]

        super().__init__(GDspace,Contspace,basisGD,basisCont,DomainField)


    def ComputeField(self, o,h):
        """Return the computed vector field on DomainField
        """
        if o not in self.GDspace:
            try:
                o = self.GDspace.element(o).copy()
            except (TypeError, ValueError) as err:
                raise TypeError(' o is not in `GDspace` instance'
                            '')

        if h not in self.Contspace:
            try:
                o = self.Contspace.element(o).copy()
            except (TypeError, ValueError) as err:
                raise TypeError(' h is not in `Contspace` instance'
                            '')

        temp=self.DomainField.tangent_bundle.element([self.DomainField.zero() - o[i] for i in range(self.dim)]).copy()
        # the control h is of dimension 1
        vector_field=h[0]*self.DomainField.tangent_bundle.element([_linear_deform(self.vect_field[d],temp) for d in range(self.dim)]).copy()

        return vector_field
    """
    @property
    def ComputeFieldEvaluate(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(odl.space.rn(ope.dim), odl.space.rn(ope.dim),
                                 linear=False)


            def _call(self,x):
                y=ope.GDspace.element(x)
                speed=self.Cont*odl.space.rn(ope.dim).element([
                        ope.vect_field[i].interpolation(np.array(y-self.GD)) for i in range(ope.dim)])
                
                return speed

        return Eval
    """
    
    @property
    def ComputeFieldDer(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(ope.GDspace, ope.DomainField.tangent_bundle,
                                 linear=True)


            def _call(self,dGD):
                dGD=ope.GDspace.element(dGD).copy()
                # the control self.Cont is of dimension 1
                vector_field=ope.DomainField.tangent_bundle.element(
                        sum(self.Cont[0]*dGD[i]*ope.vect_field_der[i] for i in range(ope.dim))).copy()


                return vector_field

        return Eval


    """
    @property
    def ComputeFieldDerEvaluate(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(odl.ProductSpace(ope.GDspace,odl.space.rn(ope.dim)), odl.space.rn(ope.dim),
                                 linear=True)


            def _call(self,X):
                dGD=ope.GDspace.element(X[0]).copy()
                x=odl.space.rn(ope.dim).element(X[1])
                speed=odl.space.rn(ope.dim).zero()
                for i in range(ope.Ntrans):
                    kern = ope.KernelClass.derivative([xd - ou for xd, ou in zip(x, self.GD[i])])
                    speed+=kern.Eval(dGD[i])*self.Cont[i]
                return speed

        return Eval
    """

    def ApplyVectorField(self,GD,vect_field):
            #GD=self.GDspace.element(X[0])
            #vect_field=self.DomainField.tangent_bundle.element(X[1])
            #speed=self.GDspace.element()
            #for u in range(self.dim):
            #   for i in range(len(GD)):
            #       speed[i][u]=vect_field[u].interpolation(GD[i])
            speed=self.GDspace.element([
                    vect_field[i].interpolation(np.array(GD)) for i in range(self.dim)])
           
            return speed

    """@property
    def ApplyModule(self):
        ope = self
        class apply(Operator):
            def __init__(self,Module,GDmod,Contmod):
                self.apply_op=Module.ComputeFieldEvaluate(GDmod,Contmod)
                super().__init__(ope.GDspace, ope.GDspace,
                                 linear=False)

            def _call(self,GD):
                speed=ope.GDspace.element()
                for i in range(len(GD)):
                    speed[i]=self.apply_op(GD[i])
                return speed
        return apply
    """
    
    def Cost(self,GD,Cont):

        # the control Cont is of dimension 1
        return Cont[0]**2



    @property
    def CostDerGD(self):
        ope = self
        class compute(Functional):
           def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(ope.GDspace,
                                 linear=True)
           def _call(self,dDG):
                energy=0

                return energy
        return compute


    """
    @property
    def CostDerCont(self):
        ope = self
        class compute(Functional):
           def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                super().__init__(ope.Contspace,
                                 linear=True)
           def _call(self,dCont):
                dCont=ope.Contspace.element(dCont).copy()
                energy=0
                for i in range(ope.Ntrans):
                    for j in range(ope.Ntrans):
                        prod=[hi*hj for hi, hj in zip(self.Cont[i],dCont[j])]
                        energy+=ope.Kernel(self.GD[i]-self.GD[j])*2*sum(prod)

                return energy
        return compute
    """



    def CostGradGD(self,GD,Cont):
        grad=self.GDspace.zero()

        return grad


    
    def CostGradCont(self,GD,Cont):
        grad=2*Cont.copy()

        return grad
    
