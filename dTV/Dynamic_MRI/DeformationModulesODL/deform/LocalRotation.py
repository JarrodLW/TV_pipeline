#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:41:29 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:23:16 2017

@author: bgris
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
__all__ = ('LocalRotation', )



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



class LocalRotation(DeformationModule):
    def __init__(self,DomainField, NRotation, Kernel):
        """Initialize a new instance.
        DomainField : space on wich vector fields will be defined
        NRotation : number of NRotations
        Kernel : kernel, class that has at least methods Eval and derivative
        """

        self.NRotation=NRotation
        self.KernelClass=Kernel
        #self.Kernel=Kernel.Eval
        def kernelOpFun(x):
            return Kernel.Eval(x)
        self.Kernel=kernelOpFun
        self.dim=DomainField.ndim

        Directions=odl.ProductSpace(odl.space.rn(self.dim),self.dim +1).element()

        if(self.dim==2):
            Directions[0][0]=0
            Directions[0][1]=1
            Directions[1][0]=0.866
            Directions[1][1]=-0.5
            Directions[2][0]=-0.866
            Directions[2][1]=-0.5
            Directions=(0.3*self.KernelClass.scale)*Directions
        elif(self.dim==3):
            Directions[0][0]=-1
            Directions[0][1]=-1
            Directions[0][2]=1
            Directions[1][0]=-1
            Directions[1][1]=1
            Directions[1][2]=-1
            Directions[2][0]=1
            Directions[2][1]=-1
            Directions[2][2]=-1
            Directions[3][0]=1
            Directions[3][1]=1
            Directions[3][2]=1
            Directions=(0.3*self.KernelClass.scale/1.732)*Directions
        else:
            print('LocalRotation only for dimension 2 or 3')

        self.DirectionsPts=Directions.copy()



        if(self.dim==2):
            Directions[0][0]=-1
            Directions[0][1]=0
            Directions[1][0]=0.5
            Directions[1][1]=0.866
            Directions[2][0]=0.5
            Directions[2][1]=-0.866
            Directions=Directions
        elif(self.dim==3):
            print('LocalRotation not yet implemented for dimension 3')
            Directions[0][0]=-1
            Directions[0][1]=-1
            Directions[0][2]=1
            Directions[1][0]=-1
            Directions[1][1]=1
            Directions[1][2]=-1
            Directions[2][0]=1
            Directions[2][1]=-1
            Directions[2][2]=-1
            Directions[3][0]=1
            Directions[3][1]=1
            Directions[3][2]=1
            Directions=(1/1.732)*Directions
        else:
            print('LocalRotation only for dimension 2 or 3')


        self.DirectionsVec=Directions.copy()

        # Each affine deformation is defined by a centre (a point, GD) and
        # dim+1 vectors (the controls)
        GDspace=odl.ProductSpace(odl.space.rn(self.dim),self.NRotation)
        Contspace=odl.space.rn(self.NRotation)

        basis=[]
        for i in range(self.NRotation):
            for d in range(self.dim):
                a=GDspace.zero()
                a[i][d]=1
                basis.append(a.copy())

        basisGD=basis.copy()

        basisCont=[]
        for i in range(self.NRotation):
            a=Contspace.zero()
            a[i]=1
            basisCont.append(a.copy())

        basisCont=basisCont.copy()

        super().__init__(GDspace,Contspace,basisGD,basisCont,DomainField)


    def ComputeToolPoints(self,o):
         """ The Rotation deformations are built as sum of local translations
         centred around the GDs. Here we compute these points"""

         TP=odl.ProductSpace(odl.ProductSpace(odl.space.rn(self.dim),self.dim+1),self.NRotation).element()
         for i in range(self.NRotation):
             TP[i]=(o[i]+self.DirectionsPts).copy()

         return TP



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

        vector_field=self.DomainField.tangent_bundle.zero()
        TP=self.ComputeToolPoints(o)
        mg = self.DomainField.meshgrid
        for i in range(self.NRotation):
            for u in range(self.dim +1):
                kern = self.Kernel([mgu - ou for mgu, ou in zip(mg, TP[i][u])])
                vector_field += self.DomainField.tangent_bundle.element([h[i]*kern * hu for hu in self.DirectionsVec[u]])

        return vector_field

    @property
    def ComputeFieldEvaluate(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                self.TP=ope.ComputeToolPoints(self.GD)
                super().__init__(odl.space.rn(ope.dim), odl.space.rn(ope.dim),
                                 linear=False)


            def _call(self,x):
                speed=odl.space.rn(ope.dim).zero()
                for i in range(ope.NRotation):
                    for u in range(ope.NRotation):
                        a=ope.Kernel(self.TP[i][u]-x)
                        speed+=a*self.Cont[i]*ope.DirectionsVec[u]

                return speed

        return Eval


    @property
    def ComputeFieldDer(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                self.TP=ope.ComputeToolPoints(GD)
                #self.TPDer_op=TP=self.ComputeToolPointDer(o)
                super().__init__(ope.GDspace, ope.DomainField.tangent_bundle,
                                 linear=True)


            def _call(self,dGD):
                dGD=ope.GDspace.element(dGD).copy()
                vector_field=ope.DomainField.tangent_bundle.zero()
                mg = ope.DomainField.meshgrid

                for i in range(ope.NRotation):
                    for u in range(ope.dim +1):
                        kern = ope.KernelClass.derivative([mgu - ou for mgu, ou in zip(mg, self.TP[i][u])])
                        vector_field += ope.DomainField.tangent_bundle.element([self.Cont[i]*kern.Eval(dGD[i]) * hu for hu in ope.DirectionsVec[u]])

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
            speed=self.GDspace.element(np.array([vect_field[i].interpolation(np.array(GD).T) for i in range(self.dim)]).T)

            return speed

    @property
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

    def Cost(self,GD,Cont):
        energy=0
        TP=self.ComputeToolPoints(GD)
        for i in range(self.NRotation):
            for j in range(self.NRotation):
                for u in range(self.dim+1):
                    for v in range(self.dim+1):
                        fac=self.Kernel(TP[i][u]-TP[j][v])
                        prod=[hi*hj for hi, hj in zip(self.DirectionsVec[u],self.DirectionsVec[v])]
                        energy+=fac*Cont[i]*Cont[j]*sum(prod)
        return energy


    """
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
                dGD=ope.GDspace.element(dDG).copy()
                energy=0
                for i in range(ope.Ntrans):
                    for j in range(ope.Ntrans):
                        prod=[hi*hj for hi, hj in zip(self.Cont[i],self.Cont[j])]
                        kern = ope.KernelClass.derivative([xd - ou for xd, ou in zip(self.GD[j], self.GD[i])])
                        energy+=kern.Eval(dGD[j]-dGD[i])*sum(prod)

                return energy
        return compute



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




    def CostGradGD(self,GD,Cont):
        grad=self.GDspace.zero()

        for i in range(self.Ntrans):
            for j in range(self.Ntrans):
                prod=[hi*hj for hi, hj in zip(Cont[i],Cont[j])]
                kern = self.KernelClass.gradient(odl.space.rn(self.dim).element([xd - ou for xd, ou in zip(GD[j], GD[i])]))
                grad[j]+=sum(prod)*odl.space.rn(self.dim).element(kern.copy())

        return grad



    def CostGradCont(self,GD,Cont):
        grad=self.Contspace.zero()

        for i in range(self.Ntrans):
            for j in range(self.Ntrans):
                grad[j]+=self.Kernel(odl.space.rn(self.dim).element(GD[i]-GD[j]))*2*Cont[i].copy()

        return grad
    """
