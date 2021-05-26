#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:12:49 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:23:51 2017

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
__all__ = ('DeformationModule', )



class DeformationModule(object):
    """Abstract deformation module


    **Abstract attributes and methods**

    `DeformationModuleAbstract` is an **abstract** class, i.e. it can only be
    subclassed, not used directly.

    Any subclass of `DeformationModuleAbstract` must have the following
    attributes:
        ''GDspace'' : space of geometrical descriptor
                      it must be a vector space
        ''Contspace'' : space of controls
                      it must be a vector space
        ''basisGD'' : orthonormal basis of GDspace
                      list of elements of GDspace
        ''basisCont'' : orthonormal basis of Contspace
                      list of elements of Contspace
        ''basisContTraj'' : orthonormal basis of L^2([0,1],Contspace)
                      discretized (it is a list of lists of list of elements
                      of Contspace)
        ''DomainField'' : set
                        space on wich vector fields will be defined


    and the following methods :
        ''ComputeField'' : Computes the generated vector field
                            operator with domain = GDspace x Contspace
                            and range =DomainField.tangent_bundle
                            (space of vector fields)
        ''ComputeFieldEvaluate'' : class defining the generated vector field
                            Initialised with an element of GDspace x Contspace
                            domaine = R^d (d is the dimension of DomainField)
        ''ComputeFieldDer'' : Computes the derivative with respect to the
                             first component of the generated vector field
                             class initialized  with an element of
                             GDspace x Contspace
                             then domain = GDspace
                            and range = space of vector fields
        ''ComputeFieldDerEvaluate'' : class defining the generated vector field
                            Initialized with an element of GDspace x Contspace
                            domaine = GDspace x R^d
                            (d is the dimension of DomainField)
                            range = R^d
        ''ApplyVectorField'' : operator
                            domaine = GDspace x DomainField.tangent_bundle
                            range = GDspace
        ''ApplyModule'' : class initialized with a module and values for
                            its GD and controls
                            then domaine = GDspace and range = GDspace
        ''Cost'' : functional (non linear)
                    domain= GDspace x Contspace
        CostDerGD : class initialized by a value in GDspace x Contspace
                    computes the derivative of the cost with respect to the
                    first component
                    domain = GDspace
        CostDerCont : class initialized by a value in GDspace x Contspace
                    computes the derivative of the cost with respect to the
                    second component
                    domain = Contspace


    """


    def __init__(self,GDspace,Contspace,basisGD,basisCont,DomainField):
        """Initialize a new instance.
        """
        #print(0)

        self.GDspace=GDspace
        self.Contspace=Contspace
        self.domain=DomainField
        #print(1)
        for i in range(len(basisGD)):
             #print(i)
             if basisGD[i] not in self.GDspace:
                 try:
                     basisGD[i] = self.domain.element(basisGD[i]).copy()
                 except (TypeError, ValueError) as err:
                     raise TypeError(' {!r} th element of `basisGD` is not in `GDspace` instance'
                            ''.format(i))


        #print(2)
        self.basisGD=basisGD

        for i in range(len(basisCont)):
             if basisCont[i] not in self.Contspace:
                 try:
                     basisCont[i] = self.domain.element(basisCont[i]).copy()
                 except (TypeError, ValueError) as err:
                     raise TypeError(' {!r} th element of `basisCont` is not in `Contspace` instance'
                            ''.format(i))


        self.basisCont=basisCont


        self.DomainField=DomainField


    def ComputeField(self, o,h):
        """Return the computed vector field on DomainField
        """

        raise odl.OpNotImplementedError('ComputeField not implemented '
                                        'for operator {!r}'
                                        ''.format(self))

    @property
    def ComputeFieldEvaluate(self):
        """Return the computed vector field applied to
           a given point
        """
        raise odl.OpNotImplementedError('ComputeFieldEvaluate not implemented '
                                    'for operator {!r}'
                                    ''.format(self))
    @property
    def ComputeFieldDer(self):
        """Return the the derivative with respect to the
           first component of computed vector field on DomainField
        """

        raise odl.OpNotImplementedError('ComputeFieldDer not implemented '
                                        'for operator {!r}'
                                        ''.format(self))

    @property
    def ComputeFieldDerEvaluate(self):
        """Return the the derivative with respect to the
           first component of computed vector field applied to
           a given point
        """
        raise odl.OpNotImplementedError('ComputeFieldDerEvaluate not implemented '
                                    'for operator {!r}'
                                    ''.format(self))


    def ApplyVectorField(self, o,v):
        """Return the the application of v on o
        """

        raise odl.OpNotImplementedError('ApplyVectorField not implemented '
                                        'for operator {!r}'
                                        ''.format(self))


    @property
    def ApplyModule(self):
        """Return the the application of the vector field generated by
          a given module initializing the class on o
        """

        raise odl.OpNotImplementedError('ApplyModule not implemented '
                                    'for operator {!r}'
                                    ''.format(self))


    def Cost(self, o,h):
        """Return the cost of (o,h)
        """

        raise odl.OpNotImplementedError('Cost not implemented '
                                        'for operator {!r}'
                                        ''.format(self))


    @property
    def CostDerGD(self):
        """computes the derivative of the cost with respect to the
                    first component, at given (o,h)
        """

        raise odl.OpNotImplementedError('CostDerGD not implemented '
                                    'for operator {!r}'
                                    ''.format(self))

    @property
    def CostDerCont(self):
        """computes the derivative of the cost with respect to the
                    second component, at given (o,h)
        """

        raise odl.OpNotImplementedError('CostDerCont not implemented '
                                    'for operator {!r}'
                                    ''.format(self))



    def CostGradGD(self,GD,Cont):

        raise odl.OpNotImplementedError('CostGradGD not implemented '
                                    'for operator {!r}'
                                    ''.format(self))




    def CostGradCont(self,GD,Cont):

        raise odl.OpNotImplementedError('CostgradCont not implemented '
                                    'for operator {!r}'
                                    ''.format(self))





class Compound(DeformationModule):
    def __init__(self,ModulesList):
        self.ModulesList=ModulesList
        domain=ModulesList[0].DomainField
        self.Nmod=len(ModulesList)
        for i in range(1,self.Nmod):
          if not (ModulesList[i].DomainField==domain):
              print('Problem domains')

        GDspace=odl.ProductSpace(*[ModulesList[i].GDspace for i in range(self.Nmod)])
        Contspace=odl.ProductSpace(*[ModulesList[i].Contspace for i in range(self.Nmod)])


        self.dim=ModulesList[0].DomainField.ndim
        basisGD=[]
        contGD=0
        basisCont=[]
        contCont=0
        for i in range(self.Nmod):
            for k in range(len(ModulesList[i].basisGD)):
                basisGD.append(GDspace.zero())
                basisGD[contGD][i]=ModulesList[i].basisGD[k].copy()
                contGD+=1

            for k in range(len(ModulesList[i].basisCont)):
                basisCont.append(Contspace.zero())
                basisCont[contCont][i]=ModulesList[i].basisCont[k].copy()
                contCont+=1


        super().__init__(GDspace,Contspace,basisGD,basisCont,domain)

    def ComputeField(self, o,h):
        vector_field=self.DomainField.tangent_bundle.zero()
        for i in range(self.Nmod):
            vector_field+=self.ModulesList[i].ComputeField(o[i],h[i]).copy()

        return vector_field


    @property
    def ComputeFieldEvaluate(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                EvalList=[]
                for i in range(ope.Nmod):
                    EvalList.append(ope.ModulesList[i].ComputeFieldEvaluate(GD[i],Cont[i]))

                self.EvalList=EvalList
                super().__init__(odl.space.rn(ope.dim), odl.space.rn(ope.dim),
                                 linear=False)

            def _call(self,x):
                speed=odl.space.rn(ope.dim).zero()
                for i in range(ope.Nmod):
                    speed+=self.EvalList[i](x).copy()
                return speed
        return Eval



    @property
    def ComputeFieldDer(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                CompList=[]
                for i in range(ope.Nmod):
                    CompList.append(ope.ModulesList[i].ComputeFieldDer(GD[i],Cont[i]))

                self.CompList=CompList
                super().__init__(ope.GDspace, ope.DomainField.tangent_bundle,
                                 linear=True)


            def _call(self,dGD):
                dGD=ope.GDspace.element(dGD).copy()
                vector_field=ope.DomainField.tangent_bundle.zero()
                for i in range(ope.Nmod):
                    vector_field+=self.CompList[i](dGD[i]).copy()
                return vector_field

        return Eval



    @property
    def ComputeFieldDerEvaluate(self):
        ope=self
        class Eval(Operator):
            def __init__(self,GD,Cont):
                self.GD=ope.GDspace.element(GD).copy()
                self.Cont=ope.Contspace.element(Cont).copy()
                EvalList=[]
                for i in range(ope.Nmod):
                    EvalList.append(ope.ModulesList[i].ComputeFieldDerEvaluate(GD[i],Cont[i]))
                self.EvalList=EvalList
                super().__init__(odl.ProductSpace(ope.GDspace,odl.space.rn(ope.dim)), odl.space.rn(ope.dim),
                                 linear=True)


            def _call(self,X):
                dGD=ope.GDspace.element(X[0]).copy()
                x=odl.space.rn(ope.dim).element(X[1])
                speed=odl.space.rn(ope.dim).zero()
                for i in range(ope.Nmod):
                    speed+=self.EvalList[i]([dGD[i],x]).copy()
                return speed

        return Eval



    def ApplyVectorField(self,GD,vect_field):
            #GD=self.GDspace.element(X[0])
            #vect_field=self.DomainField.tangent_bundle.element(X[1])
            speed=self.GDspace.element()
            for i in range(self.Nmod):
                speed[i]=self.ModulesList[i].ApplyVectorField(GD[i],vect_field)

            return speed




    def ApplyVectorField(self,GD,vect_field):
            #GD=self.GDspace.element(X[0])
            #vect_field=self.DomainField.tangent_bundle.element(X[1])
            speed=self.GDspace.element()
            for i in range(self.Nmod):
                speed[i]=self.ModulesList[i].ApplyVectorField(GD[i],vect_field)

            return speed




    """def ApplyModule(self,GD,Module):
        ope = self
        class apply(Operator):
            def __init__(self,Module,GDmod,Contmod):
                super().__init__(ope.GDspace, ope.GDspace,
                                 linear=False)

            def _call(self,GD):
                speed=ope.GDspace.element()
                for i in range(self.Nmod):
                    speed[i]=self.ModulesList[i].(GD[i])
                    self.apply_op(GD[i])

                for i in range(len(GD)):
                    speed[i]=self.apply_op(GD[i])
                return speed
        return apply
        """


    def Cost(self,GD,Cont):
        energy=0
        for i in range(self.Nmod):
            energy+=self.ModulesList[i].Cost(GD[i],Cont[i])

        return energy






    def CostGradGD(self,GD,Cont):
        grad=self.GDspace.zero()
        for i in range(self.Nmod):
            grad[i]=self.ModulesList[i].CostGradGD(GD[i],Cont[i])

        return grad



    def CostGradCont(self,GD,Cont):
        grad=self.Contspace.zero()
        for i in range(self.Nmod):
            grad[i]=self.ModulesList[i].CostGradCont(GD[i],Cont[i])

        return grad










