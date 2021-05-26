#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 12:22:26 2017

@author: barbara
"""

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


class FromFile(DeformationModule):
    def __init__(self,DomainField, Name, Kernel,update):
        """Initialize a new instance.
        DomainField : space on wich vector fields will be defined
        Name : link to a saved generated vector field from which we build everything
        Kernel : kernel, class that has at least methods Eval and derivative
        """
        # update[0]=0/1 whether the centre is transported and
        # update[1]=0/1 hether the angme is transported and
        self.update=update
        self.KernelClass=Kernel
        self.space=DomainField
        #self.Kernel=Kernel.Eval
        def kernelOpFun(x):
            return Kernel.Eval(x)
        self.Kernel=kernelOpFun
        self.dim=DomainField.ndim
        vect_field=DomainField.tangent_bundle.element(np.loadtxt(Name)).copy()
        self.vect_field=vect_field.copy()

        # Padded vector fields for interpolation
        padded_size = 2 * DomainField.shape[0]
        padded_op = ResizingOperator(
            DomainField, ran_shp=[padded_size for _ in range(DomainField.ndim)])
        padded_space=padded_op.range
        self.padded_space=padded_space
        self.vect_field_padded=padded_space.tangent_bundle.element([padded_op(vect_field[u]) for u in range(self.dim)])
        
        
        #Derivative with respect to the geometrical descriptor : 
        # corresponds to (v(x- delta) - v(x)) / delta
        vect_field_der=[]
        vect_field_der_padded=[]
        for i in range(DomainField.ndim):
            delta=2*DomainField.cell_sides[i]
            dec_temp=DomainField.tangent_bundle.zero()
            dec_temp[i]=(delta)*DomainField.one()
            temp=DomainField.tangent_bundle.element(
                    [_linear_deform(vect_field[d],-dec_temp) for d in range(DomainField.ndim)]).copy()
            vect_field_der.append((1/delta)*(temp - vect_field).copy())
            vect_field_der_padded.append(padded_space.tangent_bundle.element([padded_op(vect_field_der[i][u]) for u in range(self.dim)]))

        self.vect_field_der=vect_field_der.copy()
        self.vect_field_der_padded=vect_field_der_padded.copy()
        
        


        # Each affine deformation is defined by a centre c (a point) and an angle
        # \\theta (a scalar) so GD=(c, \theta) and a scalar control
        # dim+1 vectors (the controls)
        GDspace=odl.ProductSpace(odl.space.rn(self.dim), odl.space.rn(1))
        Contspace=odl.space.rn(1)

        basis=[]
        for d in range(self.dim):
            a=GDspace.zero()
            a[0][d]=1
            basis.append(a.copy())
        a=GDspace.zero()
        a[1]=1
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
        centre=o[0]
        angle=o[1][0]
        points_temp=self.space.points().copy() - np.array(centre.copy() )
        
        # points rotated with angle - theta[i] 
        points_temp = Rtheta(-angle, points_temp).copy()    
        # Interpolation of the vector field on points_temp
        vect_field_temp=np.array(
        [self.vect_field_padded[k].interpolation(points_temp.T) for k in range(self.dim)]).T.copy()
        
        # Rotation of the vector field with angle theta[i]
        vect_field_temp=self.space.tangent_bundle.element(Rtheta(angle,vect_field_temp).T)
        
        # Shift of the vector fieldwith translation of vector center_list[i]
        #vect_field_temp=self.space.tangent_bundle.element(
        #[vect_field_temp[k] + centre[k] for k in range(self.dim)]).copy()
        
        vect_field_temp*=(h[0])

        return vect_field_temp.copy()
    
    
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
                self.centre=self.GD[0]
                self.angle=self.GD[1][0]
                self.Cont=ope.Contspace.element(Cont).copy()
                points_temp=ope.space.points().copy() - np.array(self.centre.copy() ) 
                self.points_temp = Rtheta(-self.angle, points_temp).copy()  
                self.points_temp_prime = Rprimetheta(-self.angle, points_temp).copy()  
                super().__init__(ope.GDspace, ope.DomainField.tangent_bundle,
                                 linear=True)


            def _call(self,dGD):
                dGD=ope.GDspace.element(dGD).copy()
                # the control self.Cont is of dimension 1
                
                dcentre=dGD[0]
                dangle=dGD[1][0]
                dcentre_rot=Rtheta(-self.angle, np.array(dcentre))
                vector_field=ope.space.tangent_bundle.zero()
                tmp0=ope.space.tangent_bundle.zero()
                for i in range(ope.dim):
                    tmp=ope.space.tangent_bundle.element(
                            [ope.vect_field_der_padded[i][u].interpolation(self.points_temp.T) for u in range(ope.dim)])
                    vector_field+=self.Cont[0]*dcentre_rot[i]*tmp.copy()
                    tmp0+=(tmp*ope.space.element(self.points_temp_prime.T[i]))
                    
                vector_field-=self.Cont[0]*dangle*ope.space.tangent_bundle.element(Rtheta(self.angle,np.array(tmp0).T).T)
                    
                    
                #vector_field=-self.Cont[0]*dangle*ope.space.tangent_bundle.element(
                #       Rprimetheta(-self.angle,np.array(vector_field.copy()).T).T).copy()
                
                tmp1=np.array([ope.vect_field_padded[i].interpolation(self.points_temp.T) for i in range(ope.dim)]).T.copy()
                vector_field+=self.Cont[0]*dangle*ope.space.tangent_bundle.element(
                        Rprimetheta(self.angle,tmp1).T)
                
                         

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
            centre=GD[0]
            #angle=GD[1]
            
            speed=self.GDspace.zero()
            if(self.update[0]==1):
                speed[0]=[vect_field[i].interpolation(np.array(centre)) for i in range(self.dim)]
            
            if(self.update[1]==1):
                # For the angle we compute the numerical rotational
                step=self.space.cell_sides
                
                centre_dec0=centre.copy()
                centre_dec0[0]+=step[0]
                dtheta=(vect_field[1].interpolation(centre) - vect_field[1].interpolation(centre_dec0))/step[0]
                
                centre_dec1=centre.copy()
                centre_dec1[1]+=step[1]
                dtheta-=(vect_field[0].interpolation(centre) - vect_field[0].interpolation(centre_dec1))/step[1]
                
                speed[1]=dtheta
                
           
            
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
    
