#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:26:02 2017

@author: barbara
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
import copy
__all__ = ('FromFileV3', )



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






def fun_alpha(b,unit_vect,space):
    # returns an image that associates to each point x the value <x-b , unit_vect>
    points=space.points()
    points_dec=(points-np.array(b)).copy()
    I=space.element(sum(points_dec.T[i]*unit_vect[i] for i in range(len(unit_vect))))

    return I

def fun_beta(b,unit_vect,space):
    # returns an image that associates to each point x the value <x-b , unit_vect_orth>
    points=space.points()
    unit_vect_orth=[-unit_vect[1], unit_vect[0]]

    points_dec=(points-np.array(b)).copy()
    I=space.element(sum(points_dec.T[i]*unit_vect_orth[i] for i in range(len(unit_vect_orth))))

    return I


def fun_alpha_diff(b,unit_vect,space,d_b, d_unit_vect):

    # differential of fun_alpha
    I=space.zero()

    prod=sum([d_b[u]*unit_vect[u] for u in range(2)])
    I=(I- prod).copy()
    I+=fun_alpha(b,d_unit_vect,space).copy()

    return I.copy()



def fun_beta_diff(b,unit_vect,space,d_b, d_unit_vect):

    # differential of fun_beta
    I=space.zero()
    unit_vect_orth=[-unit_vect[1], unit_vect[0]]
    prod=sum([d_b[u]*unit_vect_orth[u] for u in range(2)])
    I=(I- prod).copy()
    I+=fun_beta(b,d_unit_vect,space).copy()

    return I.copy()


def ComputeCenterUnitvect(o):
    c0=o[0]
    theta0=o[1][0] # because o[1] is an element of rn(1) and then a list
    a=o[2]
    b=o[3]
    if a==b:
        raise TypeError(' a and b are not different'
                    '')
    diff_ab=[(b[u]-a[u]) for u in range(len(c0))].copy()
    centre=[c0[u] + 0.5*(b[u]+a[u]) for u in range(len(c0))].copy()

    norm_ab=np.sqrt(np.sum([diff_ab[u]**2 for u in range(len(c0))], dtype=np.float64))
    vect_unit_tmp=[diff_ab[u]/norm_ab for u in range(len(c0))].copy()
    # the unit vector is rotated with angle theta0
    vect_unit=[vect_unit_tmp[0]*np.cos(theta0) - vect_unit_tmp[1]*np.sin(theta0) ,
               vect_unit_tmp[0]*np.sin(theta0) + vect_unit_tmp[1]*np.cos(theta0)]

    vect_uni_orth=[-vect_unit[1],vect_unit[0] ].copy()

    return copy.deepcopy([centre,vect_unit,vect_uni_orth])

def fun_Rot(theta,x):
    #rotation of x with angle theta
    return [x[0]*np.cos(theta) - x[1]*np.sin(theta) , x[0]*np.sin(theta) + x[1]*np.cos(theta)].copy()

def ComputeCenterUnitvectdiff(o,d_o):
    c0=o[0]
    theta0=o[1][0]
    a=o[2]
    b=o[3]
    if a==b:
        raise TypeError(' a and b are not different'
                    '')

    d_c0=d_o[0]
    d_theta0=d_o[1][0]
    d_a=d_o[2]
    d_b=d_o[3]

    diff_ab=[(b[u]-a[u]) for u in range(len(c0))].copy()

    norm_ab=np.sqrt(sum([diff_ab[u]**2 for u in range(len(c0))]))
    #unit vector colin to b-a
    vect_unit_tmp=[diff_ab[u]/norm_ab for u in range(len(c0))].copy()


    #differential of elements defined previously
    d_diff_ab=[(d_b[u]-d_a[u]) for u in range(len(c0))].copy()
    d_centre=[d_c0[u] + 0.5*(d_b[u]+d_a[u]) for u in range(len(c0))].copy()
    #scalar product between diff_ab and d_diff_ab
    prod=sum([diff_ab[u]*d_diff_ab[u] for u in range(len(diff_ab))])
    d_vect_unit_tmp=[(d_diff_ab[u]/norm_ab) - (prod*diff_ab[u]/(norm_ab**3)) for u in  range(len(diff_ab))]

    d_vect_unit_tmp_rot=fun_Rot(theta0,d_vect_unit_tmp)

    theta0_der=theta0 + 0.5*np.pi
    vect_unit_tmp_rot_der_tmp=fun_Rot(theta0_der,vect_unit_tmp).copy()
    vect_unit_tmp_rot_der=[vect_unit_tmp_rot_der_tmp[u]*d_theta0 for u in range(2)]

    d_vect_unit=[d_vect_unit_tmp_rot[u] + vect_unit_tmp_rot_der[u] for u in  range(2)]
    d_vect_unit_orth=[-d_vect_unit[1] , d_vect_unit[0] ]


    return copy.deepcopy([d_centre,d_vect_unit,d_vect_unit_orth])



class FromFileV5(DeformationModule):
    """
    This creates a deformation module based on a given one v defined in a file.
    The geometrical descriptor is o=(c,theta,a,b), the control is scalar so that
    the generated vector field is zeta_o (h) = h T_{c + frac{a+b}{2} , theta + theta_{a,b}}
    and xi_o (w) = (0,0, x(a),w(b))
    Intuitively (c,theta) defines the affine deformation between (0,e_1) defining v and
    the good direction to be transported at initial time
    """

    def __init__(self,DomainField, Name, Kernel,update):
        """Initialize a new instance.
        DomainField : space on wich vector fields will be defined
        Name : link to a saved generated vector field from which we build everything
        Kernel : kernel, class that has at least methods Eval and derivative
        """
        # update[0]=0/1 whether (a,b) are transported
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




        # Each affine deformation is defined by o=(c,theta,a,b), where a,b,c are points,
        # \theta is an angle, and a scalar control h
        GDspace=odl.ProductSpace(odl.space.rn(self.dim), odl.space.rn(1),odl.space.rn(self.dim),odl.space.rn(self.dim))
        Contspace=odl.space.rn(1)

        basis=[]
        for d in range(self.dim):
            a=GDspace.zero()
            a[0][d]=1
            basis.append(a.copy())
        a=GDspace.zero()
        a[1]=1.0
        basis.append(a.copy())
        for k in range(2):
            for d in range(self.dim):
                a=GDspace.zero()
                a[2+k][d]=1
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
                h = self.Contspace.element(h).copy()
            except (TypeError, ValueError) as err:
                raise TypeError(' h is not in `Contspace` instance'
                            '')

        points=self.space.points()
        CentreVect=ComputeCenterUnitvect(o)
        centre=CentreVect[0]
        unit_vect=CentreVect[1]
        unit_vect_orth=CentreVect[2]
        alph=fun_alpha(centre,unit_vect,self.space)
        bet=fun_beta(centre,unit_vect,self.space)

        points_ref=np.empty_like(points)
        points_ref.T[0]=np.reshape(np.asarray(alph),points.T[0].shape)
        points_ref.T[1]=np.reshape(np.asarray(bet),points.T[1].shape)


        # v_interp is made of the interpolation of functions of v on
        # the reference points points_depl_origin
        v_interp=[]

        for i in range(2):
            v_interp.append(
                    self.space.element(self.vect_field_padded[i].interpolation(points_ref.T)))


        vect_field_u=self.space.tangent_bundle.zero()
        vect_field_v=self.space.tangent_bundle.zero()
        vect_field_u[0]+=unit_vect[0]
        vect_field_u[1]+=unit_vect[1]
        vect_field_v[0]+=unit_vect_orth[0]
        vect_field_v[1]+=unit_vect_orth[1]
        vect_field_u*=v_interp[0].copy()
        vect_field_v*=v_interp[1].copy()


        vect_field=h[0]*(vect_field_u.copy() + vect_field_v.copy())


        return vect_field.copy()


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
                self.Cont=ope.Contspace.element(Cont).copy()[0]

                points=ope.space.points()
                [centre,unit_vect,unit_vect_orth]=ComputeCenterUnitvect(self.GD)
                alph=fun_alpha(centre,unit_vect,ope.space)
                bet=fun_beta(centre,unit_vect,ope.space)
                self.centre=centre.copy()
                self.unit_vect=unit_vect.copy()
                self.unit_vect_orth=unit_vect_orth.copy()
                self.alpha=alph.copy()
                self.beta=bet.copy()

                points_ref=np.empty_like(points)
                points_ref.T[0]=np.reshape(np.asarray(alph),points.T[0].shape)
                points_ref.T[1]=np.reshape(np.asarray(bet),points.T[1].shape)


                # v_interp is made of the interpolation of functions of v on
                # the reference points points_depl_origin
                v_interp=[]

                for i in range(2):
                    v_interp.append(
                            ope.space.element(ope.vect_field_padded[i].interpolation(points_ref.T)))

                self.v_interp=copy.deepcopy(v_interp)
                # v_der_interp is made of the interpolation of derivatives of functions of v on
                # the reference points points_depl_origin
                # (v[i][j] is the differential of the i-th component of v wrt the j-th component)

                v_der_interp=[]
                for i in range(2):
                    tmp=[]
                    for j in range(2):
                        tmp.append(
                            ope.space.element(ope.vect_field_der_padded[i][j].interpolation(points_ref.T)).copy())
                    v_der_interp.append(copy.deepcopy(tmp))

                self.v_der_interp=copy.deepcopy(v_der_interp)


                super().__init__(ope.GDspace, ope.DomainField.tangent_bundle,
                                 linear=True)


            def _call(self,dGD):
                dGD=ope.GDspace.element(dGD).copy()

                d_var = ComputeCenterUnitvectdiff(self.GD,dGD)
                d_centre=d_var[0]
                d_unit_vect=d_var[1]
                d_unit_vect_orth=d_var[2]

                vector_field=ope.space.tangent_bundle.zero()
                points=ope.space.points()
                # Differential of reference points
                d_alpha=fun_alpha_diff(self.centre, self.unit_vect, ope.space, d_centre,d_unit_vect)
                d_beta=fun_beta_diff(self.centre, self.unit_vect, ope.space, d_centre,d_unit_vect)


                # list of 2 images corresponding to the differentials of v_i(pts_ref)
                d_v_ptsref =[]
                for i in range(2):
                    im_temp=(self.v_der_interp[i][0]*d_alpha + self.v_der_interp[i][1]*d_beta).copy()
                    d_v_ptsref.append(im_temp.copy())

                vector_field=ope.space.tangent_bundle.zero()

                # we first compute the part of the derivative corresponding
                # to the derivative of reference points
                vector_field_temp=ope.space.tangent_bundle.zero()
                for i in range(2):
                    vector_field_temp[i]+=self.unit_vect[i]
                vector_field_temp*=d_v_ptsref[0].copy()
                vector_field+=self.Cont*ope.space.tangent_bundle.element(vector_field_temp).copy()

                vector_field_temp=ope.space.tangent_bundle.zero()
                for i in range(2):
                    vector_field_temp[i]+=self.unit_vect_orth[i]
                vector_field_temp*=d_v_ptsref[1].copy()
                vector_field+=self.Cont*vector_field_temp.copy()

                # we then compute the part of the derivative corresponding
                # to the derivative of unit vectors
                vector_field_temp=ope.space.tangent_bundle.zero()
                for i in range(2):
                    vector_field_temp[i]+=d_unit_vect[i]
                vector_field_temp*=self.v_interp[0].copy()
                vector_field+=self.Cont*vector_field_temp.copy()

                vector_field_temp=ope.space.tangent_bundle.zero()
                for i in range(2):
                    vector_field_temp[i]+=d_unit_vect_orth[i]
                vector_field_temp*=self.v_interp[1].copy()
                vector_field+=self.Cont*vector_field_temp.copy()


                return vector_field.copy()

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
            GD=self.GDspace.element(GD).copy()
            c0=GD[0]
            theta0=GD[1][0]
            a=GD[2]
            b=GD[3]

            speed=self.GDspace.zero()
            if(self.update[0]==1):
                speed[2]=[vect_field[i].interpolation(np.array(a)) for i in range(self.dim)].copy()
                speed[3]=[vect_field[i].interpolation(np.array(b)) for i in range(self.dim)].copy()


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

