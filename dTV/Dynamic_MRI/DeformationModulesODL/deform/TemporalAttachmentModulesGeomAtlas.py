#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:59:34 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:20:53 2017

@author: bgris
"""

"""Operators and functions for 4D image registration and template estimation
via deformation module."""

# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

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

from odl.solvers.functional.functional import Functional
__all__ = ('FunctionalModulesGeomAtlas',
           'ShootTemplateFromVectorFields' )


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

def ShootTemplateFromVectorFields(vector_field_list, template):
    N=vector_field_list.size-1
    series_image_space_integration = ProductSpace(template.space,
                                                  N+1)
    inv_N=1/N
    I=series_image_space_integration.element()
    I[0]=template.copy()
    for i in range(N):
        I[i+1]=template.space.element(
                _linear_deform(I[i],
                               -inv_N * vector_field_list[i+1])).copy()
    return I

class FunctionalModulesGeomAtlas(Functional):

    """Deformation operator with fixed template acting on displacement fields.



        Compute the attachment term of the deformed template, copared to the
        given data at each time point.


    """

    def __init__(self,lam, nb_time_point_int,  data, data_time_points, forward_operators,Norm, Module, domain=None):
        """
        Functional to compute
        lam*Reg +\sum_j Norm(data_j -forward_operators_j (phi_tj (template)) )
        with phi and Reg of the module

        Parameters
        ----------
        nb_time_point_int : int
           number of time points for the numerical integration
        forward_operators : list of `Operator`
            list of the forward operators, one per data time point.
        Norm : functional
            Norm in the data space (ex: l2 norm)
        data_elem : list of 'DiscreteLpElement'
            Given data.
        data_time_points : list of floats
            time of the data (between 0 and 1)
        Module : 'DeformationModule'
            deformation module thanks to which we want to
            generate deformations.


        Note : here the template is not fixed
        """
        self.lam=lam
        self.data=data
        self.Norm=Norm
        self.data_time_points= np.array(data_time_points)
        self.forward_op=forward_operators
        self.Module=Module
        self.nb_data=self.data_time_points.size
        self.image_domain=forward_operators[0].domain
        self.module_domain=self.Module.DomainField
        # Give the number of time intervals
        self.N = nb_time_point_int

        # Give the inverse of time intervals for the integration
        self.inv_N = 1.0 / self.N # the indexes will go from 0 (t=0) to N (t=1)

        # list of indexes k_j such that
        # k_j /N <= data_time_points[j] < (k_j +1) /N
        self.k_j_list=np.arange(self.nb_data)
        for j in range(self.nb_data):
            self.k_j_list[j]= int(self.N*data_time_points[j])


        # sorted list of all the times
        self.alltimes=np.empty(self.nb_data + self.N +1)
        # list with 0 if time of integration and 1 if data time point
        self.naturetime=np.arange(self.nb_data + self.N +1)
        j0=0
        cont=0
        for k in range(self.N+1):
            self.alltimes[cont]=k/self.N
            self.naturetime[cont]=0
            cont+=1
            for j in range(j0,self.nb_data):
                if (data_time_points[j]<(k+1)/self.N):
                    self.alltimes[cont]=data_time_points[j]
                    self.naturetime[cont]=1
                    cont+=1
                    j0+=1

        # list of indexes j_k such that
        #  data_time_points[j_k] < k/N <= data_time_points[j_k +1]
        self.j_k_list=np.arange(self.N+1)
        for k in range(self.N+1):
            for j in range(self.nb_data):
                if data_time_points[self.k_j_list.size -1-j]>=k/self.N:
                    self.j_k_list[k]= int(self.k_j_list.size -1-j)


        # Definition S[j] = Norm(forward_operators[j] - data[j])
        S=[]
        for j in range(self.nb_data):
            S.append(self.Norm*(self.forward_op[j] - self.data[j]))
        self.S=S

        self.basisCont=self.Module.basisCont
        self.basisGD=self.Module.basisGD
        # the domain is the set of initial GD and temporal list of controls
        if domain is None:
            domain = odl.ProductSpace(
                    self.Module.GDspace,odl.ProductSpace(
                    self.Module.Contspace,self.N+1),self.forward_op[0].domain)


        super().__init__(domain,linear=False)



    def _call(self, X, out=None):
        template=X[2].copy()
        GD=X[0].copy() # initial GD
        Cont=X[1].copy() # temporal list of controls

        reg=0
        attachment=0
        contk=0
        contj=0

        I_t=template.copy()
        Cont_t=Cont[0].copy()
        # cost at time t=0
        reg+= self.Module.Cost(GD,Cont_t)

        for i in range(1,self.nb_data + self.N+1):
            # iteration i starts with value at time self.alltimes[i-1]
            # and ends with value at time self.alltimes[i]
            if self.naturetime[i-1]==0:
                Cont_t=Cont[contk].copy()
            else:
                if(self.k_j_list[contj]==self.N):
                    Cont_t=Cont[self.N]
                else:
                    # 'contj-1' because we need the control at self.alltimes[i-1]
                    delta0=(self.data_time_points[contj-1] -((self.k_j_list[contj-1])/self.N))
                    Cont_t=((1-delta0)*Cont[self.k_j_list[contj-1]] + delta0*Cont[self.k_j_list[contj-1]+1]).copy()


            d_t=self.alltimes[i]-self.alltimes[i-1]
            vect_field=self.Module.ComputeField(GD,Cont_t).copy()

            GD+=d_t*self.Module.ApplyVectorField(GD,vect_field).copy()

            vect_field_image=self.image_domain.tangent_bundle.element([vect_field[uu].interpolation(self.image_domain.points().T) for uu in range(self.image_domain.ndim)])

            I_temp=self.image_domain.element(
                _linear_deform(I_t,
                               -d_t*vect_field_image)).copy()
            I_t=I_temp.copy()
            if self.naturetime[i]==0:
                # Now GD corresponds to contk + 1
                contk+=1
                # For discretized integration only between 0 and N-1
                if(contk<self.N):
                    reg+= self.Module.Cost(GD,Cont[contk])
            else:
                attachment+=self.S[contj](I_t)
                contj+=1


        return (self.lam/self.N)*reg+(1/(self.N+1))*attachment

    def ComputeReg(self, X, out=None):

        GD=X[0].copy() # initial GD
        Cont=X[1].copy() # temporal list of controls

        reg=0
        contk=0
        contj=0

        Cont_t=Cont[0].copy()
        # cost at time t=0
        reg+= self.Module.Cost(GD,Cont_t)

        for i in range(1,self.nb_data + self.N+1):
            # iteration i starts with value at time self.alltimes[i-1]
            # and ends with value at time self.alltimes[i]
            if self.naturetime[i-1]==0:
                Cont_t=Cont[contk].copy()
            else:
                if(self.k_j_list[contj]==self.N):
                    Cont_t=Cont[self.N]
                else:

                    delta0=(self.data_time_points[contj] -((self.k_j_list[contj])/self.N))
                    Cont_t=((1-delta0)*Cont[self.k_j_list[contj]] + delta0*Cont[self.k_j_list[contj]+1]).copy()


            d_t=self.alltimes[i]-self.alltimes[i-1]
            vect_field=self.Module.ComputeField(GD,Cont_t).copy()

            GD+=d_t*self.Module.ApplyVectorField(GD,vect_field).copy()

            if self.naturetime[i]==0:
                # Now GD corresponds to contk + 1
                contk+=1
                # For discretized integration only between 0 and N-1
                if(contk<self.N):
                    reg+= self.Module.Cost(GD,Cont[contk])



        return (self.lam/self.N)*reg

    def Shoot(self, X, out=None):
        template=X[2].copy()
        GD=X[0].copy() # initial GD
        Cont=X[1].copy() # temporal list of controls

        series_image_space_integration = ProductSpace(template.space,
                                                  self.N+1)

        I=series_image_space_integration.element()
        I[0]=template.copy()


        #Cont_t=Cont[0].copy()
        # cost at time t=0

        for i in range(self.N):

            Cont_t=Cont[i].copy()

            d_t=1/self.N
            vect_field=self.Module.ComputeField(GD,Cont_t).copy()

            GD+=d_t*self.Module.ApplyVectorField(GD,vect_field).copy()

            vect_field_image=self.image_domain.tangent_bundle.element([vect_field[uu].interpolation(self.image_domain.points().T) for uu in range(self.image_domain.ndim)])

            I[i+1]=self.image_domain.element(
                _linear_deform(I[i],
                               -d_t*vect_field_image)).copy()

        return I

    def ComputetrajectoryGD(self,X):

        GD=X[0].copy() # initial GD
        Cont_t=X[1].copy() # temporal list of controls

        GD_t=odl.ProductSpace(
                          self.Module.GDspace,self.N+1).element()
        GD_t[0]=GD.copy()
        for i in range(self.N):
            vect_field=self.Module.ComputeField(GD_t[i],Cont_t[i]).copy()
            GD_t[i+1]=(GD_t[i] + self.inv_N*self.Module.ApplyVectorField(GD_t[i],vect_field)).copy()
            #GD_t[i+1]=(GD_t[i] + self.inv_N*self.Module.ApplyModule(self.Module,GD_t[i],Cont_t[i])(GD_t[i])).copy()

        return GD_t




#    @property
#    def gradient(self):
#
#        functional = self
#
#        class FunctionalModulesGeomGradientAtlas(Operator):
#
#            """The gradient operator of the FunctionalModulesGeom
#            functional."""
#
#            def __init__(self):
#                """Initialize a new instance."""
#                super().__init__(functional.domain, functional.domain,
#                                 linear=False)
#
#            def _call(self, X):
#                dim=functional.Module.dim
#                GD=X[0].copy() # initial GD
#                Cont=X[1].copy() # temporal list of controls
#                template=X[2].copy()
#
#                # Forward integration, we save values of GD
#                GD_tk=odl.ProductSpace(
#                          functional.Module.GDspace,functional.N+1).element()
#                GD_tj=odl.ProductSpace(
#                          functional.Module.GDspace,functional.nb_data).element()
#
#                I_t=template.copy()
#                contk=0
#                contj=0
#                GD_tk[0]= GD.copy()
#                series_image_space_data = ProductSpace(functional.image_domain, functional.nb_data)
#                image_data = series_image_space_data.element()
#                for i in range(1,functional.nb_data + functional.N +1):
#
#                    if functional.naturetime[i-1]==0:
#                        Cont_t=Cont[contk].copy()
#                        GD_t=GD_tk[contk].copy()
#                    else:
#                        delta0=(functional.data_time_points[contj] -((functional.k_j_list[contj])/functional.N))
#                        Cont_t=((1-delta0)*Cont[functional.k_j_list[contj]] + delta0*Cont[functional.k_j_list[contj]+1]).copy()
#                        GD_t=GD_tj[contj].copy()
#
#
#                    d_t=functional.alltimes[i]-functional.alltimes[i-1]
#                    vect_field=functional.Module.ComputeField(GD_t,Cont_t).copy()
#                    vect_field_image=functional.image_domain.tangent_bundle.element([vect_field[uu].interpolation(functional.image_domain.points().T) for uu in range(functional.image_domain.ndim)])
#
#                    if functional.naturetime[i]==0:
#                        GD_tk[contk]=(GD_t+d_t*functional.Module.ApplyVectorField(GD_t,vect_field)).copy()
#                        I_t=functional.image_domain.element(
#                                _linear_deform(I_t,
#                                   -functional.inv_N*vect_field_image)).copy()
#
#                        contk+=1
#                    else:
#                        image_data[contj]=functional.image_domain.element(
#                                _linear_deform(I_t,
#                                   -d_t*vect_field_image)).copy()
#
#                        GD_tj[contj]=(GD_t+d_t*functional.Module.ApplyVectorField(GD_t,vect_field)).copy()
#                        contj+=1
#
#                # Computation of the derivative of gamma (trajectory of GD)
#                nX=len(functional.Module.basisGD)
#                #gamma_derGD=odl.ProductSpace(
#                #        odl.ProductSpace(functional.Module.GDspace,functional.N+1),
#                #        nX).element()
#                gamma_derGD=[]
#                nH=len(functional.Module.basisCont)
#                #gamma_derCont=odl.ProductSpace(
#                  #      odl.ProductSpace(functional.Module.GDspace,functional.N+1),
#                 #       nH).element()
#                gamma_derCont=[]
#                # generating nH elements of L^2([0,1],H) which are, constant
#                # equal to an element of the basis of H
#                #basisContTraj=odl.ProductSpace(
#                #        odl.ProductSpace(functional.Module.Contspace,functional.N+1),
#                #        nH).element()
##                basisContTraj=[]
##
##                for i in range(nH):
##                    temp=[]
##                    for u in range(functional.N+1):
##                        #basisContTraj[i][u]=functional.Module.basisCont[i].copy()
##                        temp.append(functional.Module.basisCont[i].copy())
##                    basisContTraj.append(temp.copy())
#
#
#
#                eps=0.001
#                gamma_init=functional.ComputetrajectoryGD([GD,Cont])
#                for i in range(nX):
#                    GDtemp=(GD+ eps*functional.Module.basisGD[i]).copy()
#                    Xtemp=[GDtemp,Cont.copy()]
#                    gamma_derGD.append((1/eps)*(functional.ComputetrajectoryGD(Xtemp)-gamma_init).copy())
#                    #gamma_derGD[i]=functional.ComputetrajectoryGD(Xtemp)
#
#                for i in range(nH):
#                    Conttemp=Cont.copy()
#                    Cont_basis=functional.Module.basisCont[i].copy()
#                    for u in range(functional.N+1):
#                        Conttemp[u]+=eps*Cont_basis.copy()
#                    Xtemp=[GD.copy(),Conttemp]
#                    gamma_derCont.append((1/eps)*(functional.ComputetrajectoryGD(Xtemp)-gamma_init).copy())
#                    #gamma_derCont[i]=functional.ComputetrajectoryGD(Xtemp)
#
#                # define the gradients
#                grad1=odl.ProductSpace(functional.Module.Contspace,functional.N+1).zero()
#                grad2=odl.ProductSpace(functional.Module.Contspace,functional.N+1).zero()
#                grad3=functional.Module.GDspace.zero()
#
#
#                # FFT setting for data matching term, 1 means 100% padding
#                #padded_size = 2 * functional.image_domain.shape[0]
#                #padded_ft_fit_op = padded_ft_op(functional.image_domain, padded_size)
#                #vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
#
#                # Create the gradient op
#                grad_op = Gradient(domain=functional.image_domain, method='forward',pad_mode='symmetric')
#                # Create the divergence op
#                div_op = -grad_op.adjoint
##                detDphi_N1 = series_image_space_integration.element()
#
#                for j in range(functional.nb_data):
#
#                    delta0=(functional.data_time_points[j] -((functional.k_j_list[j])/functional.N))
#                    #Contj=((1-delta0)*Cont[self.k_j_list[contj]] + delta0*Cont[self.k_j_list[contj]+1]).copy()
#                    #print(j)
#                    #print(functional.k_j_list)
#                    #vect_field_j=self.Module.ComputeField(GD_tj[j],Contj).copy()
#                    vect_field_kj=functional.Module.ComputeField(GD_tk[functional.k_j_list[j]],Cont[functional.k_j_list[j]]).copy()
#
#                    grad_S=functional.image_domain.element()
#                    detDphi=functional.image_domain.element()
#                    # initialization at time t_j
#                    #detDphi_t_j=image_domain.one() # initialization not necessary here
#                    grad_S_tj=functional.S[j].gradient(image_data[j]).copy()
#                    # computation at time tau_k_j
#                    delta_t= functional.data_time_points[j]-(functional.k_j_list[j]*functional.inv_N)
#                    detDphi=functional.image_domain.element(
#                                      np.exp(delta_t *
#                                      div_op(vect_field_kj))).copy()
#                    grad_S=functional.image_domain.element(
#                                   _linear_deform(grad_S_tj,
#                                   delta_t * vect_field_kj)).copy()
#
#                    I_t=functional.image_domain.element(
#                            _linear_deform(image_data[j],
#                                           delta_t*vect_field_kj))
#
#
#                    tmp= grad_op(I_t).copy()
#                    tmp1=(grad_S *detDphi).copy()
#                    for d in range(dim):
#                        tmp[d] *= tmp1
#
#                    vect_derGD_op=functional.Module.ComputeFieldDer(GD_tk[functional.k_j_list[j]],Cont[functional.k_j_list[j]])
#
#                    for i in range(nH):
#                        dGD=gamma_derCont[i][functional.k_j_list[j]].copy()
#                        vect_derGD=vect_derGD_op(dGD).copy()
#                        #tmp3= (2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(
#                        #       vectorial_ft_fit_op(tmp) * vectorial_ft_fit_op(vect_derGD)).copy()
#                        #grad1[i][functional.k_j_list[j]]-=(1/(functional.N+1))*tmp3.copy()
#                        tmp3=tmp.inner(vect_derGD)
#                        grad1[functional.k_j_list[j]]-=((1/(functional.N+1))*tmp3)*functional.basisCont[i].copy()
#
#                        vect_i=functional.Module.ComputeField(GD_tk[functional.k_j_list[j]],functional.Module.basisCont[i]).copy()
#                        #tmp3= (2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(
#                        #        vectorial_ft_fit_op(tmp) * vectorial_ft_fit_op(vect_i)).copy()
#                        #grad2[i][functional.k_j_list[j]]-=(1/(functional.N+1))*tmp3.copy()
#
#                        tmp3=tmp.inner(vect_i)
#                        grad2[functional.k_j_list[j]]-=((1/(functional.N+1))*tmp3)*functional.basisCont[i].copy()
#
#
#                    for i in range(nX):
#                        dGD=gamma_derGD[i][functional.k_j_list[j]].copy()
#                        vect_derGD=vect_derGD_op(dGD).copy()
#                        #tmp3= (2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(
#                        #        vectorial_ft_fit_op(tmp) * vectorial_ft_fit_op(vect_derGD)).copy()
#                        #grad3[i]-=(1/(functional.N+1))*tmp3.copy()
#                        tmp3=tmp.inner(vect_derGD)
#                        grad3-=((1/(functional.N+1))*tmp3)*functional.basisGD[i].copy()
#
#
#
#
#                    # loop for k < k_j
#                    delta_t= functional.inv_N
#                    for u in range(functional.k_j_list[j]):
#                        k=functional.k_j_list[j] -u-1
#                        vect_field_k=functional.Module.ComputeField(GD_tk[k],Cont[k]).copy()
#
#
#                        detDphi=functional.image_domain.element(
#                                _linear_deform(detDphi.copy(),
#                                delta_t*vect_field_k)).copy()
#                        detDphi=functional.image_domain.element(detDphi.copy()*
#                                   functional.image_domain.element(np.exp(delta_t *
#                                     div_op(vect_field_k)))).copy()
#                        grad_S=functional.image_domain.element(
#                                       _linear_deform(grad_S.copy(),
#                                       delta_t * vect_field_k)).copy()
#                        I_t=functional.image_domain.element(
#                            _linear_deform(I_t.copy(),
#                                           delta_t*vect_field_k))
#
#                        tmp= grad_op(I_t).copy()
#                        tmp1=(grad_S * detDphi).copy()
#                        for d in range(dim):
#                            tmp[d] *= tmp1
#
#                        vect_derGD_op=functional.Module.ComputeFieldDer(GD_tk[k],Cont[k])
#
#                        for i in range(nH):
#                            dGD=gamma_derCont[i][k].copy()
#                            vect_derGD=vect_derGD_op(dGD).copy()
#                            #tmp3= (2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(
#                            #        vectorial_ft_fit_op(tmp) * vectorial_ft_fit_op(vect_derGD)).copy()
#                            #grad1[i][k]-=(1/(functional.N+1))*tmp3.copy()
#                            tmp3=tmp.inner(vect_derGD)
#                            grad1[k]-=((1/(functional.N+1))*tmp3)*functional.basisCont[i].copy()
#
#                            vect_i=functional.Module.ComputeField(GD_tk[k],functional.Module.basisCont[i]).copy()
#                            #tmp3= (2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(
#                             #       vectorial_ft_fit_op(tmp) * vectorial_ft_fit_op(vect_i)).copy()
#                            #grad2[i][k]-=(1/(functional.N+1))*tmp3.copy()
#                            tmp3=tmp.inner(vect_i)
#                            grad2[k]-=((1/(functional.N+1))*tmp3)*functional.basisCont[i].copy()
#
#
#
#                        for i in range(nX):
#                            dGD=gamma_derGD[i][k].copy()
#                            vect_derGD=vect_derGD_op(dGD).copy()
#                            #tmp3= (2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(
#                            #        vectorial_ft_fit_op(tmp) * vectorial_ft_fit_op(vect_derGD)).copy()
#                            #grad3[i]-=(1/(functional.N+1))*tmp3.copy()
#                            tmp3=tmp.inner(vect_derGD)
#                            grad3-=((1/(functional.N+1))*tmp3)*functional.basisGD[i].copy()
#
#
#
#                gradCont=(grad1 + grad2).copy()
#
#                for i in range(functional.N+1):
#                    grad3+= functional.lam*functional.inv_N*functional.Module.CostGradGD(GD_tk[i], Cont[i]).copy()
#                    gradCont+= functional.lam*functional.inv_N*functional.Module.CostGradCont(GD_tk[i], Cont[i]).copy()
#
#                grad=functional.domain.element([grad3,gradCont])
#
#
#                return grad
#
#        return FunctionalModulesGeomGradient()
#



