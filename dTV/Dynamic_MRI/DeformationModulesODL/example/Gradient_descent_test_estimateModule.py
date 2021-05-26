#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:42:31 2017

@author: barbara
"""

"""
Gradient descent when the module is FromFile
different step size for centre and theta
"""

#%% Gradient descent Deformation module


#GD_init=Module.GDspace.zero()
#GD_init=Module.GDspace.element([[[3, 0],0]])
#GD_init=Module.GDspace.element([[1,-1]])
#GD_init=Module.GDspace.element([[[0,0],0]])
Cont_init=odl.ProductSpace(Module.Contspace,nb_time_point_int+1).zero()
#GD_init=Module.GDspace.element([[[-0.77478469, -9.85281818], -0.0*np.pi, [0.0,0.0], [1.0,0.0]]])
GD_init=Module.GDspace.element([[[-1,0], -0.0*np.pi, [-1.77478469,-9.85281818], [1.2521,-9.85281818]]])
#Cont_init=Module.Contspace.zero()

vect_field=Module.ComputeField(GD_init,Module.Contspace.one())
vect_field.show('bis2')


niter=500
eps = 0.01

X=functional_mod.domain.element([GD_init,Cont_init].copy())

vector_fields_list_init=energy_op_lddmm.domain.zero()
vector_fields_list=vector_fields_list_init.copy()
#%%
dim=2
attachment_term=attach_mod(X[0],X[1])
energy=attachment_term
Reg_mod=functional_mod.ComputeReg(X)
print(" Initial , attachment term : {}, reg_mod = {}".format(attachment_term,Reg_mod))
gradGD=functional_mod.Module.GDspace.element()
gradCont=odl.ProductSpace(functional_mod.Module.Contspace,nb_time_point_int+1).element()

d_GD=functional_mod.Module.GDspace.zero()
d_Cont=odl.ProductSpace(functional_mod.Module.Contspace,nb_time_point_int+1).zero()
ModulesList=Module.ModulesList
NbMod=len(ModulesList)
delta=0.1
epsmax=0.5
deltamin=0.1
deltaCont=0.1
deltaGD=0.2
# 0=SumTranslations, 1=affine, 2=scaling, 3=rotation, 4 = ellipsemvt
Types=[4]
#energy=functional(X)+1
inv_N=1/nb_time_point_int
epsContmax=1
epsGDmax=1
epsCont=0.1
epsGD_Pts=0.1
epsGD_theta=0.1
epsGD_c=0.1
epsGD_ab=0.0
eps_vect_field=0.1
cont=1
space_pts=template.space.points()


energy=attach_mod(X[0],X[1])
Reg_mod=functional_mod.ComputeReg(X)
energy_mod=Reg_mod+energy

#%%
print('Initial  energy = {} '.format(energy_mod))
for k in range(niter):
    gradGD=functional_mod.Module.GDspace.zero()
    gradCont=odl.ProductSpace(functional_mod.Module.Contspace,nb_time_point_int+1).zero()


    #energy=attach_tot(vector_fields_list,X[0],X[1])
    #Reg_mod=functional_mod.ComputeReg(X)
    #energy_mod=Reg_mod+energy
    GD=ComputeGD_list(X[0],X[1]).copy()
    #print('k={}  before vect field attachment term = {}, reg_mod={}'.format(k,energy,Reg_mod))
    # gradient with respect to vector field
    #grad_vect_field=grad_attach_vector_field(X[0],X[1])
    grad_vect_field=grad_attach_vector_fieldL2(X[0],X[1])
    # (1-lamb1) because of the gradient of the regularity term

    #GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()

    #energy=attach_tot(vector_fields_list,X[0],X[1])
    #Reg_mod=functional_mod.ComputeReg(X)
    #energy_mod=Reg_mod+energy
    #print('      k={}  after vect field  attachment term = {}, reg_mod={}'.format(k,energy,Reg_mod))

    for i in range(NbMod):

        basisCont=ModulesList[i].basisCont.copy()
        dimCont=len(basisCont)
        #print('i = {}'.format(i))
        for iter_cont in range(dimCont):
            for t in range(nb_time_point_int):
                X_temp=X.copy()
                X_temp[1][t][i]+=deltaCont*basisCont[iter_cont]
                GD_diff=ComputeGD_list(X[0],X_temp[1]).copy()
                temp=0
                #print('t = {}'.format(t))
                for s in range(nb_time_point_int):
                    # vec_temp is the derivative of the generated vector field at s with respect to h[t][iter_cont]
                    vec_temp=( Module.ComputeField(GD_diff[s], X_temp[1][s]).copy()-Module.ComputeField(GD[s], X[1][s]).copy() )/deltaCont

                    # It is necessary to interpolate in order to do the inner product
                    vec_temp_interp=space.tangent_bundle.element(vec_temp).copy()

                    temp+=inv_N*grad_vect_field[s].inner(vec_temp_interp)
                    #print('s = {}'.format(s))

                #ATTENTION : WE SUPPOSE THAT THE DERIVATIVE OF THE COST
                # WITH RESPECT TO GD IS NULL
                temp+=lamb0*Module.CostGradCont(GD[t], X[1][t])[iter_cont]
                gradCont[t][i][iter_cont]+=temp
    for i in range(NbMod):
        basisGD=ModulesList[i].basisGD.copy()
        dimGD=len(basisGD)

        for iter_gd in range(dimGD):
            X_temp=X.copy()
            X_temp[0][i]+=deltaGD*basisGD[iter_gd]
            GD_diff=ComputeGD_list(X_temp[0],X_temp[1]).copy()
            temp=0
            for s in range(nb_time_point_int):
                # vec_temp is the derivative of the generated vector field at s with respect to GD[iter_cont]
                vec_temp=( Module.ComputeField(GD_diff[s], X_temp[1][s]).copy()-Module.ComputeField(GD[s], X[1][s]).copy() )/deltaGD

                # It is necessary to interpolate in order to do the inner product
                vec_temp_interp=space.tangent_bundle.element(vec_temp).copy()
                        #[vec_temp[u].interpolation(space_pts.T) for u in range(dim)]).copy()

                temp+=inv_N*grad_vect_field[s].inner(vec_temp_interp)
            gradGD[i]+=temp*basisGD[iter_gd]

    for ite in range(20):
        X_temp=X.copy()
        X_temp[1]-=epsCont*gradCont.copy()
        X_temp[0][0][0]-=epsGD_c*gradGD[0][0]
        X_temp[0][0][1]-=epsGD_theta*gradGD[0][1]
        X_temp[0][0][2]-=epsGD_ab*gradGD[0][2]
        X_temp[0][0][3]-=epsGD_ab*gradGD[0][3]
        energy=attach_mod(X_temp[0],X_temp[1])
        Reg_mod=functional_mod.ComputeReg(X_temp)
        energy_mod0=Reg_mod+energy

        X_temp=X.copy()
        X_temp[1]-=0.8*epsCont*gradCont.copy()
        X_temp[0][0][0]-=epsGD_c*gradGD[0][0]
        X_temp[0][0][1]-=epsGD_theta*gradGD[0][1]
        X_temp[0][0][2]-=epsGD_ab*gradGD[0][2]
        X_temp[0][0][3]-=epsGD_ab*gradGD[0][3]
        energy=attach_tot(vector_fields_list,X_temp[0],X_temp[1])
        Reg_mod=functional_mod.ComputeReg(X_temp)
        energy_mod1=Reg_mod+energy

        X_temp=X.copy()
        X_temp[1]-=epsCont*gradCont.copy()
        X_temp[0][0][0]-=0.8*epsGD_c*gradGD[0][0]
        X_temp[0][0][1]-=epsGD_theta*gradGD[0][1]
        X_temp[0][0][2]-=epsGD_ab*gradGD[0][2]
        X_temp[0][0][3]-=epsGD_ab*gradGD[0][3]
        energy=attach_tot(vector_fields_list,X_temp[0],X_temp[1])
        Reg_mod=functional_mod.ComputeReg(X_temp)
        energy_mod2=Reg_mod+energy

        X_temp=X.copy()
        X_temp[1]-=epsCont*gradCont.copy()
        X_temp[0][0][0]-=epsGD_c*gradGD[0][0]
        X_temp[0][0][1]-=0.8*epsGD_theta*gradGD[0][1]
        X_temp[0][0][2]-=epsGD_ab*gradGD[0][2]
        X_temp[0][0][3]-=epsGD_ab*gradGD[0][3]
        energy=attach_mod(X_temp[0],X_temp[1])
        Reg_mod=functional_mod.ComputeReg(X_temp)
        energy_mod3=Reg_mod+energy

        X_temp=X.copy()
        X_temp[1]-=epsCont*gradCont.copy()
        X_temp[0][0][0]-=epsGD_c*gradGD[0][0]
        X_temp[0][0][1]-=epsGD_theta*gradGD[0][1]
        X_temp[0][0][2]-=0.8*epsGD_ab*gradGD[0][2]
        X_temp[0][0][3]-=0.8*epsGD_ab*gradGD[0][3]
        energy=attach_mod(X_temp[0],X_temp[1])
        Reg_mod=functional_mod.ComputeReg(X_temp)
        energy_mod4=Reg_mod+energy



        print('energy0 = {}, energy1 = {}, energy2 = {}, energy3 = {} '.format(energy_mod0,energy_mod1,energy_mod2,energy_mod3) )
        if (energy_mod0 <= energy_mod1 and energy_mod0 <= energy_mod2 and energy_mod0 <= energy_mod3 and energy_mod0 <= energy_mod4):
            X_temp=X.copy()
            X_temp[1]-=epsCont*gradCont.copy()
            X_temp[0][0][0]-=epsGD_c*gradGD[0][0]
            X_temp[0][0][1]-=epsGD_theta*gradGD[0][1]
            X_temp[0][0][2]-=epsGD_ab*gradGD[0][2]
            X_temp[0][0][3]-=epsGD_ab*gradGD[0][3]
            energy_mod_temp=energy_mod0
        elif (energy_mod1 <= energy_mod0 and energy_mod1 <= energy_mod2 and energy_mod1 <= energy_mod3 and energy_mod1 <= energy_mod4):
            X_temp=X.copy()
            X_temp[1]-=0.8*epsCont*gradCont.copy()
            X_temp[0][0][0]-=epsGD_c*gradGD[0][0]
            X_temp[0][0][1]-=epsGD_theta*gradGD[0][1]
            X_temp[0][0][2]-=epsGD_ab*gradGD[0][2]
            X_temp[0][0][3]-=epsGD_ab*gradGD[0][3]
            energy_mod_temp=energy_mod1
            epsCont*=0.8
        elif (energy_mod2 <= energy_mod0 and energy_mod2 <= energy_mod1 and energy_mod2 <= energy_mod3 and energy_mod2 <= energy_mod4):
            X_temp=X.copy()
            X_temp[1]-=epsCont*gradCont.copy()
            X_temp[0][0][0]-=0.8*epsGD_c*gradGD[0][0]
            X_temp[0][0][1]-=epsGD_theta*gradGD[0][1]
            X_temp[0][0][2]-=epsGD_ab*gradGD[0][2]
            X_temp[0][0][3]-=epsGD_ab*gradGD[0][3]
            energy_mod_temp=energy_mod2
            epsGD_c*=0.8
        elif (energy_mod3 <= energy_mod0 and energy_mod3 <= energy_mod1 and energy_mod3 <= energy_mod3 and energy_mod3 <= energy_mod4):
            X_temp=X.copy()
            X_temp[1]-=epsCont*gradCont.copy()
            X_temp[0][0][0]-=epsGD_c*gradGD[0][0]
            X_temp[0][0][1]-=0.8*epsGD_theta*gradGD[0][1]
            X_temp[0][0][2]-=epsGD_ab*gradGD[0][2]
            X_temp[0][0][3]-=epsGD_ab*gradGD[0][3]
            energy_mod_temp=energy_mod3
            epsGD_theta*=0.8
        else:
            X_temp=X.copy()
            X_temp[1]-=epsCont*gradCont.copy()
            X_temp[0][0][0]-=epsGD_c*gradGD[0][0]
            X_temp[0][0][1]-=epsGD_theta*gradGD[0][1]
            X_temp[0][0][2]-=0.8*epsGD_ab*gradGD[0][2]
            X_temp[0][0][3]-=0.8*epsGD_ab*gradGD[0][3]
            energy_mod_temp=energy_mod4
            epsGD_ab*=0.8



        if (energy_mod_temp < energy_mod):
            X=X_temp.copy()
            energy_mod=energy_mod_temp
            print('k={} , energy = {} '.format(k,energy_mod_temp))
            print('GD =  {}'.format(X[0]))
            epsCont*=1.2
            epsGD_c*=1.2
            epsGD_theta*=1.2
            epsGD_ab*=1.2
            break
        else:
           epsCont*=0.8
           epsGD_c*=0.8
           epsGD_theta*=0.8
           epsGD_ab*=0.8

    if (ite==19):
        print('No possible to descent')
        break

    if epsGD_Pts>epsmax:
        epsGD_Pts=epsmax

    if epsGD_theta>epsmax:
        epsGD_theta=epsmax

    print('epsGDc= {} ,epsGDtheta= {},  epsGD_ab ={}, epsCont = {}'.format(epsGD_c,epsGD_theta,epsGD_ab,epsCont))
    #vector_fields_list=((1-eps_vect_field*lamb1)*vector_fields_list-eps_vect_field*grad_vect_field ).copy()
#

#%%
#I_t=Shoot_mixt(vector_fields_list,X[0],X[1])
vect_field_list_mod=vect_field_list(X[0],X[1])
I_t=TemporalAttachmentModulesGeom.ShootTemplateFromVectorFields(vect_field_list_mod, template)

name='/home/bgris/Results/DeformationModules/testEstimation/Rotation/EstimatedTrajectory_FromFileV6_from_vect_field_V5_Rotation_SheppLogan'
#name+= '_sigma_2_INDIRECT_num_angle_10_initial_epsCont_0_01_fixed_GD'
name+= '_sigma_lddmm_1_sigma_cp_0_5_nbtrans_30_expectedvalue_not_optimized_GD'
image_N0=I_t
plot_result(name,image_N0)


#%%
GD_list=ComputeGD_list(X[0],X[1])

points=space.points()

for k in range(nb_time_point_int):
    v0=Module.ComputeField(GD_list[k],X[1][k])
    plt.figure()
    plt.quiver(points.T[0][::20],points.T[1][::20],v0[0][::20],v0[1][::20])
    plt.axis('equal')
    plt.title('Vector field at time {}'.format(k))
#
name_mod=name+'plotGD'
plot_result_moduleV5(name_mod,image_N0,GD_list)

for i in range(nb_time_point_int+1):
    #vect_field_list_mod[i][0].show('{}'.format(i))
    I_t[i].show('{}'.format(i))
    plt.plot(GD_list[i][0][2][0],GD_list[i][0][2][1],'o')
    plt.plot(GD_list[i][0][3][0],GD_list[i][0][3][1],'x')

#    plt.plot(GD_init[0][2][0],GD_init[0][2][1],'o')
#    plt.plot(GD_init[0][3][0],GD_init[0][3][1],'x')
