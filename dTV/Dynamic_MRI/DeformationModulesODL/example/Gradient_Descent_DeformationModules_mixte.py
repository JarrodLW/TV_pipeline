#%%
GD_init=Module.GDspace.element([[[-0.0500, -9.5]]])
GD_init=Module.GDspace.element([[[3,0]]])
Cont_init=odl.ProductSpace(Module.Contspace,nb_time_point_int+1).zero()

#%% Naive Gradient descent : gradient computed by finite differences
# Descent of vect field too
# Descent for all times simultaneously
#Combination of modules
#functional=functionalF
niter=200
eps = 0.01

X=functional_mod.domain.element([GD_init,Cont_init].copy())

vector_fields_list_init=energy_op_lddmm.domain.zero()
vector_fields_list=vector_fields_list_init.copy()
#%%
attachment_term=attach_tot(vector_fields_list,X[0],X[1])
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
epsmax=1
deltamin=0.1
deltaCont=0.1
deltaGD=0.1
# 0=SumTranslations, 1=affine, 2=scaling, 3=rotation
Types=[3]
#energy=functional(X)+1
inv_N=1/nb_time_point_int
epsContmax=1
epsGDmax=1
epsCont=0.01
epsGD=0.000001
eps_vect_field=0.01
cont=1
space_pts=template.space.points()

GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()

for k in range(niter):
    energy=attach_tot(vector_fields_list,X[0],X[1])
    Reg_mod=functional_mod.ComputeReg(X)
    energy_mod=Reg_mod+energy
    GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()
    print('k={}  before vect field attachment term = {}, reg_mod={}'.format(k,energy,Reg_mod))
    # gradient with respect to vector field
    grad_vect_field=grad_attach_vector_field(vector_fields_list,X[0],X[1])
    # (1-lamb1) because of the gradient of the regularity term

    #GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()

    #energy=attach_tot(vector_fields_list,X[0],X[1])
    #Reg_mod=functional_mod.ComputeReg(X)
    #energy_mod=Reg_mod+energy
    #print('      k={}  after vect field  attachment term = {}, reg_mod={}'.format(k,energy,Reg_mod))

    for i in range(NbMod):

        basisCont=ModulesList[i].basisCont.copy()
        dimCont=len(basisCont)

        for iter_cont in range(dimCont):
            X_temp=X.copy()
            for t in range(nb_time_point_int):
                X_temp[1][t][i]+=deltaCont*basisCont[iter_cont]
            GD_diff=ComputeGD_mixt(vector_fields_list,X[0],X_temp[1])
            for t in range(nb_time_point_int+1):
                GD_diff[t]=(GD_diff[t]-GD[t])/deltaCont

            for t in range(nb_time_point_int):
                vect_field_der=ModulesList[i].ComputeFieldDer(GD[t][i],X[1][t][i])(GD_diff[t][i]).copy()
                vect_field_der+=ModulesList[i].ComputeField(GD[t][i],basisCont[iter_cont]).copy()
                vect_field_der_interp=template.space.tangent_bundle.element(vect_field_der).copy()
                gradCont[t][i]+=(grad_vect_field[t].inner(vect_field_der_interp)*basisCont[iter_cont]).copy()

    for i in range(NbMod):
        basisGD=ModulesList[i].basisGD.copy()
        dimGD=len(basisGD)

        for iter_gd in range(dimGD):
            X_temp=X.copy()
            X_temp[0][i]+=deltaGD*basisGD[iter_gd]
            GD_diff=ComputeGD_mixt(vector_fields_list,X_temp[0],X_temp[1])

            for t in range(nb_time_point_int+1):
                GD_diff[t]=(GD_diff[t]-GD[t])/deltaGD

            for t in range(nb_time_point_int):
                vect_field_der=ModulesList[i].ComputeFieldDer(GD[t][i],X[1][t][i])(GD_diff[t][i]).copy()
                vect_field_der_interp=template.space.tangent_bundle.element(vect_field_der).copy()
                gradGD[i]+=inv_N*(grad_vect_field[t].inner(vect_field_der_interp)*basisGD[iter_gd]).copy()

    X[1]-=epsCont*gradCont.copy()
    print(X[1][0])
    #X[0]-=epsGD*gradGD
    vector_fields_list=((1-eps_vect_field*lamb1)*vector_fields_list-eps_vect_field*grad_vect_field ).copy()
#


























#%%
for k in range(niter):
    energy=attach_tot(vector_fields_list,X[0],X[1])
    Reg_mod=functional_mod.ComputeReg(X)
    energy_mod=Reg_mod+energy
    GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()
    print('k={}  before vect field attachment term = {}, reg_mod={}'.format(k,energy,Reg_mod))
    # gradient with respect to vector field
    grad_vect_field=grad_attach_vector_field(vector_fields_list,X[0],X[1])
    # (1-lamb1) because of the gradient of the regularity term

    #GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()

    #energy=attach_tot(vector_fields_list,X[0],X[1])
    #Reg_mod=functional_mod.ComputeReg(X)
    #energy_mod=Reg_mod+energy
    #print('      k={}  after vect field  attachment term = {}, reg_mod={}'.format(k,energy,Reg_mod))

    for i in range(NbMod):

        basisCont=ModulesList[i].basisCont.copy()
        dimCont=len(basisCont)

        for iter_cont in range(dimCont):
            X_temp=X.copy()
            for t in range(nb_time_point_int):
                X_temp[1][t][i]+=deltaCont*basisCont[iter_cont]
            GD_diff=ComputeGD_mixt(vector_fields_list,X[0],X_temp[1])
            for t in range(nb_time_point_int+1):
                GD_diff[t]=(GD_diff[t]-GD[t])/deltaCont

            for t in range(nb_time_point_int):
                vect_field_der=ModulesList[i].ComputeFieldDer(GD[t][i],X[1][t][i])(GD_diff[t][i]).copy()
                vect_field_der+=ModulesList[i].ComputeField(GD[t][i],basisCont[iter_cont]).copy()
                vect_field_der_interp=template.space.tangent_bundle.element(vect_field_der).copy()
                gradCont[t][i]+=(grad_vect_field[t].inner(vect_field_der_interp)*basisCont[iter_cont]).copy()

    """
    X_temp=X.copy()
    if (eps0Cont>epsContmax):
        eps=epsContmax
    else:
        eps=eps0Cont

    X_temp[1]-=eps*gradCont.copy()
    energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])

    if(energy_temp>energy_mod):
        for ite in range(10):
            eps*=0.8
            X_temp=X.copy()
            X_temp[1]-=eps*gradCont.copy()
            energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
            if (energy_temp<energy_mod):
                ismax=1
                eps0Cont=eps
                break
    else:
        for ite in range(10):
            eps*=1.2
            X_temp=X.copy()
            X_temp[1]-=eps*gradCont.copy()
            energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
            if (energy_temp>=energy):
                eps/=1.2
                break
        eps0Cont=eps
        ismax=1


    # Now we have 'the best' eps
    if(ismax==1):
        X[1]-=eps*gradCont.copy()
        energy=attach_tot(vector_fields_list,X[0],X[1])
        energy_mod=functional_mod.ComputeReg(X)+attach_tot(vector_fields_list,X[0],X[1])
        Reg_mod=functional_mod.ComputeReg(X)
        GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()
        """

    for i in range(NbMod):
        basisGD=ModulesList[i].basisGD.copy()
        dimGD=len(basisGD)

        for iter_gd in range(dimGD):
            X_temp=X.copy()
            X_temp[0][i]+=deltaGD*basisGD[iter_gd]
            GD_diff=ComputeGD_mixt(vector_fields_list,X_temp[0],X_temp[1])

            for t in range(nb_time_point_int+1):
                GD_diff[t]=(GD_diff[t]-GD[t])/deltaGD

            for t in range(nb_time_point_int):
                vect_field_der=ModulesList[i].ComputeFieldDer(GD[t][i],X[1][t][i])(GD_diff[t][i]).copy()
                vect_field_der_interp=template.space.tangent_bundle.element(vect_field_der).copy()
                gradGD[i]+=inv_N*(grad_vect_field[t].inner(vect_field_der_interp)*basisGD[iter_gd]).copy()

            """
            X_temp=X.copy()
            if (eps0GD>epsGDmax):
                eps=epsGDmax
            else:
                eps=eps0GD

            X_temp[0]-=eps*gradGD
            energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
            print('energy_temp = {}'.format(energy_temp))
            if(energy_temp>energy):
                for ite in range(10):
                    eps*=0.8
                    X_temp=X.copy()
                    X_temp[0]-=eps*gradGD
                    energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                    if (energy_temp<energy_mod):
                        ismax=1
                        eps0GD=eps
                        break
            else:
                for ite in range(10):
                    eps*=1.2
                    X_temp=X.copy()
                    X_temp[0]-=eps*gradGD
                    energy_temp=functional_mod.ComputeReg(X_temp)+attach_tot(vector_fields_list,X_temp[0],X_temp[1])
                    if (energy_temp>energy_mod):
                        eps/=1.2
                        break
                eps0GD=eps
                ismax=1

            # Now we have 'the best' eps
            if (ismax==1):
                X[0]-=eps*gradGD
                energy=attach_tot(vector_fields_list,X[0],X[1])
                energy_mod=functional_mod.ComputeReg(X)+attach_tot(vector_fields_list,X[0],X[1])
                Reg_mod=functional_mod.ComputeReg(X)
                GD=ComputeGD_mixt(vector_fields_list,X[0],X[1]).copy()
            """

    X[1]-=epsCont*gradCont.copy()
    X[0]-=epsGD*gradGD
    vector_fields_list=((1-eps_vect_field*lamb1)*vector_fields_list-eps_vect_field*grad_vect_field ).copy()
#
