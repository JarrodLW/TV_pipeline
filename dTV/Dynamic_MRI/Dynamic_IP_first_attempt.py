import odl
import numpy as np
from time import time
from myOperators import RealFourierTransform, Complex2Real, Real2Complex
from Utils import *

data_stack = np.zeros((3, 2, 32, 32))

height = data_stack.shape[2]
width = data_stack.shape[3]
frame_num = data_stack.shape[0]
complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.], shape=[height, width], dtype='complex')
image_space = complex_space.real_space ** 2
recon_space = image_space ** frame_num
fourier_transform = RealFourierTransform(image_space)
forward_op_diagonal = [fourier_transform]*frame_num
forward_op = odl.operator.pspace_ops.DiagonalOperator(*forward_op_diagonal)

data_odl = forward_op.range.element(data_stack)
data_fit = odl.solvers.L2Norm(forward_op.range).translated(data_odl)

## broadcasting spatial gradient operator and regulariser
alpha = 10**3
G_0 = odl.Gradient(image_space[0]) * odl.ComponentProjection(image_space, 0)
G_1 = odl.Gradient(image_space[1]) * odl.ComponentProjection(image_space, 1)
gradient_codomain = complex_space.real_space ** 4
# stacking derivatives - must be a better way to do this
projection_real_derivs = odl.BroadcastOperator(odl.ComponentProjection(gradient_codomain, 0),
                                               odl.ComponentProjection(gradient_codomain, 1))
projection_imag_derivs = odl.BroadcastOperator(odl.ComponentProjection(gradient_codomain, 2),
                                               odl.ComponentProjection(gradient_codomain, 3))
embedding_real_derivs = projection_real_derivs.adjoint
embedding_imag_derivs = projection_imag_derivs.adjoint
G = embedding_real_derivs*G_0 + embedding_imag_derivs*G_1
# the norm
group_norm_spatial = alpha * odl.solvers.GroupL1Norm(gradient_codomain)
spatial_gradients_diagonal = [G]*frame_num
spatial_gradients_combined = odl.operator.pspace_ops.DiagonalOperator(*spatial_gradients_diagonal)
group_norms_spatial = [group_norm_spatial]*frame_num
group_norms_spatial_combined = odl.solvers.SeparableSum(*group_norms_spatial)

## the time-variation regulariser

gamma = 10**2
group_norm_temporal = gamma * odl.solvers.GroupL1Norm(image_space)
time_grads_combined = odl.operator.pspace_ops.ProductSpaceOperator(discretised_time_gradients(frame_num, image_space))
group_norms_temporal = [group_norm_temporal]*(frame_num - 1)
group_norms_temporal_combined = odl.solvers.SeparableSum(*group_norms_temporal)


def discretised_time_gradients(dim, space):

    ''' dim: a positive integer
        space: a uniform odl (product) space '''

    id_op = odl.IdentityOperator(space)
    zero_op = odl.ZeroOperator(space)
    time_step_diff_list = []
    for i in range(dim-1):
        row_list = []
        for j in range(dim):
            if j==i:
                row_list.append(-id_op)
            elif j==i+1:
                row_list.append(id_op)
            else:
                row_list.append(zero_op)

        time_step_diff_list.append(row_list)

    return time_step_diff_list


op = odl.BroadcastOperator(forward_op, spatial_gradients_combined, time_grads_combined)
g = odl.solvers.SeparableSum(data_fit, group_norms_spatial_combined, group_norms_temporal_combined)
f = odl.ZeroOperator(recon_space)

# space_time_cube = odl.uniform_discr(min_pt=[-1., -1., -1.], max_pt=[1., 1., 1.], shape=[frame_num, height, width], dtype='complex')
# grad = odl.Gradient(space_time_cube, pad_mode='symmetric')
# time_grad = odl.ComponentProjection(grad.range, 0)*grad
