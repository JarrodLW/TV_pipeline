#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:39:06 2020

@author: jlw31
"""

import odl
import numpy as np
from time import time
from myOperators import RealFourierTransform, Complex2Real, Real2Complex
#import astra
from scipy.ndimage import interpolation
from Utils import *



class VariationalRegClass:

    # Decide on grid size. Should width be equal to number to pixels? Seems to be the case in MRI.

    def __init__(self, measurement_type, reg_type):

        self.measurement_type = measurement_type
        self.reg_type = reg_type
        self.image_space = None
        self.subsampled_forward_op = None
        self.reg_param = None
        self.reg_param_2 = None

    def regularised_recons_from_subsampled_data(self, data_stack,
                                                reg_param, recon_dims=None,
                                                subsampling_arr=None, niter=200, recon_init=None,
                                                enforce_positivity=False, a_offset=None, a_range=None,
                                                d_offset=None, d_width=None, reg_param_2=1):
        # data_stack: a rank 3 numpy array
        # measurement_type: string 'MRI', 'CT', 'STEM'
        # reg_type: string 'TV', 'TGV'
        # reg_param: float
        # recon_dims: tuple, only needed for CT
        # subsampling_array: a rank 2 numpy array of the same dimensions as the data

        self.reg_param = reg_param
        self.reg_param_2 = reg_param_2
        self.enforce_positivity = enforce_positivity

        # in case a single 2d array is passed rather than a stack...
        if len(data_stack.shape) == 2:
            data_stack = np.expand_dims(data_stack, axis=0)

        # --- Building the forward operator and image space --- #
        if self.measurement_type == 'STEM':

            _, height, width = data_stack.shape
            self.image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                            shape=[height, width], dtype='float')

            forward_op = odl.IdentityOperator(self.image_space)

        elif self.measurement_type == 'MRI':

            height = data_stack.shape[1]
            width = data_stack.shape[2]
            complex_space = odl.uniform_discr(min_pt=[-width//2, -height//2], max_pt=[width//2, height//2],
                                            shape=[height, width], dtype='complex')
            self.image_space = complex_space.real_space ** 2
            forward_op = RealFourierTransform(self.image_space)

        elif self.measurement_type == 'CT':

            assert recon_dims is not None, "reconstruction dimensions not provided"
            assert a_offset is not None, "angle offset not provided"
            assert a_range is not None, "max. angle not provided"
            assert d_offset is not None, "detector offset not provided"
            assert d_width is not None, "detector width not provided"

            height, width = recon_dims
            self.image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                            shape=[height, width], dtype='float')
            # Make a parallel beam geometry with flat detector
            angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, data_stack.shape[1])
            # Detector: uniformly sampled
            detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, data_stack.shape[2])
            geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

            # Create the forward operator
            #forward_op = odl.tomo.RayTransform(self.image_space, geometry, impl='skimage')  # should be using astra!
            forward_op = odl.tomo.RayTransform(self.image_space, geometry, impl='astra_cpu')
        else:
            raise ValueError("Measurement type " + str(self.measurement_type) + " not implemented")

        # --- Composing the forward operator with the subsampling operator --- #
        if subsampling_arr is None:
            self.subsampled_forward_op = forward_op

        else:
            if self.measurement_type == 'MRI':
                subsampling_arr_doubled = np.zeros((2, *subsampling_arr.shape))
                subsampling_arr_doubled[0, :, :] = subsampling_arr[:, :]
                subsampling_arr_doubled[1, :, :] = subsampling_arr[:, :]
                self.subsampled_forward_op = forward_op.range.element(subsampling_arr_doubled) * forward_op
            else:
                self.subsampled_forward_op = forward_op.range.element(subsampling_arr) * forward_op

            # --- Building the regulariser and cost functional --- #
        if self.reg_type == 'TV':
            # Column vector of two operators

            op, reg_norms = self.build_tv_model()

        elif self.reg_type == 'TGV':
            # adapted from "odl/examples/solvers/pdhg_denoising_tgv.py"

            op, reg_norms = self.build_tgv_model()

        else:
            raise ValueError("Regulariser " + str(self.reg_type) + " not implemented")

        if enforce_positivity:
            if self.reg_type == 'TV':
                f = odl.solvers.IndicatorNonnegativity(self.image_space)
            elif self.reg_type == 'TGV':
                f = odl.solvers.SeparableSum(odl.solvers.IndicatorNonnegativity(self.image_space),
                                             odl.solvers.ZeroFunctional(V))
        else:
            f = odl.solvers.ZeroFunctional(op.domain)

        # --- Running PDHG --- #
        if self.measurement_type == 'MRI':
            reconstructions = np.zeros((data_stack.shape[0], height, width), dtype='complex')
        else:
            reconstructions = np.zeros((data_stack.shape[0], height, width))

        for i in range(data_stack.shape[0]):
            data = data_stack[i, :, :]

            if subsampling_arr is not None:
                data = subsampling_arr * data

            # recasting the data in the appropriate form

            if self.measurement_type == 'MRI':
                data_complex_odl = complex_space.element(data)
                J = Complex2Real(complex_space)
                data_odl = J(data_complex_odl)

            else:
                data_odl = forward_op.range.element(data)

            # l2-squared data matching
            l2_norm_squared = odl.solvers.L2NormSquared(forward_op.range).translated(data_odl)

            # Make separable sum of functionals, order must be the same as in `op`
            g = odl.solvers.SeparableSum(l2_norm_squared, *reg_norms)

            # --- Select solver parameters and solve using PDHG --- #
            # Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
            op_norm = 1.1 * odl.power_method_opnorm(op)
            tau = 1.0 / op_norm  # Step size for the primal variable
            sigma = 1.0 / op_norm  # Step size for the dual variable

            # Choose a starting point
            if recon_init is None:
                x = op.domain.zero()
            else:
                if self.reg_type == 'TV':
                    x = op.domain.element(recon_init)
                elif self.reg_type == 'TGV':
                    x = op.domain.zero()
                    x[0] = self.image_space.element(recon_init)
                    x[1] = V.zero()

            # Run the algorithm
            print('Running PDHG on data ' + str(i + 1) + " of " + str(data_stack.shape[0]))
            t0 = time()
            print("so far so good")
            odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma)
            dt = time() - t0
            print('done in %.2fs.' % dt)

            if self.measurement_type == 'MRI':
                RealToComplexOp = Real2Complex(self.image_space)

                if self.reg_type == 'TGV': # x consists not only of the image reconstruction but also the auxiliary vector field
                    reconstructions[i, :, :] = np.fft.fftshift(RealToComplexOp(x[0].asarray()))
                else:
                    reconstructions[i, :, :] = np.fft.fftshift(RealToComplexOp(x.asarray()))

            else:
                if self.reg_type == 'TGV':  # x consists not only of the image reconstruction but also the auxiliary vector field
                    reconstructions[i, :, :] = x[0].asarray()
                else:
                    reconstructions[i, :, :] = x.asarray()

        return reconstructions

    def build_tv_model(self):

        if self.measurement_type == 'MRI':
            G_0 = odl.Gradient(self.image_space[0]) * odl.ComponentProjection(self.image_space, 0)
            G_1 = odl.Gradient(self.image_space[1]) * odl.ComponentProjection(self.image_space, 1)
            op = odl.BroadcastOperator(self.subsampled_forward_op, G_0, G_1)
            reg_norms = [self.reg_param * odl.solvers.GroupL1Norm(G_0.range),
                        self.reg_param * odl.solvers.GroupL1Norm(G_1.range)]

        else:
            G = odl.Gradient(self.image_space)
            op = odl.BroadcastOperator(self.subsampled_forward_op, G)
            reg_norms = [self.reg_param * odl.solvers.GroupL1Norm(G.range)]

        return op, reg_norms

    def build_tgv_model(self):

        if self.measurement_type == 'MRI': # This can be made neater with product spaces
            G_0 = odl.Gradient(self.image_space[0]) * odl.ComponentProjection(self.image_space, 0)
            G_1 = odl.Gradient(self.image_space[1]) * odl.ComponentProjection(self.image_space, 1)
            V_0 = G_0.range
            V_1 = G_1.range

            Dx_0 = odl.PartialDerivative(self.image_space[0], 0, method='backward', pad_mode='symmetric')
            Dy_0 = odl.PartialDerivative(self.image_space[0], 1, method='backward', pad_mode='symmetric')
            Dx_1 = odl.PartialDerivative(self.image_space[1], 0, method='backward', pad_mode='symmetric')
            Dy_1 = odl.PartialDerivative(self.image_space[1], 1, method='backward', pad_mode='symmetric')

            # Create symmetrized operator and weighted space.
            E_0 = odl.operator.ProductSpaceOperator(
                [[Dx_0, 0], [0, Dy_0], [0.5 * Dy_0, 0.5 * Dx_0], [0.5 * Dy_0, 0.5 * Dx_0]])
            E_1 = odl.operator.ProductSpaceOperator(
                [[Dx_1, 0], [0, Dy_1], [0.5 * Dy_1, 0.5 * Dx_1], [0.5 * Dy_1, 0.5 * Dx_1]])
            W_0 = E_0.range
            W_1 = E_1.range

            domain = odl.ProductSpace(self.image_space, V_0, V_1)

            op = odl.BroadcastOperator(
                self.subsampled_forward_op * odl.ComponentProjection(domain, 0),
                odl.ReductionOperator(G_0, odl.ScalingOperator(V_0, -1), odl.ZeroOperator(V_1)),
                odl.ReductionOperator(G_1, odl.ZeroOperator(V_0), odl.ScalingOperator(V_1, -1)),
                E_0 * odl.ComponentProjection(domain, 1), E_1 * odl.ComponentProjection(domain, 2))

            reg_norms = [self.reg_param * odl.solvers.GroupL1Norm(V_0), self.reg_param * odl.solvers.GroupL1Norm(V_1),
                        self.reg_param_2 * self.reg_param * odl.solvers.GroupL1Norm(W_0),
                        self.reg_param_2 * self.reg_param * odl.solvers.GroupL1Norm(W_1)]

        else:
            G = odl.Gradient(self.image_space, method='forward', pad_mode='symmetric')
            V = G.range

            Dx = odl.PartialDerivative(self.image_space, 0, method='backward', pad_mode='symmetric')
            Dy = odl.PartialDerivative(self.image_space, 1, method='backward', pad_mode='symmetric')

            # Create symmetrized operator and weighted space.
            E = odl.operator.ProductSpaceOperator(
                [[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx], [0.5 * Dy, 0.5 * Dx]])
            W = E.range

            domain = odl.ProductSpace(self.image_space, V)

            op = odl.BroadcastOperator(
                self.subsampled_forward_op * odl.ComponentProjection(domain, 0),
                odl.ReductionOperator(G, odl.ScalingOperator(V, -1)), E * odl.ComponentProjection(domain, 1))
            #
            reg_norms = [self.reg_param * odl.solvers.GroupL1Norm(V),
                         self.reg_param_2 * self.reg_param * odl.solvers.GroupL1Norm(W)]

        return op, reg_norms


def recon_astra(sinogram, center, angles=None, ratio=1.0, method="SIRT", num_iter=1, win="hann", pad=0):
    # Taken from Vo's code
    """
    Wrapper of reconstruction methods implemented in the astra toolbox package.
    https://www.astra-toolbox.com/docs/algs/index.html
    ---------
    Parameters: - sinogram: 2D tomographic data.
                - center: center of rotation.
                - angles: tomographic angles in radian.
                - ratio: apply a circle mask to the reconstructed image.
                - method: Reconstruction algorithms
                    for CPU: 'FBP', 'SIRT', 'SART', 'ART', 'CGLS'.
                    for GPU: 'FBP_CUDA', 'SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA'.
                - num_iter: Number of iterations if using iteration methods.
                - filter: apply filter if using FBP method:
                    'hamming', 'hann', 'lanczos', 'kaiser', 'parzen',...
                - pad: padding to reduce the side effect of FFT.
    ---------
    Return:     - square array.
    """
    if pad > 0:
        sinogram = np.pad(sinogram, ((0, 0), (pad, pad)), mode='edge')
        center = center + pad
    (nrow, ncol) = sinogram.shape
    if angles is None:
        angles = np.linspace(0.0, 180.0, nrow) * np.pi / 180.0
    proj_geom = astra.create_proj_geom('parallel', 1, ncol, angles)
    vol_geom = astra.create_vol_geom(ncol, ncol)
    cen_col = (ncol - 1.0) / 2.0
    shift = cen_col - center
    sinogram = interpolation.shift(sinogram, (0, shift), mode='nearest')
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict(method)
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    if method == "FBP_CUDA":
        cfg["FilterType"] = win
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iter)
    rec = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    if pad > 0:
        rec = rec[pad:-pad, pad:-pad]
    if not (ratio is None):
        rec = rec * circle_mask(rec.shape[0], ratio)
    return rec


