#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:39:06 2020

@author: jlw31
"""

import odl
import numpy as np
from time import time
# from Utils import *


def regularised_recons_from_subsampled_data(data_stack, measurement_type,
                                            reg_type, reg_param, recon_dims=None,
                                            subsampling_arr=None, niter=200, recon_init=None,
                                            enforce_positivity=False, a_offset=None, a_range=None,
                                            d_offset=None, d_width=None):
    # data_stack: a rank 3 numpy array
    # measurement_type: string 'MRI', 'CT', 'STEM'
    # reg_type: string 'TV', 'TGV'
    # reg_param: float
    # recon_dims: tuple, only needed for CT
    # subsampling_array: a rank 2 numpy array of the same dimensions as the data

    # in case a single 2d array is passed rather than a stack...
    if len(data_stack.shape) == 2:
        data_stack = np.expand_dims(data_stack, axis=0)

    # --- Building the forward operator and image space --- #
    if measurement_type == 'STEM':

        _, height, width = data_stack.shape
        image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                        shape=[height, width], dtype='float')

        forward_op = odl.IdentityOperator(image_space)

    elif measurement_type == 'MRI':

        height = data_stack.shape[1]
        width = data_stack.shape[2]
        image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                        shape=[height, width], dtype='complex')
        forward_op = odl.trafos.FourierTransform(image_space, halfcomplex=False)

    elif measurement_type == 'CT':

        assert recon_dims is not None, "reconstruction dimensions not provided"
        assert a_offset is not None, "angle offset not provided"
        assert a_range is not None, "max. angle not provided"
        assert d_offset is not None, "detector offset not provided"
        assert d_width is not None, "detector width not provided"

        height, width = recon_dims
        image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                                        shape=[height, width], dtype='float')
        # Make a parallel beam geometry with flat detector
        angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, data_stack.shape[1])
        # Detector: uniformly sampled
        detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, data_stack.shape[2])
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

        # Create the forward operator
        forward_op = odl.tomo.RayTransform(image_space, geometry, impl='skimage')

    else:
        raise ValueError("Measurement type " + str(measurement_type) + " not implemented")

    # --- Composing the forward operator with the subsampling operator --- #
    if subsampling_arr is None:
        subsampled_forward_op = forward_op

    else:
        subsampled_forward_op = forward_op.range.element(subsampling_arr) * forward_op

        # --- Building the regulariser and cost functional --- #
    if reg_type == 'TV':
        # Column vector of two operators
        G = odl.Gradient(image_space)
        op = odl.BroadcastOperator(subsampled_forward_op, G)
        l1_norms = [reg_param * odl.solvers.L1Norm(G.range)]

    elif reg_type == 'TGV':
        # adapted from "odl/examples/solvers/pdhg_denoising_tgv.py"

        G = odl.Gradient(image_space, method='forward', pad_mode='symmetric')
        V = G.range

        Dx = odl.PartialDerivative(image_space, 0, method='backward', pad_mode='symmetric')
        Dy = odl.PartialDerivative(image_space, 1, method='backward', pad_mode='symmetric')

        # Create symmetrized operator and weighted space.
        E = odl.operator.ProductSpaceOperator(
            [[Dx, 0], [0, Dy], [0.5 * Dy, 0.5 * Dx], [0.5 * Dy, 0.5 * Dx]])
        W = E.range

        domain = odl.ProductSpace(image_space, V)

        op = odl.BroadcastOperator(
        subsampled_forward_op * odl.ComponentProjection(domain, 0),
        odl.ReductionOperator(G, odl.ScalingOperator(V, -1)), E * odl.ComponentProjection(domain, 1))
        #
        l1_norms = [reg_param * odl.solvers.L1Norm(V), reg_param * odl.solvers.L1Norm(W)]

    else:
        raise ValueError("Regulariser " + str(reg_type) + " not implemented")

    if enforce_positivity:
        if reg_type == 'TV':
            f = odl.solvers.IndicatorNonnegativity(image_space)
        elif reg_type == 'TGV':
            f = odl.solvers.SeparableSum(odl.solvers.IndicatorNonnegativity(image_space), odl.solvers.ZeroFunctional(V))
    else:
        f = odl.solvers.ZeroFunctional(op.domain)

    # --- Running PDHG --- #
    if measurement_type == 'MRI':
        reconstructions = np.zeros((data_stack.shape[0], height, width), dtype='complex')
    else:
        reconstructions = np.zeros((data_stack.shape[0], height, width))

    for i in range(data_stack.shape[0]):

        data = data_stack[i, :, :]

        if subsampling_arr is not None:
            data = subsampling_arr * data

        # recasting the data in the appropriate form
        data_odl = forward_op.range.element(data)

        # l2-squared data matching
        l2_norm_squared = odl.solvers.L2NormSquared(forward_op.range).translated(data_odl)

        # Make separable sum of functionals, order must be the same as in `op`
        g = odl.solvers.SeparableSum(l2_norm_squared, *l1_norms)

        # --- Select solver parameters and solve using PDHG --- #
        # Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
        op_norm = 1.1 * odl.power_method_opnorm(op)
        tau = 1.0 / op_norm  # Step size for the primal variable
        sigma = 1.0 / op_norm  # Step size for the dual variable

        # Choose a starting point
        if recon_init is None:
            x = op.domain.zero()
        else:
            if reg_type == 'TV':
                x = op.domain.element(recon_init)
            elif reg_type == 'TGV':
                x = op.domain.zero()
                x[0] = image_space.element(recon_init)
                x[1] = V.zero()

        # Run the algorithm
        print('Running PDHG on data ' + str(i + 1) + " of " + str(data_stack.shape[0]))
        t0 = time()
        odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma)
        dt = time() - t0
        print('done in %.2fs.' % dt)

        if reg_type == 'TGV': # x consists not only of the image reconstruction but also the auxiliary vector field
            reconstructions[i, :, :] = x[0].asarray()
        else:
            reconstructions[i, :, :] = x.asarray()

    return reconstructions
