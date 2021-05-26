import numpy as np
import odl

class RealFourierTransformStacked(odl.Operator):

    def __init__(self, domain, mask=None):
        """TBC

        Applies the RealFourierTransform to each "spatial" slice of given time-series data.

        domain: a Product Space instance of two 3-dimensional uniform_discr odl spaces ---the real and imaginary parts---
                for each of which the zeroth index is "temporal" one, the first and second indices being spatial.

        mask: rank-3 numpy array, of same dimensions as domain.real_space

        Parameters
        ----------
        TBC

        Examples
        --------
        """

        # defining the fourier operator acting on each slice
        #self.domain = domain
        fourier_domain_complex = odl.uniform_discr(min_pt=domain[0].min_pt[1:], max_pt=domain[0].max_pt[1:],
                                                   shape=domain[0].shape[1:], dtype='complex')

        self.fourier = odl.trafos.DiscreteFourierTransform(fourier_domain_complex)
        self.mask = mask

        print(self.fourier.domain)

        #range = self.fourier.range.real_space ** 2 # needs changing?
        range = odl.uniform_discr(min_pt=[-1., 0., 0.], max_pt=[domain[0].max_pt[0], float(domain[0].shape[1]), float(domain[0].shape[2])],
                                                   shape=domain[0].shape, nodes_on_bdry=True)**2

        print(range)

        super(RealFourierTransformStacked, self).__init__(
            domain=domain, range=range, linear=True)

    def _call(self, x, out):

        for i in range(self.domain.shape[1]):
            Fx = self.fourier(np.fft.fftshift(x[0, i].asarray()) + 1j * np.fft.fftshift(x[1, i].asarray()))

            if not isinstance(self.mask, type(None)):
                Fx = self.mask[i] * Fx

            out[0, i][:] = np.real(Fx)
            out[1, i][:] = np.imag(Fx)

        #print(self.domain[0].cell_volume)
        print(self.fourier.domain.cell_volume)

        #out *= self.domain[0].cell_volume
        #print(self.fourier.domain.cell_volume)
        out *= self.fourier.domain.cell_volume

    @property
    def adjoint(self):
        op = self

        class RealFourierTransformAdjoint(odl.Operator):

            def __init__(self, op):
                """TBC

                Parameters
                ----------
                TBC

                Examples
                --------

               import odl
               import numpy as np
               from dTV.Dynamic_MRI.Real_Fourier_op_stacked import RealFourierTransformStacked
               complex_space = odl.uniform_discr(min_pt=[-1., -1., -1.], max_pt=[1., 1., 1.], shape=[4, 100, 100], dtype='complex')
               X = complex_space.real_space ** 2
               #mask = np.zeros((3, 10, 10))
               #mask[0,::2, :] = 1
               #mask[1, 1::2, :] = 1
               #mask[2, :, :] = 1
               #A = RealFourierTransformStacked(X, mask=mask)
               A = RealFourierTransformStacked(X)
               #x = odl.phantom.white_noise(A.domain)
               #y = odl.phantom.white_noise(A.range)
               x = A.domain.one()
               y = A.range.one()
               t1 = A(x).inner(y)
               t2 = x.inner(A.adjoint(y))
               print(t1 / t2)
               print(A.domain)
               print(A.range)

                """
                self.op = op

                super(RealFourierTransformAdjoint, self).__init__(
                    domain=op.range, range=op.domain, linear=True)

            def _call(self, x, out):
                for i in range(self.domain.shape[1]):
                    y = x[0, i].asarray() + 1j * x[1, i].asarray()

                    if not isinstance(self.op.mask, type(None)):
                        y = self.op.mask[i] * y

                    Fadjy = self.op.fourier.adjoint(y)

                    out[0, i][:] = np.fft.ifftshift(np.real(Fadjy))
                    out[1, i][:] = np.fft.ifftshift(np.imag(Fadjy))

                print(self.op.fourier.domain.size)
                out *= self.op.fourier.domain.size

            @property
            def adjoint(self):
                return op

        return RealFourierTransformAdjoint(op)

    @property
    def inverse(self):
        op = self

        class RealFourierTransformInverse(odl.Operator):

            def __init__(self, op):
                """TBC

                Parameters
                ----------
                TBC

                Examples
                --------

                """
                self.op = op

                super(RealFourierTransformInverse, self).__init__(
                    domain=op.range, range=op.domain, linear=True)

            def _call(self, x, out):

                assert self.mask is not None, "masked Fourier transform is not invertible"

                for i in range(self.domain.shape[1]):

                    y = x[0, i].asarray() + 1j * x[1, i].asarray()
                    Fadjy = self.op.fourier.inverse(y)
                    out[0, i][:] = np.fft.ifftshift(np.real(Fadjy))
                    out[1, i][:] = np.fft.ifftshift(np.imag(Fadjy))

                out /= self.op.fourier.domain.cell_volume

            @property
            def inverse(self):
                return op

        return RealFourierTransformInverse(op)


## testing

'''
from myOperators import RealFourierTransform
import matplotlib.pyplot as plt

complex_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.], shape=[100, 100], dtype='complex')
image_space = complex_space.real_space**2
phantom = odl.phantom.transmission.shepp_logan(image_space[0], modified=True)
complex_im_1 = image_space.element([phantom, image_space[1].zero()])
complex_im_2 = -2*complex_im_1

single_fourier_op = RealFourierTransform(image_space)
fourier_1 = single_fourier_op(complex_im_1)
fourier_2 = single_fourier_op(complex_im_2)
fourier_real_stacked = [fourier_1.asarray()[0], fourier_2.asarray()[0]]
fourier_imag_stacked = [fourier_1.asarray()[1], fourier_2.asarray()[1]]

space_time_complex = odl.uniform_discr(min_pt=[-1., -1., -1.], max_pt=[1., 1., 1.], shape=[2, 100, 100], dtype='complex')
recon_space = space_time_complex.real_space ** 2
fourier_op_stacked = RealFourierTransformStacked(recon_space)

fourier_stacked = fourier_op_stacked.range.element([fourier_real_stacked, fourier_imag_stacked])
fourier_inverse_stacked = fourier_op_stacked.inverse(fourier_stacked)
fourier_recon_1_real = fourier_inverse_stacked.asarray()[0, 0]
fourier_recon_1_imag = fourier_inverse_stacked.asarray()[1, 0]
fourier_recon_2_real = fourier_inverse_stacked.asarray()[0, 1]
fourier_recon_2_imag = fourier_inverse_stacked.asarray()[1, 1]

plt.figure()
plt.imshow(phantom.asarray(), cmap=plt.cm.gray)

plt.figure()
plt.imshow(fourier_recon_1_real, cmap=plt.cm.gray)

plt.figure()
plt.imshow(fourier_recon_1_imag, cmap=plt.cm.gray)

plt.figure()
plt.imshow(-phantom.asarray(), cmap=plt.cm.gray)

plt.figure()
plt.imshow(fourier_recon_2_real, cmap=plt.cm.gray)

plt.figure()
plt.imshow(fourier_recon_2_imag, cmap=plt.cm.gray)
'''