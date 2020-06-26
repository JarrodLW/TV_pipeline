"""
Taken from Image fusion project

"""

import numpy as np
import odl
from odl.operator import (Operator, IdentityOperator)
from scipy.ndimage import convolve as sp_convolve
from skimage.measure import block_reduce

###########################################
#################OPERATORS#################
###########################################


class RealFourierTransform(odl.Operator):

    def __init__(self, domain):
        """TBC

        Parameters
        ----------
        TBC

        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr(0, 1, 10) ** 2
        >>> F = myOperators.RealFourierTransform(X)
        >>> x = X.one()
        >>> y = F(x)
        """
        domain_complex = domain[0].complex_space
        self.fourier = odl.trafos.DiscreteFourierTransform(domain_complex)

        range = self.fourier.range.real_space ** 2

        super(RealFourierTransform, self).__init__(
            domain=domain, range=range, linear=True)

    def _call(self, x, out):
        Fx = self.fourier(x[0].asarray() + 1j * x[1].asarray())
        out[0][:] = np.real(Fx)
        out[1][:] = np.imag(Fx)

        out *= self.domain[0].cell_volume

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
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr(0, 2, 10) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2

                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2
                """
                self.op = op

                super(RealFourierTransformAdjoint, self).__init__(
                    domain=op.range, range=op.domain, linear=True)

            def _call(self, x, out):
                y = x[0].asarray() + 1j * x[1].asarray()
                Fadjy = self.op.fourier.adjoint(y)
                out[0][:] = np.real(Fadjy)
                out[1][:] = np.imag(Fadjy)

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
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr(0, 2, 10) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = A(x)
                >>> (A.inverse(y)-x).norm()

                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = A(x)
                >>> (A.inverse(y)-x).norm()
                """
                self.op = op

                super(RealFourierTransformInverse, self).__init__(
                    domain=op.range, range=op.domain, linear=True)

            def _call(self, x, out):
                y = x[0].asarray() + 1j * x[1].asarray()
                Fadjy = self.op.fourier.inverse(y)
                out[0][:] = np.real(Fadjy)
                out[1][:] = np.imag(Fadjy)

                out /= self.op.fourier.domain.cell_volume

            @property
            def inverse(self):
                return op

        return RealFourierTransformInverse(op)



class Complex2Real(odl.Operator):

    def __init__(self, domain):
        """TBC

        Parameters
        ----------
        TBC

        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.cn(3)
        >>> J = myOperators.Complex2Real(X)
        >>> x = X.one()
        >>> y = J(x)

        >>> import odl
        >>> import myOperators
        >>> X = odl.cn(3)
        >>> A = myOperators.Complex2Real(X)
        >>> x = odl.phantom.white_noise(A.domain)
        >>> y = odl.phantom.white_noise(A.range)
        >>> t1 = A(x).inner(y)
        >>> t2 = x.inner(A.adjoint(y))
        """

        super(Complex2Real, self).__init__(domain=domain,
                                           range=domain.real_space ** 2,
                                           linear=True)

    def _call(self, x, out):
        out[0][:] = np.real(x)
        out[1][:] = np.imag(x)

    @property
    def adjoint(self):
        return Real2Complex(self.range)


class Real2Complex(odl.Operator):

    def __init__(self, domain):
        """TBC

        Parameters
        ----------
        TBC

        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.rn(3) ** 2
        >>> J = myOperators.Real2Complex(X)
        >>> x = X.one()
        >>> y = J(x)
        """

        super(Real2Complex, self).__init__(domain=domain,
                                           range=domain[0].complex_space, linear=True)

    def _call(self, x, out):
        out[:] = x[0].asarray() + 1j * x[1].asarray()

    @property
    def adjoint(self):
        return Complex2Real(self.range)


