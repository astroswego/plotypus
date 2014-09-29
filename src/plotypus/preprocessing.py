import collections
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from .utils import autocorrelation, rowvec

__all__ = [
    'Fourier'
]

class Fourier():
    def __init__(self, degree=3, degree_range=None,
                 regressor=LinearRegression()):
        self.degree = degree
        self.degree_range = degree_range
        self.regressor = regressor

    def fit(self, X, y=None):
        if self.degree_range is not None:
            self.degree = self.baart_criteria(X, y)
        return self

    def transform(self, X, y=None, **params):
        data = numpy.dstack((numpy.array(X).T[0], range(len(X))))[0]
        phase, order = data[data[:,0].argsort()].T
        coefficients = self.trigonometric_coefficient_matrix(phase, self.degree)
        return coefficients[order.argsort()]
    
    def get_params(self, deep):
        return {'degree': self.degree}
    
    def set_params(self, **params):
        if 'degree' in params:
            self.degree = params['degree']

    def baart_criteria(self, X, y):
        try:
            min_degree, max_degree = self.degree_range
        except Exception:
            raise Exception("Degree range must be a length two sequence")

        cutoff = self.baart_tolerance(X)
        pipeline = Pipeline([('Fourier', Fourier()),
                             ('Regressor', self.regressor)])
        sorted_X = numpy.sort(X, axis=0)
        X_sorting = numpy.argsort(rowvec(X))
        for degree in range(min_degree, max_degree):
            pipeline.set_params(Fourier__degree=degree)
            pipeline.fit(X, y)
            lc = pipeline.predict(sorted_X)
            residuals = y[X_sorting] - lc
            p_c = autocorrelation(residuals)
            if abs(p_c) <= cutoff:
                return degree
        # reached max_degree without reaching cutoff
        return max_degree

    @staticmethod
    def baart_tolerance(X):
        return (2 * (len(X) - 1))**(-1/2)

    @staticmethod
    def trigonometric_coefficient_matrix(phases, degree):
        """Constructs an Nx2n+1 matrix of the form:
        / 1 sin(1*2*pi*phase[0]) cos(1*2*pi*phase[0]) ... cos(n*2*pi*phase[0]) \
        | 1 sin(1*2*pi*phase[1]) cos(1*2*pi*phase[1]) ... cos(n*2*pi*phase[1]) |
        | .         .                    .            .             .          |
        | .         .                    .             .            .          |
        | .         .                    .              .           .          |
        \ 1 sin(1*2*pi*phase[N]) cos(1*2*pi*phase[N]) ... cos(n*2*pi*phase[N]) /
        """
        # initialize coefficient matrix
        M = numpy.empty((phases.size, 2*degree+1))
        # indices
        i = numpy.arange(1, degree+1)
        # initialize the Nxn matrix that is repeated within the
        # sine and cosine terms
        x = numpy.empty((phases.size, degree))
        # the Nxn matrix now has N copies of the same row, and each row is
        # integer multiples of pi counting from 1 to the degree
        x[:,:] = i*2*numpy.pi
        # multiply each row of x by the phases
        x.T[:,:] *= phases
        # place 1's in the first column of the coefficient matrix
        M[:,0]    = 1
        # the odd indices of the coefficient matrix have sine terms
        M[:,1::2] = numpy.sin(x)
        # the even indices of the coefficient matrix have cosine terms
        M[:,2::2] = numpy.cos(x)
        return M
    
    @staticmethod
    def phase_shifted_coefficients(amplitude_coefficients, form='cos'):
        """Converts Fourier coefficients from the form
        m(t) = A_0 + \Sum_{k=1}^n a_k \sin(k \omega t)
                                + b_k \cos(k \omega t)
        into the form
        m(t) = A_0 + \Sum_{k=1}^n A_k \sin(k \omega t + \Phi_k)
        """
        if 'cos' in form:
            pass # this will do something once sine series are supported
        elif 'sin' in form:
            raise Exception('Fourier sine series not yet supported')
        else:
            raise Exception('Fourier series must have form sine or cosine')
        
        # separate array of coefficients into respective parts
        A_0 = amplitude_coefficients[0]
        a_k = amplitude_coefficients[1::2]
        b_k = amplitude_coefficients[2::2]

        # determine which portions of the coefficients are in which of the
        # four cartesian quadrants. a_k is the x-axis, and b_k is the y-axis
        Q34 = b_k <  0
        Q14 = a_k >= 0
        Q23 = a_k <  0
        Q4  = numpy.logical_and(Q34, Q14)
        # A_k is simply the hypotenuse of the right triangle formed with
        # a_k and b_k as the opposite and adjacent
        A_k         = numpy.sqrt(a_k**2 + b_k**2)
        # since we are about to divide by b_k, we need to make sure there
        # are no zeroes. If b_k is zero, then a_k and A_k should also be zero,
        # so we just need to make b_k anything but zero
        b_k[b_k == 0] = 1.0
        # Phi_k needs to be shifted by pi in quadrants II and III,
        # and 2 pi in quadtrant IV
        Phi_k       = numpy.arctan(a_k/b_k)
        Phi_k[Q23] += numpy.pi
        Phi_k[Q4]  += 2.0*numpy.pi

        phase_shifted_coefficients_ = numpy.empty(amplitude_coefficients.shape)
        phase_shifted_coefficients_[0]    = A_0
        phase_shifted_coefficients_[1::2] = A_k
        phase_shifted_coefficients_[2::2] = Phi_k

        return phase_shifted_coefficients_

    @staticmethod
    def fourier_ratios(phase_shifted_coeffs, N):
        """Returns an array containing
        [ R_{N+1 N}, Phi_{N+1 N}, ..., R_{n N}, Phi_{n N} ],
        where R_{i j} is the amplitude ratio R_i / R_j,
        and Phi_{i j} is the phase delta Phi_i - Phi_j.
        """
        amplitudes = phase_shifted_coeffs[1::2]
        phases = phase_shifted_coeffs[2::2]

        # the number of ratios is 3 less than the number of coefficients
        # because the 0th coefficient is not used, and there are ratios of
        # two 
        ratios = numpy.empty(phase_shifted_coeffs.size-3, dtype=float)
        amplitude_ratios = ratios[::2]
        phase_deltas = ratios[1::2]
        
        amplitude_ratios[:] = amplitudes[1:]
        amplitude_ratios   /= amplitudes[0]

        phase_deltas[:] = phases[1:]
        phase_deltas   -= phases[0]

        return ratios
    
    # @staticmethod
    # def amplitude_ratios(amplitudes, N):
    #     """Returns an array containing
    #     [ R_{i N} for N < i < n ],
    #     where n is the degree of the fit.
    #     """
    #     return amplitudes[N+1:] / amplitudes[N]

    # @staticmethod
    # def phase_deltas(phase_shifts, N):
    #     """Returns an array containing
    #     [ Phi_{i N} for N < i < n ],
    #     where n is the degree of the fit.
    #     """
    #     return phase_shifts[N+1:] - phase_shifts[N]
