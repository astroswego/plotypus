import collections
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from .utils import autocorrelation, rowvec
from sys import stderr

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
        data = numpy.array(list(zip(numpy.array(X).T[0], range(len(X)))))
        phase, order = data[data[:,0].argsort()].T
        coefficients = self.trigonometric_coefficient_matrix(phase, self.degree)
        return numpy.array([mag for (orig, mag) # Put back in original order
                            in sorted(zip(order, coefficients),
                                      key=lambda pair: pair[0])])
    
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
        print("cutoff = {}".format(cutoff))
        for degree in range(min_degree, max_degree):
            pipeline.set_params(Fourier__degree=degree)
            pipeline.fit(X, y)
            lc = pipeline.predict(sorted_X)
            residuals = y[X_sorting] - lc
            p_c = autocorrelation(residuals)
            print("degree = {}; p_c = {}".format(degree, p_c))
            if abs(p_c) <= cutoff:
                return degree
        # reached max_degree without reaching cutoff
        return max_degree

    @staticmethod
    def baart_tolerance(X):
        return (2 * (len(X) - 1))**(-1/2)

    @staticmethod
    def trigonometric_coefficient_matrix(phases, degree):
        return numpy.array([
            [1 if j == 0
             else numpy.cos(numpy.pi*(j+1)*phases[i]) if j % 2
             else numpy.sin(numpy.pi*j*phases[i])
             for j in range(2*degree+1)]
            for i in range(len(phases))])

    @staticmethod
    def phase_shifted_coefficients(amplitude_coefficients):
        """Converts Fourier coefficients from the form
        m(t) = A_0 + \Sum_{k=1}^n a_k \sin(k \omega t)
                                + b_k \cos(k \omega t)
        into the form
        m(t) = A_0 + \Sum_{k=1}^n A_k \sin(k \omega t + \Phi_k)
        """
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

        phase_shifted_coefficients_ = numpy.zeros(amplitude_coefficients.shape)
        phase_shifted_coefficients_[0]    = A_0
        phase_shifted_coefficients_[1::2] = A_k
        phase_shifted_coefficients_[2::2] = Phi_k

        return phase_shifted_coefficients_
