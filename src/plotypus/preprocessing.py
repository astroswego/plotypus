import numpy

__all__ = [
    'Fourier',
    'trigonometric_coefficient_matrix'
]

class Fourier():
    def __init__(self, degree=3):
        self.degree = degree
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        data = numpy.array(list(zip(numpy.array(X).T[0], range(len(X)))))
        phase, order = data[data[:,0].argsort()].T
        coefficients = trigonometric_coefficient_matrix(phase, self.degree)
        return numpy.array([mag for (orig, mag) # Put back in original order
                            in sorted(zip(order, coefficients),
                                      key=lambda pair: pair[0])])
    
    def get_params(self, deep):
        return {'degree': self.degree}
    
    def set_params(self, **params):
        if 'degree' in params:
            self.degree = params['degree']

    def phase_shifted_coefficients(amplitude_coefficients):
        """Converts Fourier coefficients from the form
        m(t) = A_0 + \Sum_{k=1}^n a_k \sin(k \omega t)
                                + b_k \cos(k \omega t)
        into the form
        m(t) = A_0 + \Sum_{k=1}^n A_k \sin(k \omega t + \Phi_k)
        """
        A_0 = amplitude_coefficients[0]
        a_k = amplitude_coefficients[1::2]
        b_k = amplitude_coefficients[2::2]

        A_k = numpy.sqrt(a_k**2 + b_k**2)
        Phi_k = numpy.arccos(a_k/A_k) + numpy.pi/2

        # This should start out as a numpy zeros array, not a list
        phase_shifted_coefficients_ = numpy.zeros(amplitude_coefficients.shape)
        phase_shifted_coefficients_[0]    = A_0
        phase_shifted_coefficients_[1::2] = A_k
        phase_shifted_coefficients_[2::2] = Phi_k

        # If this began as a zeros array, it wouldn't have to be reshaped
        return phase_shifted_coefficients_

def trigonometric_coefficient_matrix(phases, degree):
    return numpy.array([[1 if j == 0
                         else numpy.cos(numpy.pi*(j+1)*phases[i]) if j % 2
                         else numpy.sin(numpy.pi*j*phases[i])
                        for j in range(2*degree+1)]
                       for i in range(len(phases))])
