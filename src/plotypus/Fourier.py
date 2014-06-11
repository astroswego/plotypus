import numpy

class Fourier():
    def __init__(self, degree=3):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = numpy.array(zip(numpy.array(X).T[0], range(len(X))))
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

def trigonometric_coefficient_matrix(phases, degree):
    return numpy.array([[numpy.cos((j+1)*numpy.pi*phases[i]) if j % 2
                         else numpy.sin(j*numpy.pi*phases[i])
                         for j in range(2*degree)]
                        for i in range(len(phases))])
