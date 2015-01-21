import numpy
from numpy import cos, pi
import numpy.testing as npt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from plotypus.lightcurve import get_lightcurve, make_predictor
from plotypus.periodogram import rephase
from plotypus.preprocessing import Fourier

def simulated_lc(X, shift):
    return 10 + cos(2*pi*X+shift) + 0.1*cos(18*pi*X+2*shift)

def phase_shifted_reconstruction(X, coeffs, form):
    A_0    = coeffs[0]
    A_ks   = coeffs[1::2]
    Phi_ks = coeffs[2::2]

    N, n = X.size, A_ks.size
    
    b = numpy.empty((N, n))
    b.T[:,:] = 2*numpy.pi*X
    b *= numpy.arange(1, n+1)
    b += Phi_ks

    if form == 'cos':
        B = numpy.cos(b)
    elif form == 'sin':
        B = numpy.sin(b)
    y = A_0 + numpy.dot(B, A_ks)

    return y

def main(shift=pi/2):
    N = 1000
    data = numpy.empty((3,N))
    phase, mag, err = data

    phase[:] = numpy.random.uniform(size=N)
    mag[:]   = simulated_lc(phase, shift)
    err[:]   = numpy.random.normal(0.0, 0.1, N)

    data = numpy.ma.array(data=data, mask=None, dtype=float, copy=False).T
    phases = numpy.arange(0, 1, 0.01)

    results = get_lightcurve(data, period=1.0, phases=phases, copy=True,
        predictor=make_predictor(LinearRegression(fit_intercept=False),
                                 use_baart=True))

    shift = results['shift']
    amp_coefficients = results['coefficients']
    cos_coefficients = Fourier.phase_shifted_coefficients(amp_coefficients,
                                                          form='cos',
                                                          shift=shift)
    sin_coefficients = Fourier.phase_shifted_coefficients(amp_coefficients,
                                                          form='sin',
                                                          shift=shift)

    data_ = rephase(data, shift=shift)
    data_ = data_[data[:,0].argsort()]
    phase_, mag_, err_ = data_.T

    cos_lc = phase_shifted_reconstruction(phase_, cos_coefficients, form='cos')
    sin_lc = phase_shifted_reconstruction(phase_, sin_coefficients, form='sin')

    plt.scatter(phase, mag, marker='o',
                edgecolors='b', facecolors='none')

    plt.scatter(phase_, mag_, marker='.', color='k')
    plt.plot(phase_, cos_lc, 'r--')
    plt.plot(phase_, sin_lc, 'g-.')

    plt.xlim([0,1])

    plt.savefig('phase_shifted_reconstruction.png')

    npt.assert_almost_equal(sin_lc, cos_lc, decimal=7)
    npt.assert_almost_equal(sin_lc, mag_, decimal=7)

if __name__ == '__main__':
    exit(main())
