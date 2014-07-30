import numpy
from numpy import cos, pi
import numpy.testing as npt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotypus.lightcurve import get_lightcurve
from plotypus.preprocessing import Fourier

def simulated_lc(X):
    return 10 + cos(2*pi*X) + 0.1*cos(18*pi*X)

def phase_shifted_reconstruction(X, coeffs):
    A_0    = coeffs[0]
    A_ks   = coeffs[1::2]
    Phi_ks = coeffs[2::2]

    N, n = X.size, A_ks.size
    
    b = numpy.empty((N, n))
    b.T[:,:] = 2*numpy.pi*X
    b *= numpy.arange(1, n+1)
    b += Phi_ks

    B = numpy.cos(b)
    y = A_0 + numpy.dot(B, A_ks)

    return y

def main():
    N = 1000
    data = numpy.empty((3,N))
    phase, mag, err = data

    phase[:] = numpy.random.uniform(size=N)
    mag[:]   = simulated_lc(phase)
    err[:]   = numpy.random.normal(0.0, 0.1, N)

    data = numpy.ma.array(data=data, mask=None, dtype=float, copy=False).T
    phases = numpy.arange(0, 1, 0.01)

    results = get_lightcurve(data, period=1.0, phases=phases)
    period, lc, data, amp_coefficients, R2, MSE, t_max, dA_0 = results

    phase_coefficients = Fourier.phase_shifted_coefficients(amp_coefficients)
    phase_shifted_lc = phase_shifted_reconstruction(phases,
                                                    phase_coefficients)
    arg_max_light = int(t_max * len(phases))
#    argmin = phase_shifted_lc.argmin()
    phase_shifted_lc = numpy.concatenate((phase_shifted_lc[arg_max_light:],
                                          phase_shifted_lc[:arg_max_light]))
    
    plt.plot(phase, mag, 'k.')
    plt.plot(phases, lc, 'g-')
    plt.plot(phases, phase_shifted_lc, 'r--')
    plt.savefig('phase_shifted_reconstruction.png')

    npt.assert_almost_equal(phase_shifted_lc, lc, decimal=7)
#    difference = numpy.abs(phase_shifted_lc - lc)
#    print('mean difference: {}, std difference: {}'.format(difference.mean(),
#                                                           difference.std()))

if __name__ == '__main__':
    exit(main())
