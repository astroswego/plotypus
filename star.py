import os
import numpy
import matplotlib.pyplot as plt
import interpolation
from scipy.signal import lombscargle
from math import modf
from re import split
from utils import raw_string, get_masked, get_unmasked
from scale import normalize

class Star:
    def __init__(self, name, period, rephased, coefficients,
                       y_min=None, y_max=None):
        self.name = name
        self.period = period
        self.rephased = rephased
        self.coefficients = coefficients
        self.y_min = y_min
        self.y_max = y_max

def lightcurve(filename,
               interpolant = interpolation.least_squares_polynomial,
               evaluator = interpolation.polynomial_evaluator,
               degree = 10,
               min_period = 0.2,
               max_period = 32.,
               period_bins = 50000,
               sigma = 1,
               min_obs = 25,
               **options):
    """Returns a four-tuple containing 
    1) the name of the star, 
    2) the period from a Lomb-Scargle Periodogram, 
    3) the rephased observations, and 
    4) a list of interpolation coefficients, or 
    None if no adequate model can be found. 
    
    Searches for periods within the specified bounds discretized by the 
    specified number of period bins. 
    
    If sigma is greater than zero, then outliers will be sigma clipped with the 
    specified value. 
    """
    name = filename.split(os.sep)[-1]
    data = numpy.ma.masked_array(data=numpy.loadtxt(filename), mask=None)
    while True:
        if get_unmasked(data).shape[0] < min_obs:
            return None
        period = find_period(data, min_period, max_period, period_bins)
        if not min_period <= period <= max_period:
            return None
        rephased, y_min, y_max = rephase(data, period)
        coefficients = interpolant(rephased, degree)
        if sigma > 0:
            prev_mask = data.mask
#            print("hey")
            outliers = find_outliers(rephased, evaluator, coefficients, sigma)
#            print("ho")
            data.mask = numpy.ma.mask_or(data.mask, outliers)
#            print('o: {}, m: {}'.format(outliers.shape, data.shape))
            if numpy.all(data.mask == prev_mask):
                rephased.mask = data.mask
            else:
                continue
        return rephased is not None and Star(name, period, rephased,
                                             coefficients, y_min, y_max)

def find_period(data, min_period, max_period, period_bins):
    """Uses the Lomb-Scargle Periodogram to discover the period."""
    time, mags = data.T[0], data.T[1]
    scaled_mags = (mags-mags.mean())/mags.std()
    minf, maxf = 2*numpy.pi/max_period, 2*numpy.pi/min_period
    freqs = numpy.linspace(minf, maxf, period_bins)
    pgram = lombscargle(time, scaled_mags, freqs)
    return 2*numpy.pi/freqs[numpy.argmax(pgram)]

def rephase(data, period):
    """Non-destructively rephases all of the points in the given data set to
    be between 0 and 1 and shifted to max light."""
    max_light_phase = get_phase(max_light_time(data), period)
    rephased = numpy.ma.copy(data)
    for observation in rephased:
        observation[0] = get_phase(observation[0], period, max_light_phase)
    rephased.T[1], y_min, y_max = normalize(rephased.T[1])
    #print mean, std
    return rephased, y_min, y_max

def max_light_time(data):
    """Returns the time at which the star is at its brightest, i.e. the Julian
    Date of the smallest magnitude."""
    return data.T[0][data.T[1].argmin()]

def get_phase(time, period, offset=0):
    """Returns the phase associated with a given time based on the period."""
    return (modf(time/period)[0]-offset)%1

def find_outliers(rephased, evaluator, coefficients, sigma):
    """Finds rephased values that are too far from the light curve."""
    expected = evaluator(coefficients, rephased.T[0])
    actual, error = rephased.T[1], rephased.T[2]
    outliers = (expected-actual)**2 > sigma*actual.std()**2+error
    return numpy.tile(numpy.vstack(outliers), rephased.shape[1])

def pca(star_matrix):
    """Finds the eigenvalues and eigenvectors of the covariance matrix"""
    

    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(star_matrix))
    return eigvals, eigvecs
    
x = numpy.arange(0, 1.01, 0.01)#numpy.linspace(0, 0.99, 100)


def plot_lightcurves(star, evaluator, output, **options):
    plt.gca().grid(True)
    if options["plot_lightcurves_observed"]:
        plt.scatter(star.rephased.T[0], star.rephased.T[1])
        #plt.errorbar(rephased.T[0], rephased.T[1], rephased.T[2], ls='none')
        outliers = get_masked(star.rephased)
        plt.scatter(outliers.T[0], outliers.T[1], color='r')
    if options["plot_lightcurves_interpolated"]:
        #plt.errorbar(outliers.T[0], outliers.T[1], outliers.T[2], ls='none')
        plt.plot(x, evaluator(star.coefficients, x), linewidth=2.5)
    #plt.errorbar(x, options['evaluator'](x, coefficients), rephased.T[1].std())
    plt.xlabel('Period ({0:0.5} days)'.format(star.period))
#    plt.xlabel('Period (' + str(star.period)[:5] + ' days)')
    plt.ylabel('Magnitude')
    plt.title(star.name)
    plt.axis([0,1,1,0])
    out = split(raw_string(os.sep), star.name)[-1]+'.png'
    plt.savefig(os.path.join(output, out))
    plt.clf()

"""
def plot(star, evaluator, output, **options):
    plt.gca().grid(True)
    plt.scatter(star.T[0], star.T[1])
    out = split(raw_string(os.sep), str(star[0][0]))[-1]+'.png'
    plt.savefig(os.path.join(output, out))
    plt.clf()
"""
