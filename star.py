import os
import numpy
import math
import matplotlib
matplotlib.use("Agg") # Uses Agg backend
import matplotlib.pyplot as plt
#import mdp # Use this for implementing PCA in python
import interpolation
from scipy.signal import lombscargle
from math import modf
from re import split
from utils import raw_string, get_masked, get_unmasked
from scale import normalize_single, standardize, unnormalize, unstandardize

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
            print("not in range - None")
            return None
        rephased, y_min, y_max = rephase(data, period)
        coefficients = interpolant(rephased, degree)
        if sigma > 0:
            prev_mask = data.mask
            outliers = find_outliers(rephased, evaluator, coefficients, sigma)
            data.mask = numpy.ma.mask_or(data.mask, outliers)
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
    rephased.T[1], y_min, y_max = normalize_single(rephased.T[1])
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

x = numpy.arange(0, 1.00, 0.01)#numpy.linspace(0, 0.99, 100)

def lightcurve_matrix(stars, evaluator, x=x):
    m = numpy.vstack(tuple(numpy.array(evaluator(s.coefficients, x))
                           for s in stars))
#    iterable = (evaluator(s.coefficients, x) for s in stars)
#    m = numpy.vstack(numpy.fromiter((evaluator(s.coefficients, x),numpy.float) for s in stars))

 #    m = numpy.vstack(tuple(numpy.fromiter(iter(evaluator(s.coefficients, x)),
 #                                          numpy.float)
 #                           for s in stars))
    return m

 # def principle_component_analysis(data, degree):
 #     standardized_data, data_mean, data_std = standardize(data)
 #     pcanode = mdp.nodes.PCANode(output_dim=degree)
 #     pcanode.train(standardized_data.T)
 #     pcanode.stop_training()
 #     eigenvectors = pcanode.execute(standardized_data.T)
 #     principle_scores = numpy.dot(standardized_data, eigenvectors)
 #     standardized_reconstruction_matrix = pca_reconstruction(eigenvectors,
 #                                                             principle_scores)
 #     reconstruction_matrix = unstandardize(standardized_reconstruction_matrix,
 #                                           data_mean, data_std)
 #     return eigenvectors, principle_scores, reconstruction_matrix

def pca_reconstruction(eigenvectors, principle_scores):
    """Returns an array in which each row contains the magnitudes of one star's
    lightcurve. eigenvectors is a (number of phases)x(order of PCA) array,
    principle_components is a (number or stars)x(order of PCA) array, and the
    return array has shape (number of stars)x(number of phases)."""
    return numpy.dot(eigenvectors, principle_scores.T).T    

def plot_lightcurves(star, evaluator, output, **options):
#    print("raw: {}\n\nPCA: {}".format(star.rephased.T[1],PCA))
    plt.gca().grid(True)
    if "plot_lightcurves_observed" in options:
        plt.scatter(star.rephased.T[0], star.rephased.T[1])
        #plt.errorbar(rephased.T[0], rephased.T[1], rephased.T[2], ls='none')
        outliers = get_masked(star.rephased)
        plt.scatter(outliers.T[0], outliers.T[1], color='r')
    if "plot_lightcurves_interpolated" in options:
        #plt.errorbar(outliers.T[0], outliers.T[1], outliers.T[2], ls='none')
        plt.plot(x, evaluator(star.coefficients, x), linewidth=2.5)
    if True:# "plot_lightcurves_pca" in options:
        plt.plot(x, star.PCA, linewidth=1.5, color="yellow")
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

def plot_parameter(logP, parameter, parameter_name, output):
    plt.gca().grid(True)
    plt.scatter(logP, parameter)
    plt.xlabel("logP")
    plt.ylabel(parameter_name)
    title = parameter_name + " vs logP"
    plt.title(title)
    out = title + ".png"
    plt.savefig(os.path.join(output, out))
    plt.clf()


def trig_param_plot(stars, output):
    logP = numpy.fromiter((math.log(star.period, 10) for star in stars),
                                    numpy.float)
    parameters = numpy.vstack(tuple(ak_bk2Ak_Phik(star.coefficients)
                                    for star in stars))
    (A0, A1, Phi1, A2, Phi2, A3, Phi3) = parameters[:,:7]
    (R21, R31, R32) = (A2/A1, A3/A1, A3/A2)
    (Phi21, Phi31, Phi32) = (Phi2/Phi1, Phi3/Phi1, Phi3/Phi2)
    plot_parameter(logP, R21, "R21", output)
    plot_parameter(logP, R31, "R31", output)
    plot_parameter(logP, R32, "R32", output)
    plot_parameter(logP, Phi21, "Phi21", output)
    plot_parameter(logP, Phi31, "Phi31", output)
    plot_parameter(logP, Phi32, "Phi32", output)
