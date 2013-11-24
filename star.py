import os
import numpy
import math
import matplotlib
matplotlib.use("Agg") # Uses Agg backend
import matplotlib.pyplot as plt
import interpolation
from scipy.signal import lombscargle
from math import modf
from re import split
from utils import raw_string, get_noise, get_signal, make_sure_path_exists

class Star:
    __slots__ = ['name', 'period', 'rephased', 'coefficients', 'PCA']
    
    def __init__(self, name, period, rephased, coefficients):
        self.name = name
        self.period = period
        self.rephased = rephased
        self.coefficients = coefficients

def lightcurve(filename, min_obs=25, min_period=0.2, max_period=32.,
               coarse_precision=0.001, fine_precision=0.0000001,
               interpolant=interpolation.trigonometric,
               evaluator=interpolation.trigonometric_evaluator,
               min_degree=4, max_degree=15, sigma=3, **options):
    """Takes as input the filename for a data file containing the time,
    magnitude, and error for observations of a variable star. Uses a Lomb-
    Scargle periodogram to detect periodicities in the data within the
    specified bounds (discretized by the specified number of period bins).
    Rephases observations based on the star's primary period and normalizes the
    time domain to unit length. Creates a model of the star's light curve using
    the specified interpolant and its corresponding evaluation function.
    Searches for the best order of fit within the specified range
    using the unit-lag auto-correlation subject to Baart's criterion. Rejects
    points with a residual greater than sigma times the standard deviation
    if sigma is positive. Returns a star object containing the name, period,
    preprocessed data, and parameters to the fitted model."""
    name = filename.split(os.sep)[-1]
    data = numpy.ma.masked_array(data=numpy.loadtxt(filename), mask=None)
    while True: # Iteratively process and find models of the data
        if get_signal(data).shape[0] < min_obs:
            print(name + " has too few observations - None")
            return None
        period = find_period(data, min_period, max_period, coarse_precision,
                             fine_precision)
        rephased = rephase(data, period)
        coefficients = find_model(get_signal(rephased), min_degree, max_degree,
                                  interpolant, evaluator)
        if sigma:
            prev_mask = data.mask
            outliers = find_outliers(rephased, evaluator, coefficients, sigma)
            data.mask = numpy.ma.mask_or(data.mask, outliers)
            if not numpy.all(data.mask == prev_mask): continue
            rephased.mask = data.mask
        coefficients = shift_to_max_light(rephased, interpolant, evaluator,
                                          coefficients)
        return rephased is not None and Star(name, period, rephased,
                                             coefficients)

def find_period(data, min_period, max_period, coarse_precision, fine_precision):
    """Uses the Lomb-Scargle Periodogram to discover the period."""
    if min_period >= max_period: return min_period
    time, mags = data.T[0:2]
    scaled_mags = (mags-mags.mean())/mags.std()
    coarse_period = periodogram(time, scaled_mags, coarse_precision,
                                min_period, max_period)
    if coarse_precision <= fine_precision: return coarse_period
    return periodogram(time, scaled_mags, fine_precision,
                       coarse_period - coarse_precision,
                       coarse_period + coarse_precision)

def periodogram(time, scaled_mags, precision, min_period, max_period):
    minf, maxf = 2*numpy.pi/max_period, 2*numpy.pi/min_period
    freqs = numpy.arange(minf, maxf, precision)
    pgram = lombscargle(time, scaled_mags, freqs)
    return 2*numpy.pi/freqs[numpy.argmax(pgram)]

def rephase(data, period=1, col=0):
    """Non-destructively rephases all of the values in the given column by the
    given period and scales to be between 0 and 1."""
    rephased = numpy.ma.copy(data)
    rephased.T[col] = [get_phase(x[col], period) for x in rephased]
    return rephased

def get_phase(time, period=1, offset=0):
    """Returns the phase associated with a given time based on the period."""
    return (modf(time/period)[0]-offset)%1

def find_model(signal, min_degree, max_degree, interpolant, evaluator):
    """Iterates through the degree space to find the model that best fits the
    data with the fewest parameters and best fit as measured by the unit-lag
    autocorrelation function subject to Baart's criterion."""
    if min_degree >= max_degree: return interpolant(signal, min_degree)
    cutoff = (2 * (signal.shape[0] - 1)) ** (-1/2) # Baart's tolerance
    p_values = [] # To hold (autocorrelation, coefficients) for each degree
    for degree in range(min_degree, max_degree+1):
        coefficients = interpolant(signal, degree)
        p_c = auto_correlation(signal, evaluator, coefficients)
        if p_c <= cutoff: # Baart's criterion satisfied 
            return coefficients
        p_values += [(p_c, coefficients)]
    ps, cs = list(zip(*p_values)) # If we run out of range,
    return cs[numpy.argmin(ps)]   # Return the model best we saw.

def auto_correlation(signal, evaluator, coefficients):
    """Calculates trends in the residuals between the data and its model."""
    sorted = signal[signal[:,0].argsort()]
    residuals = sorted.T[1] - evaluator(coefficients, sorted.T[0])
    mean = residuals.mean()
    indices = range(sorted.shape[0]-1)
    return (sum((residuals[i] - mean) * (residuals[i+1] - mean)
                for i in indices)
          / sum((residuals[i] - mean) ** 2
                for i in indices))

def find_outliers(rephased, evaluator, coefficients, sigma):
    """Finds rephased values that are too far from the light curve."""
    if sigma <= 0: return None
    phases, actual, errors = rephased.T
    residuals = abs(evaluator(coefficients, phases) - actual)
    mse = numpy.array([0 if residual < error else (residual - error)**2
                       for residual, error in zip(residuals, errors)])
    outliers = mse > sigma * mse.std()
    return numpy.tile(numpy.vstack(outliers), rephased.shape[1])

def shift_to_max_light(rephased, interpolant, evaluator, coefficients):
    """Destructively shifts the data and its model so that the brightest part
    of the cycle occurs at phase 0. Returns the coefficients for the new
    model."""
    phases = numpy.arange(0, 1, 0.01)
    model = evaluator(coefficients, phases)
    max_light = phases[model.argmin()]
    rephased.T[0] = [get_phase(phase, 1, max_light)
                     for phase in rephased.T[0].data]
    return interpolant(get_signal(rephased), (coefficients.shape[0]-1)//2)

x = numpy.arange(0, 2, 0.01)

def plot_lightcurves(star, evaluator, output, **options):
#    print("raw: {}\n\nPCA: {}".format(star.rephased.T[1],PCA))
    ax = plt.gca()
    ax.grid(True)
    ax.invert_yaxis()
    plt.xlim(0,2)
    if "plot_lightcurves_interpolated" in options:
        plt.plot(x, evaluator(star.coefficients, x), linewidth=1.5, color='r')
    if "plot_lightcurves_pca" in options:
        plt.plot(x, numpy.hstack((star.PCA,star.PCA)),
                 linewidth=1.5, color="yellow")
    if "plot_lightcurves_observed" in options:
        time, mags, err = star.rephased.T
        plt.errorbar(numpy.hstack((time,1+time)), numpy.hstack((mags, mags)),
              yerr = numpy.hstack((err,err)), ls='None', ms=.01, mew=.01)
        time, mags, err = get_noise(star.rephased).T#[0:2]
        plt.errorbar(numpy.hstack((time,1+time)), numpy.hstack((mags, mags)),
              yerr = numpy.hstack(
                  (err,err)), ls='None', ms=.01, mew=.01, color='r')
    plt.xlabel('Period ({0:0.7} days)'.format(star.period))
    plt.ylabel('Magnitude ({0}th order)'.format(
        (star.coefficients.shape[0]-1)//2))
    plt.title(star.name)
    out = os.path.join(output, split(raw_string(os.sep), star.name)[-1]+'.png')
    make_sure_path_exists(output)
    plt.savefig(out)
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
    parameters = numpy.vstack(tuple(
        interpolation.ak_bk2Ak_Phik(star.coefficients)[:,:7]
        for star in stars))
    (A0, A1, Phi1, A2, Phi2, A3, Phi3) = numpy.hsplit(parameters, 7)
    (R21, R31, R32) = (A2/A1, A3/A1, A3/A2)
    (Phi21, Phi31, Phi32) = (Phi2/Phi1, Phi3/Phi1, Phi3/Phi2)
    plot_parameter(logP, R21,   "R21",   output)
    plot_parameter(logP, R31,   "R31",   output)
    plot_parameter(logP, R32,   "R32",   output)
    plot_parameter(logP, Phi21, "Phi21", output)
    plot_parameter(logP, Phi31, "Phi31", output)
    plot_parameter(logP, Phi32, "Phi32", output)
