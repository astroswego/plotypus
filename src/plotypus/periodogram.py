import numpy
from scipy.signal import lombscargle
from math import modf

def find_period(time, mags, min_period, max_period,
                coarse_precision, fine_precision, method=None):
    if min_period >= max_period: return min_period
    scaled_mags = (mags-mags.mean())/mags.std()
    coarse_period = LombScargle(time, scaled_mags, coarse_precision,
                                min_period, max_period)
    if coarse_precision <= fine_precision: return coarse_period
    return LombScargle(time, scaled_mags, fine_precision,
                       coarse_period - coarse_precision,
                       coarse_period + coarse_precision)

def LombScargle(time, scaled_mags, precision, min_period, max_period):
    minf, maxf = 2*numpy.pi/max_period, 2*numpy.pi/min_period
    freqs = numpy.arange(minf, maxf, precision)
    pgram = lombscargle(time, scaled_mags, freqs)
    return 2*numpy.pi/freqs[numpy.argmax(pgram)]

def rephase(data, period=1, col=0):
    rephased = numpy.ma.copy(data)
    rephased.T[col] = [get_phase(x[col], period)
                       for x in rephased]
    return rephased

def get_phase(time, period=1, offset=0):
    return (modf(time/period)[0]-offset)%1
