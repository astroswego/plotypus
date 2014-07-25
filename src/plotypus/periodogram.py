import numpy
from scipy.signal import lombscargle
from math import modf

__all__ = [
    'find_period',
    'LombScargle',
    'rephase',
    'get_phase'
]

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

def conditional_entropy(data, min_period=0.2, max_period=5.0001,
                        precision=0.0001, xbins=10, ybins=5):
    return periods[numpy.argmin(CE(period, data, xbins=xbins, ybins=ybins)
        for period in numpy.arange(min_period, max_period, precision)]

def CE(period, data, xbins=10, ybins=5):
    if period <= 0: return numpy.PINF
    r = rephase(data, period)
    r.T[1] = (r.T[1]-numpy.min(r.T[1]))/(numpy.max(r.T[1])-numpy.min(r.T[1]))
    bins, *_ = numpy.histogram2d(r.T[0],r.T[1],[xbins,ybins],[[0,1],[0,1]])
    size = len(r.T[1])
    return numpy.sum((lambda p: p * numpy.log(numpy.sum(bins[i,:]) / size / p) \
                             if p > 0 else 0)(bins[i][j] / size)
                  for i in numpy.arange(0, xbins)
                  for j in numpy.arange(0, ybins)) if size > 0 else numpy.PINF

def rephase(data, period=1, col=0):
    rephased = numpy.ma.copy(data)
    rephased[:, col] = get_phase(rephased[:, col], period)
    return rephased

def get_phase(time, period=1, offset=0):
    return (time / period - offset)%1
