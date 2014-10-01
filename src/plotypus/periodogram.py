import numpy as np
from scipy.signal import lombscargle
from math import modf
from multiprocessing import Pool
from functools import partial

__all__ = [
    'find_period',
    'Lomb_Scargle',
    'rephase',
    'get_phase'
]

def find_period(data, min_period, max_period,
                coarse_precision, fine_precision,
                periodogram='Lomb_Scargle'):
    if min_period >= max_period: return min_period
    
    if periodogram == 'Lomb_Scargle':
      method = Lomb_Scargle
    else:
      method = conditional_entropy
    
    coarse_period = method(data, coarse_precision, min_period, max_period)
    if coarse_precision <= fine_precision: return coarse_period
    return method(data, fine_precision,
                  coarse_period - coarse_precision,
                  coarse_period + coarse_precision)

def Lomb_Scargle(data, precision, min_period, max_period):
    time, mags, *e = data
    scaled_mags = (mags-mags.mean())/mags.std()
    minf, maxf = 2*np.pi/max_period, 2*np.pi/min_period
    freqs = np.arange(minf, maxf, precision)
    pgram = lombscargle(time, scaled_mags, freqs)
    return 2*np.pi/freqs[np.argmax(pgram)]

def conditional_entropy(data, precision, min_period, max_period,
                        xbins=10, ybins=5):
    periods = np.arange(min_period, max_period, precision)
    print(periods)
    data.T[1] = (data.T[1] - np.min(data.T[1])) \
      / (np.max(data.T[1]) - np.min(data.T[1]))
    entropies = list(map(partial(CE, data=data, xbins=xbins, ybins=ybins),
                         periods))
    """np.savetxt(os.path.join(out, name_period),
                np.dstack((periods, entropies))[0],
                fmt='%s')"""
    print(entropies)
    print("period: ", str(periods[np.argmin(entropies)]))
    return periods[np.argmin(entropies)]

def CE(period, data, xbins=10, ybins=5):
    if period <= 0: return np.PINF
    r = rephase(data, period)
    bins, *_ = np.histogram2d(r.T[0], r.T[1], [xbins, ybins], [[0,1], [0,1]])
    size = len(r.T[1])
    return np.sum((lambda p: p * np.log(np.sum(bins[i,:]) / size / p) \
                             if p > 0 else 0)(bins[i][j] / size)
                  for i in np.arange(0, xbins)
                  for j in np.arange(0, ybins)) if size > 0 else np.PINF

def rephase(data, period=1, shift=0, col=0):
    rephased = np.ma.copy(data)
    rephased[:, col] = get_phase(rephased[:, col], period, shift)
    return rephased

def get_phase(time, period=1, shift=0):
    return (time / period - shift)%1
