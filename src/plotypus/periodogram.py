"""
Period finding and rephasing functions.
"""
import numpy as np
from scipy.signal import lombscargle
from multiprocessing import Pool
from functools import partial


__all__ = [
    'find_period',
    'conditional_entropy',
    'CE',
    'Lomb_Scargle',
    'rephase',
    'get_phase'
]


def Lomb_Scargle(data, precision, min_period, max_period, period_jobs=1):
    """
    Returns the period of *data* according to the
    `Lomb-Scargle periodogram <https://en.wikipedia.org/wiki/Least-squares_spectral_analysis#The_Lomb.E2.80.93Scargle_periodogram>`_.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    precision : number
        Distance between contiguous frequencies in search-space.
    min_period : number
        Minimum period in search-space.
    max_period : number
        Maximum period in search-space.
    period_jobs : int, optional
        Number of simultaneous processes to use while searching. Only one
        process will ever be used, but argument is included to conform to
        *periodogram* standards of :func:`find_period` (default 1).

    **Returns**

    period : number
        The period of *data*.
    """
    time, mags, *err = data.T
    scaled_mags = (mags-mags.mean())/mags.std()
    minf, maxf = 2*np.pi/max_period, 2*np.pi/min_period
    freqs = np.arange(minf, maxf, precision)
    pgram = lombscargle(time, scaled_mags, freqs)

    return 2*np.pi/freqs[np.argmax(pgram)]


def conditional_entropy(data, precision, min_period, max_period,
                        xbins=10, ybins=5, period_jobs=1):
    """
    Returns the period of *data* by minimizing conditional entropy.
    See `link <http://arxiv.org/pdf/1306.6664v2.pdf>`_ [GDDMD] for details.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    precision : number
        Distance between contiguous frequencies in search-space.
    min_period : number
        Minimum period in search-space.
    max_period : number
        Maximum period in search-space.
    xbins : int, optional
        Number of phase bins for each trial period (default 10).
    ybins : int, optional
        Number of magnitude bins for each trial period (default 5).
    period_jobs : int, optional
        Number of simultaneous processes to use while searching. Only one
        process will ever be used, but argument is included to conform to
        *periodogram* standards of :func:`find_period` (default 1).

    **Returns**

    period : number
        The period of *data*.

    **Citations**

    .. [GDDMD] Graham, Matthew J. ; Drake, Andrew J. ; Djorgovski, S. G. ;
               Mahabal, Ashish A. ; Donalek, Ciro, 2013,
               Monthly Notices of the Royal Astronomical Society,
               Volume 434, Issue 3, p.2629-2635
    """
    periods = np.arange(min_period, max_period, precision)
    copy = np.ma.copy(data)
    copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
       / (np.max(copy[:,1]) - np.min(copy[:,1]))
    partial_job = partial(CE, data=copy, xbins=xbins, ybins=ybins)
    m = map if period_jobs <= 1 else Pool(period_jobs).map
    entropies = list(m(partial_job, periods))

    return periods[np.argmin(entropies)]


def CE(period, data, xbins=10, ybins=5):
    """
    Returns the conditional entropy of *data* rephased with *period*.

    **Parameters**

    period : number
        The period to rephase *data* by.
    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    xbins : int, optional
        Number of phase bins (default 10).
    ybins : int, optional
        Number of magnitude bins (default 5).
    """
    if period <= 0:
        return np.PINF

    r = rephase(data, period)
    bins, *_ = np.histogram2d(r[:,0], r[:,1], [xbins, ybins], [[0,1], [0,1]])
    size = r.shape[0]

# The following code was once more readable, but much slower.
# Here is what it used to be:
# -----------------------------------------------------------------------
#    return np.sum((lambda p: p * np.log(np.sum(bins[i,:]) / size / p) \
#                             if p > 0 else 0)(bins[i][j] / size)
#                  for i in np.arange(0, xbins)
#                  for j in np.arange(0, ybins)) if size > 0 else np.PINF
# -----------------------------------------------------------------------
# TODO: replace this comment with something that's not old code
    if size > 0:
        # bins[i,j] / size
        divided_bins = bins / size
        # indices where that is positive
        # to avoid division by zero
        arg_positive = divided_bins > 0

        # array containing the sums of each column in the bins array
        column_sums = np.sum(divided_bins, axis=1) #changed 0 by 1
        # array is repeated row-wise, so that it can be sliced by arg_positive
        column_sums = np.repeat(np.reshape(column_sums, (x_bins,1)), ybins, axis=1)
        #column_sums = np.repeat(np.reshape(column_sums, (1,-1)), xbins, axis=0)


        # select only the elements in both arrays which correspond to a
        # positive bin
        select_divided_bins = divided_bins[arg_positive]
        select_column_sums  = column_sums[arg_positive]

        # initialize the result array
        A = np.empty((xbins, ybins), dtype=float)
        # store at every index [i,j] in A which corresponds to a positive bin:
        # bins[i,j]/size * log(bins[i,:] / size / (bins[i,j]/size))
        A[ arg_positive] = select_divided_bins \
                         * np.log(select_column_sums / select_divided_bins)
        # store 0 at every index in A which corresponds to a non-positive bin
        A[~arg_positive] = 0

        # return the summation
        return np.sum(A)
    else:
        return np.PINF


def find_period(data,
                min_period=0.2, max_period=32.0,
                coarse_precision=1e-5, fine_precision=1e-9,
                periodogram=Lomb_Scargle,
                period_jobs=1):
    """find_period(data, min_period=0.2, max_period=32.0, coarse_precision=1e-5, fine_precision=1e-9, periodogram=Lomb_Scargle, period_jobs=1)

    Returns the period of *data* according to the given *periodogram*,
    searching first with a coarse precision, and then a fine precision.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    min_period : number
        Minimum period in search-space.
    max_period : number
        Maximum period in search-space.
    coarse_precision : number
        Distance between contiguous frequencies in search-space during first
        sweep.
    fine_precision : number
        Distance between contiguous frequencies in search-space during second
        sweep.
    periodogram : function
        A function with arguments *data*, *precision*, *min_period*,
        *max_period*, and *period_jobs*, and return value *period*.
    period_jobs : int, optional
        Number of simultaneous processes to use while searching (default 1).

    **Returns**

    period : number
        The period of *data*.
    """
    if min_period >= max_period:
        return min_period

    coarse_period = periodogram(data, coarse_precision, min_period, max_period,
                                period_jobs=period_jobs)

    return coarse_period if coarse_precision <= fine_precision else \
        periodogram(data, fine_precision,
                    coarse_period - coarse_precision,
                    coarse_period + coarse_precision,
                    period_jobs=period_jobs)


def rephase(data, period=1.0, shift=0.0, col=0, copy=True):
    """
    Returns *data* (or a copy) phased with *period*, and shifted by a
    phase-shift *shift*.

    **Parameters**

    data : array-like, shape = [n_samples, n_cols]
        Array containing the time or phase values to be rephased in column
        *col*.
    period : number, optional
        Period to phase *data* by (default 1.0).
    shift : number, optional
        Phase shift to apply to phases (default 0.0).
    col : int, optional
        Column in *data* containing the time or phase values to be rephased
        (default 0).
    copy : bool, optional
        If True, a new array is returned, otherwise *data* is rephased
        in-place (default True).

    **Returns**

    rephased : array-like, shape = [n_samples, n_cols]
        Array containing the rephased *data*.
    """
    rephased = np.ma.array(data, copy=copy)
    rephased[:, col] = get_phase(rephased[:, col], period, shift)

    return rephased


def get_phase(time, period=1.0, shift=0.0):
    """
    Returns *time* transformed to phase-space with *period*, after applying a
    phase-shift *shift*.

    **Parameters**

    time : array-like, shape = [n_samples]
        The times to transform.
    period : number, optional
        The period to phase by (default 1.0).
    shift : number, optional
        The phase-shift to apply to the phases (default 0.0).

    **Returns**

    phase : array-like, shape = [n_samples]
        *time* transformed into phase-space with *period*, after applying a
        phase-shift *shift*.
    """
    return (time / period - shift) % 1
