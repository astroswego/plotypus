"""
Period finding and rephasing functions.
"""

import numpy as np
from scipy.signal import lombscargle
from multiprocessing import Pool
from functools import partial
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

__all__ = [
    'find_period',
    'conditional_entropy',
    'CE',
    'Lomb_Scargle',
    'rephase',
    'get_phase'
    'plot_periodogram',
    'plot_periodogram_mpl',
    'plot_periodogram_tikz'
]


def Lomb_Scargle(data, precision, min_period, max_period,
                 period_jobs=1, output_periodogram=False):
    """Lomb_Scargle(data, precision, min_period, max_period, period_jobs=1, output_periodogram=False)

    Returns the period of *data* according to the
    `Lomb-Scargle periodogram <https://en.wikipedia.org/wiki/Least-squares_spectral_analysis#The_Lomb.E2.80.93Scargle_periodogram>`_.
    Optionally outputs a tuple of the period and the periodogram itself.

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
    output_periodogram : bool, optional
        Whether or not to output the periodogram as well. If true, output
        is a tuple instead of a single number (default False).

    **Returns**

    period : number
        The period of *data*.
    periodogram : None or array, shape = [(max_period-min_period)/precision, 2]
        Array with columns (period, pgram), or None if *output_periodogram* is
        False.
    """
    time, mags, *err = data.T
    scaled_mags = (mags-mags.mean())/mags.std()
    minf, maxf = 2*np.pi/max_period, 2*np.pi/min_period
    freqs = np.arange(minf, maxf, precision)
    pgram = lombscargle(time, scaled_mags, freqs)

    if output_periodogram:
        periods = 2*np.pi/freqs
        period = periods[np.argmax(pgram)]
        return period, np.column_stack((periods, pgram))
    else:
        freq = freqs[np.argmax(pgram)]
        period = 2*np.pi/freq
        return period, None


def conditional_entropy(data, precision, min_period, max_period,
                        xbins=10, ybins=5, period_jobs=1,
                        output_periodogram=False):
    """conditional_entropy(data, precision, min_period, max_period, xbins=10, ybins=5, period_jobs=1, output_periodogram=False)

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
    output_periodogram : bool, optional
        Whether or not to output the periodogram as well. If true, output
        is a tuple instead of a single number (default False).

    **Returns**

    period : number
        The period of *data*.
    periodogram : None or array, shape = [(max_period-min_period)/precision, 2]
        Array with columns (period, pgram), or None if *output_periodogram* is
        False.

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

    period = periods[np.argmin(entropies)]

    if output_periodogram:
        return period, np.column_stack((periods, entropies))
    else:
        return period, None


def CE(period, data, xbins=10, ybins=5):
    """CE(period, data, xbins=10, ybins=5)

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
        column_sums = np.repeat(np.reshape(column_sums, (xbins,1)), ybins, axis=1)
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
                period_jobs=1,
                output_periodogram=False):
    """find_period(data, min_period=0.2, max_period=32.0, coarse_precision=1e-5, fine_precision=1e-9, periodogram=Lomb_Scargle, period_jobs=1, output_periodogram=False)

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
    output_periodogram : bool, optional
        Whether or not to output the periodogram as well. If true, output
        is a tuple instead of a single number (default False).

    **Returns**

    period : number
        The period of *data*.
    periodogram : None or array, shape = [(max_period-min_period)/precision, 2]
        Array with columns (period, pgram), or None if *output_periodogram* is
        False.
    """
    if min_period >= max_period:
        return min_period

    coarse_period, coarse_pgram = periodogram(
        data, coarse_precision,
        min_period, max_period,
        period_jobs=period_jobs,
        output_periodogram=output_periodogram)

    if coarse_precision <= fine_precision:
        return coarse_period, coarse_pgram
    else:
        fine_period, fine_pgram = periodogram(
            data, fine_precision,
            coarse_period - coarse_precision,
            coarse_period + coarse_precision,
            period_jobs=period_jobs,
            output_periodogram=output_periodogram)
        if output_periodogram:
            index = (coarse_period - min_period) / coarse_precision
            merged_pgram = np.concatenate([coarse_pgram[:index, :],
                                           fine_pgram,
                                           coarse_pgram[index+1:, :]])

            return fine_period, merged_pgram
        else:
            return fine_period, None


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


def plot_periodogram(*args, engine='mpl', **kwargs):
    """plot_periodogram(*args, engine='mpl', **kwargs)

    **Parameters**

    engine : str, optional
        Engine to use for plotting, choices are "mpl" and "tikz"
        (default "mpl")

    kwargs :
        See :func:`plot_periodogram_mpl` and :func:`plot_periodogram_tikz`,
        depending on *engine* specified.

    **Returns**

    plot : object
        Plot object. Type depends on *engine* used. "mpl" engine returns a
        `matplotlib.pyplot.Figure` object, and "tikz" engine returns a `str`.
    """
    if engine == "mpl":
        return(plot_periodogram_mpl(*args, **kwargs))
    elif engine == "tikz":
        return(plot_periodogram_tikz(*args, **kwargs))
    else:
        raise KeyError("engine '{}' does not exist".format(engine))


def plot_periodogram_mpl(name, periodogram, period=None,
                         form="frequency",
                         output=None,
                         sanitize_latex=False, color=True,
                         **kwargs):
    """plot_periodogram_mpl(name, periodogram, period, form="frequency", output=None, legend=False, sanitize_latex=False, color=True, **kwargs)

    Save a plot of the given *periodogram* to file *output*, using
    matplotlib and return the resulting plot object.

    **Parameters**

    name : str
        Name of the star. Used in plot title.
    periodogram : array-like, shape = [n_periods, 2]
        The periodogram, containing columns: periods/frequencies, pgram
    period : number (optional)
        The optimal period, to display on the plot.
    form : str (optional)
        Form of the periodogram, "frequency" or "period" (default "frequency")
    output : str, optional
        File to save plot to (default None).
    color : boolean, optional
        Whether or not to display color in plot (default True).

    **Returns**

    plot : matplotlib.pyplot.Figure
        Matplotlib Figure object which contains the plot.
    """
    # initialize Figure and Axes objects
    fig, ax = plt.subplots()

    periods, pgram = periodogram[periodogram[:,0].argsort()].T
    

    # display vertical line for chosen period, if given
    if period is not None:
        ax.axvline(period, color="red", ls='--', zorder=2)
    # plot the periodogram
    ax.plot(periods, pgram, 'k-', zorder=1)
    
    ax.set_xlim(0, max(periods))
    ax.set_ylim(min(0, min(pgram)), max(pgram)+max(pgram)*0.05)

    ax.set_xlabel('Period (days)' if form == 'period' else 'Frequency (1/d)')
    ax.set_ylabel('Power')
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_title(utils.sanitize_latex(name) if sanitize_latex else name)
    fig.tight_layout(pad=0.1)

    if output is not None:
        fig.savefig(output)

    return fig


def plot_periodogram_tikz(name, residuals, period=None,
                          form="frequency",
                          output=None, sanitize_latex=False,
                          color=True,
                          **kwargs):
    """plot_periodogram_tikz(name, residuals, period=None, form="frequency", output=None, sanitize_latex=False, color=True, **kwargs)

    Save TikZ source code for a plot of the given *residuals* to directory
    *output*, and return the string holding the source code.

    **Parameters**

    name : str
        Name of the star. Used in plot title.
    residuals : array-like, shape = [n_samples]
        Residuals between fitted lightcurve and observations.
    output : str, optional
        File to save plot to (default None).
    color : boolean, optional
        Whether or not to display color in plot (default True).

    **Returns**

    plot : matplotlib.pyplot.Figure
        Matplotlib Figure object which contains the plot.
    """
    return ""
