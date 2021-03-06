"""
Light curve fitting and plotting functions.
"""

import numpy
numpy.random.seed(0)
from scipy.stats import sem
from sys import stderr
from math import floor
from os import path
import plotypus.utils
from .utils import (verbose_print, make_sure_path_exists,
                    get_signal, get_noise, colvec, mad)
from .periodogram import find_period, Lomb_Scargle, rephase
from .preprocessing import Fourier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.utils import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import matplotlib
import matplotlib.pyplot as plt

__all__ = [
    'make_predictor',
    'get_lightcurve',
    'get_lightcurve_from_file',
    'find_outliers',
    'plot_lightcurve'
]


def make_predictor(regressor=LassoLarsIC(fit_intercept=False),
                   Selector=GridSearchCV, fourier_degree=(2, 25),
                   selector_processes=1,
                   use_baart=False, scoring='r2', scoring_cv=3,
                   **kwargs):
    """make_predictor(regressor=LassoLarsIC(fit_intercept=False), Selector=GridSearchCV, fourier_degree=(2, 25), selector_processes=1, use_baart=False, scoring='r2', scoring_cv=3, **kwargs)

    Makes a predictor object for use in :func:`get_lightcurve`.

    **Parameters**

    regressor : object with "fit" and "transform" methods, optional
        Regression object used for solving Fourier matrix
        (default ``sklearn.linear_model.LassoLarsIC(fit_intercept=False)``).
    Selector : class with "fit" and "predict" methods, optional
        Model selection class used for finding the best fit
        (default :class:`sklearn.grid_search.GridSearchCV`).
    selector_processes : positive integer, optional
        Number of processes to use for *Selector* (default 1).
    use_baart : boolean, optional
        If True, ignores *Selector* and uses Baart's Criteria to find
        the Fourier degree, within the boundaries (default False).
    fourier_degree : 2-tuple, optional
        Tuple containing lower and upper bounds on Fourier degree, in that
        order (default (2, 25)).
    scoring : str, optional
        Scoring method to use for *Selector*. This parameter can be:
            * "r2", in which case use :math:`R^2` (the default)
            * "mse", in which case use mean square error
    scoring_cv : positive integer, optional
        Number of cross validation folds used in scoring (default 3).

    **Returns**

    out : object with "fit" and "predict" methods
        The created predictor object.
    """
    fourier = Fourier(degree_range=fourier_degree, regressor=regressor) \
              if use_baart else Fourier()
    pipeline = Pipeline([('Fourier', fourier), ('Regressor', regressor)])
    if use_baart:
        return pipeline
    else:
        params = {'Fourier__degree': list(range(fourier_degree[0],
                                                fourier_degree[1]+1))}
        return Selector(pipeline, params, scoring=scoring, cv=scoring_cv,
                        n_jobs=selector_processes)


def get_lightcurve(data, copy=False, name=None,
                   predictor=None, periodogram=Lomb_Scargle,
                   sigma_clipping=mad,
                   scoring='r2', scoring_cv=3, scoring_processes=1,
                   period=None, min_period=0.2, max_period=32,
                   coarse_precision=1e-5, fine_precision=1e-9,
                   period_processes=1,
                   sigma=20,
                   shift=None,
                   min_phase_cover=0.0, min_observations=1, n_phases=100,
                   verbosity=None, **kwargs):
    """get_lightcurve(data, copy=False, name=None, predictor=None, periodogram=Lomb_Scargle, sigma_clipping=mad, scoring='r2', scoring_cv=3, scoring_processes=1, period=None, min_period=0.2, max_period=32, coarse_precision=1e-5, fine_precision=1e-9, period_processes=1, sigma=20, shift=None, min_phase_cover=0.0, n_phases=100, verbosity=None, **kwargs)

    Fits a light curve to the given `data` using the specified methods,
    with default behavior defined for all methods.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Photometry array with columns *time*, *magnitude*, and (optional)
        *error*. *time* should be unphased.
    name : string or None, optional
        Name of star being processed.
    predictor : object that has "fit" and "predict" methods, optional
        Object which fits the light curve obtained from *data* after rephasing
        (default ``make_predictor(scoring=scoring, scoring_cv=scoring_cv)``).
    periodogram : function, optional
        Function which finds one or more *period*\s. If *period* is already
        provided, the function is not used. Defaults to
        :func:`plotypus.periodogram.Lomb_Scargle`
    sigma_clipping : function, optional
        Function which takes an array and assigns sigma scores to each element.
        Defaults to :func:`plotypus.utils.mad`.
    scoring : str, optional
        Scoring method used by *predictor*. This parameter can be
            * "r2", in which case use :func:`R^2` (the default)
            * "mse", in which case use mean square error
    scoring_cv : positive integer, optional
        Number of cross validation folds used in scoring (default 3).
    scoring_processes : positive integer, optional
        Number of processes to use for scoring cross validation (default 1).
    period : number or None, optional
        Period of oscillation used in the fit. This parameter can be:
            * None, in which case the period is obtained with the given
              *periodogram* function (the default).
            * A single positive number, giving the period to phase *data*.
    min_period : non-negative number, optional
        Lower bound on period obtained by *periodogram* (default 0.2).
    max_period : non-negative number, optional
        Upper bound on period obtained by *periodogram* (default 32.0).
    course_precision : positive number, optional
        Precision used in first period search sweep (default 1e-5).
    fine_precision : positive number, optional
        Precision used in second period search sweep (default 1e-9).
    period_processes : positive integer, optional
        Number of processes to use for period finding (default 1).
    sigma : number, optional
        Upper bound on score obtained by *sigma_clipping* for a point to be
        considered an inlier.
    shift : number or None, optional
        Phase shift to apply to light curve if provided. Light curve is shifted
        such that max light occurs at ``phase[0]`` if None given (default None).
    min_phase_cover : number on interval [0, 1], optional
        Fraction of binned light curve that must contain points in order to
        proceed. If light curve has insufficient coverage, a warning is
        printed if "outlier" *verbosity* is on, and None is returned
        (default 0.0).
    n_phases : positive integer
        Number of equally spaced phases to predict magnitudes at (default 100)
    verbosity : list or None, optional
        Verbosity level. See :func:`plotypus.utils.verbose_print`.

    **Returns**

    out : dict
        Results of the fit in a dictionary. The keys are:
            * name : str or None
                The name of the star.
            * period : number
                The star's period.
            * lightcurve : array-like, shape = [n_phases]
                Magnitudes of fitted light curve sampled at sample phases.
            * coefficients : array-like, shape = [n_coeffs]
                Fitted light curve coefficients.
            * dA_0 : non-negative number
                Error on mean magnitude.
            * phased_data : array-like, shape = [n_samples]
                *data* transformed from temporal to phase space.
            * model : predictor object
                The predictor used to fit the light curve.
            * R2 : number
                The :math:`R^2` score of the fit.
            * MSE : number
                The mean square error of the fit.
            * degree : positive integer
                The degree of the Fourier fit.
            * shift : number
                The phase shift applied.
            * coverage : number on interval [0, 1]
                The light curve coverage.

    **See also**

    :func:`get_lightcurve_from_file`
    """
    data = numpy.ma.array(data, copy=copy)
    phases = numpy.linspace(0, 1, n_phases, endpoint=False)
# TODO ###
# Replace dA_0 with error matrix dA
    if predictor is None:
        predictor = make_predictor(scoring=scoring, scoring_cv=scoring_cv)

    while True:
        signal = get_signal(data)
        if len(signal) <= scoring_cv:
            verbose_print(
                "{}: length of signal ({}) less than cv folds ({})".format(
                    name, len(signal), scoring_cv),
                operation="coverage", verbosity=verbosity)
            return
        elif len(signal) < min_observations:
            verbose_print(
                "{}: length of signal ({}) "
                "less than min_observations ({})".format(
                    name, len(signal), min_observations),
                operation="coverage", verbosity=verbosity)
            return
        # Find the period of the inliers
        if period is not None:
            _period = period
        else:
            verbose_print("{}: finding period".format(name),
                          operation="period", verbosity=verbosity)
            _period = find_period(signal,
                                  min_period, max_period,
                                  coarse_precision, fine_precision,
                                  periodogram, period_processes)

        verbose_print("{}: using period {}".format(name, _period),
                      operation="period", verbosity=verbosity)
        phase, mag, *err = rephase(signal, _period).T

# TODO ###
# Generalize number of bins to function parameter ``coverage_bins``, which
# defaults to 100, the current hard-coded behavior
        # Determine whether there is sufficient phase coverage
        coverage = numpy.zeros((100))
        for p in phase:
            coverage[int(floor(p*100))] = 1
        coverage = sum(coverage)/100
        if coverage < min_phase_cover:
            verbose_print("{}: {} {}".format(name, coverage, min_phase_cover),
                          operation="coverage",
                          verbosity=verbosity)
            verbose_print("Insufficient phase coverage",
                          operation="outlier",
                          verbosity=verbosity)
            return

        # Predict light curve
        with warnings.catch_warnings(record=True) as w:
            try:
                predictor = predictor.fit(colvec(phase), mag)
            except Warning:
                # not sure if this should be only in verbose mode
                print(name, w, file=stderr)
                return

        # Reject outliers and repeat the process if there are any
        if sigma:
            outliers = find_outliers(rephase(data.data, _period), predictor,
                                     sigma, sigma_clipping)
            num_outliers = sum(outliers)[0]
            if num_outliers == 0 or \
               set.issubset(set(numpy.nonzero(outliers.T[0])[0]),
                            set(numpy.nonzero(data.mask.T[0])[0])):
                data.mask = outliers
                break
            if num_outliers > 0:
                verbose_print("{}: {} outliers".format(name, sum(outliers)[0]),
                              operation="outlier",
                              verbosity=verbosity)
            data.mask = numpy.ma.mask_or(data.mask, outliers)

    # Build light curve and optionally shift to max light
    lightcurve = predictor.predict([[i] for i in phases])
    if shift is None:
        arg_max_light = lightcurve.argmin()
        lightcurve = numpy.concatenate((lightcurve[arg_max_light:],
                                        lightcurve[:arg_max_light]))
        shift = arg_max_light/len(phases)

    data.T[0] = rephase(data.data, _period, shift).T[0]

    # Grab the coefficients from the model
    coefficients = predictor.named_steps['Regressor'].coef_ \
        if isinstance(predictor, Pipeline) \
        else predictor.best_estimator_.named_steps['Regressor'].coef_,

    # compute R^2 and MSE if they haven't already been
    # (one or zero have been computed, depending on the predictor)
    estimator = predictor.best_estimator_ \
        if hasattr(predictor, 'best_estimator_') \
        else predictor

    get_score = lambda scoring: predictor.best_score_ \
        if hasattr(predictor, 'best_score_') \
        and predictor.scoring == scoring \
        else cross_val_score(estimator, colvec(phase), mag,
                             cv=scoring_cv, scoring=scoring,
                             n_jobs=scoring_processes).mean()

    return {'name':         name,
            'period':       _period,
            'lightcurve':   lightcurve,
            'coefficients': coefficients[0],
            'dA_0':         sem(lightcurve),
            'phased_data':  data,
            'model':        predictor,
            'R2':           get_score('r2'),
            'MSE':          abs(get_score('mean_squared_error')),
            'degree':       estimator.get_params()['Fourier__degree'],
            'shift':        shift,
            'coverage':     coverage}


def get_lightcurve_from_file(file, *args, use_cols=None, skiprows=0,
                             verbosity=None,
                             **kwargs):
    """get_lightcurve_from_file(file, *args, use_cols=None, skiprows=0, **kwargs)

    Fits a light curve to the data contained in *file* using
    :func:`get_lightcurve`.

    **Parameters**

    file : str or file
        File or filename to load data from.
    use_cols : iterable or None, optional
        Iterable of columns to read from data file, or None to read all columns
        (default None).
    skiprows : number, optional
        Number of rows to skip at beginning of *file* (default 0)

    **Returns**

    out : dict
        See :func:`get_lightcurve`.
    """
    data = numpy.loadtxt(file, skiprows=skiprows, usecols=use_cols)
    if len(data) != 0:
        masked_data = numpy.ma.array(data=data, mask=None, dtype=float)
        return get_lightcurve(masked_data, *args,
                              verbosity=verbosity, **kwargs)
    else:
        verbose_print("{}: file contains no data points".format(file),
                      operation="coverage", verbosity=verbosity)
        return



## These functions were used briefly and then not maintained.
## Will make comebacks of some form in a later release.
##
# def get_lightcurves_from_file(filename, directories, *args, **kwargs):
#     return [get_lightcurve_from_file(path.join(d, filename), *args, **kwargs)
#             for d in directories]
#
#
# def single_periods(data, period, min_points=10, copy=False, *args, **kwargs):
#     data = numpy.ma.array(data, copy=copy)
#     time, mag, *err = data.T
#
#     tstart, tfinal = numpy.min(time), numpy.max(time)
#     periods = numpy.arange(tstart, tfinal+period, period)
#     data_range = (
#         data[numpy.logical_and(time>pstart, time<=pend),:]
#         for pstart, pend in zip(periods[:-1], periods[1:])
#     )
#
#     return (
#         get_lightcurve(d, period=period, *args, **kwargs)
#         for d in data_range
#         if d.shape[0] > min_points
#     )
#
#
# def single_periods_from_file(filename, *args, use_cols=(0, 1, 2), skiprows=0,
#                              **kwargs):
#     data = numpy.ma.array(data=numpy.loadtxt(filename, usecols=use_cols,
#                                              skiprows=skiprows),
#                           mask=None, dtype=float)
#     return single_periods(data, *args, **kwargs)


def find_outliers(data, predictor, sigma,
                  method=mad):
    """find_outliers(data, predictor, sigma, method=mad)

    Returns a boolean array indicating the outliers in the given *data* array.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Photometry array containing columns *phase*, *magnitude*, and
        (optional) *error*.
    predictor : object that has "fit" and "predict" methods, optional
        Object which fits the light curve obtained from *data* after rephasing.
    sigma : number
        Outlier cutoff criteria.
    method : function, optional
        Function to score residuals for outlier detection
        (default :func:`plotypus.utils.mad`).

    **Returns**

    out : array-like, shape = data.shape
        Boolean array indicating the outliers in the given *data* array.
    """
    phase, mag, *err = data.T
    residuals = numpy.absolute(predictor.predict(colvec(phase)) - mag)
    outliers = numpy.logical_and((residuals > err[0]) if err else True,
                                 residuals > sigma * method(residuals))

    return numpy.tile(numpy.vstack(outliers), data.shape[1])


def plot_lightcurve(name, lightcurve, period, data,
                    output='.', legend=False, sanitize_latex=False,
                    color=True, n_phases=100,
                    err_const=0.005,
                    **kwargs):
    """plot_lightcurve(name, lightcurve, period, data, output='.', legend=False, color=True, n_phases=100, err_const=0.005, **kwargs)

    Save a plot of the given *lightcurve* to directory *output*.

    **Parameters**

    name : str
        Name of the star. Used in filename and plot title.
    lightcurve : array-like, shape = [n_samples]
        Fitted lightcurve.
    period : number
        Period to phase time by.
    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Photometry array containing columns *time*, *magnitude*, and
        (optional) *error*. *time* should be unphased.
    output : str, optional
        Directory to save plot to (default '.').
    legend : boolean, optional
        Whether or not to display legend on plot (default False).
    color : boolean, optional
        Whether or not to display color in plot (default True).
    n_phases : integer, optional
        Number of phase points in fit (default 100).
    err_const : number, optional
        Constant to use in absence of error (default 0.005).

    **Returns**

    None
    """
    phases = numpy.linspace(0, 1, n_phases, endpoint=False)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlim(0,2)

    # Plot points used
    phase, mag, *err = get_signal(data).T

    error = err[0] if err else mag*err_const

    inliers = plt.errorbar(numpy.hstack((phase,1+phase)),
                           numpy.hstack((mag, mag)),
                           yerr=numpy.hstack((error, error)),
                           ls='None',
                           ms=.01, mew=.01, capsize=0)

    # Plot outliers rejected
    phase, mag, *err = get_noise(data).T

    error = err[0] if err else mag*err_const

    outliers = plt.errorbar(numpy.hstack((phase,1+phase)),
                            numpy.hstack((mag, mag)),
                            yerr=numpy.hstack((error, error)),
                            ls='None', marker='o' if color else 'x',
                            ms=.01 if color else 4,
                            mew=.01 if color else 1,
                            capsize=0 if color else 1)

    # Plot the fitted light curve
    signal, = plt.plot(numpy.hstack((phases,1+phases)),
                       numpy.hstack((lightcurve, lightcurve)),
                       linewidth=1)

    if legend:
        plt.legend([signal, inliers, outliers],
                   ["Light Curve", "Inliers", "Outliers"],
                   loc='best')

    plt.xlabel('Phase ({0:0.7} day period)'.format(period))
    plt.ylabel('Magnitude')

    plt.title(utils.sanitize_latex(name) if sanitize_latex else name)
    plt.tight_layout(pad=0.1)
    make_sure_path_exists(output)
    plt.savefig(path.join(output, name))
    plt.clf()
