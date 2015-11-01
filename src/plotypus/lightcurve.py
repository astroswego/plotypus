"""
Light curve fitting and plotting functions.
"""

import numpy as np
np.random.seed(0)
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

__all__ = [
    'make_predictor',
    'get_lightcurve',
    'get_lightcurve_from_file',
    'find_outliers',
    'savetxt_lightcurve',
    'plot_lightcurve',
    'plot_lightcurve_mpl',
    'plot_lightcurve_tikz',
    'plot_residual',
    'plot_residual_mpl',
    'plot_residual_tikz'
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
                   period_processes=1, output_periodogram=False,
                   sigma=20,
                   fourier_form="sin_cos",
                   shift=None,
                   min_phase_cover=0.0, min_observations=1, n_phases=100,
                   verbosity=None, **kwargs):
    """get_lightcurve(data, copy=False, name=None, predictor=None, periodogram=Lomb_Scargle, sigma_clipping=mad, scoring='r2', scoring_cv=3, scoring_processes=1, period=None, min_period=0.2, max_period=32, coarse_precision=1e-5, fine_precision=1e-9, period_processes=1, sigma=20, fourier_form="sin_cos", shift=None, min_phase_cover=0.0, n_phases=100, verbosity=None, **kwargs)

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
    fourier_form : str, optional
        Return coefficients for the Fourier series of the given form
        Form of coefficients in Fourier series
            * "sin_cos", a_k sin(kwt) + b_k cos(kwt) (the default)
            * "sin", A_k sin(kwt + Phi_k)
            * "cos", A_k cos(kwt + Phi_k)
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
            * coefficients : array-like, shape = [degree+1, 2]
                Fitted light curve coefficients.
            * coefficient_errors : array-like, shape = [degree+1, 2]
                Errors in coefficients
            * fourier_ratios : array-like, shape = [degree-1, 2]
                Ratios and phase deltas in Fourier coefficients,
                e.g. :math:`R_{21}` and :math:`\\Phi_{21}`
            * fourier_ratio_errors : array-like, shape = [degree+1, 2]
                Errors in fourier_ratios
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
    ################################
    ## Argument Validity Checking ##
    ################################
    # Fourier form is allowed
    _allowed_fourier_forms = {"sin_cos", "sin", "cos"}
    if fourier_form not in _allowed_fourier_forms:
        raise TypeError("Fourier form must be one of: {}"
                        .format(_allowed_fourier_forms))

    data = np.ma.array(data, copy=copy)
    phases = np.linspace(0, 1, n_phases, endpoint=False)
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
            period = period
            pgram  = None
        else:
            verbose_print("{}: finding period".format(name),
                          operation="period", verbosity=verbosity)
            period, pgram = find_period(signal,
                                        min_period, max_period,
                                        coarse_precision, fine_precision,
                                        periodogram, period_processes,
                                        output_periodogram=output_periodogram)

        verbose_print("{}: using period {}".format(name, period),
                      operation="period", verbosity=verbosity)
        phase, mag, *err = rephase(signal, period).T

# TODO ###
# Generalize number of bins to function parameter ``coverage_bins``, which
# defaults to 100, the current hard-coded behavior
        # Determine whether there is sufficient phase coverage
        coverage = np.zeros((100))
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
            outliers = find_outliers(rephase(data.data, period), predictor,
                                     sigma, sigma_clipping)
            num_outliers = sum(outliers)[0]
            if num_outliers == 0 or \
               set.issubset(set(np.nonzero(outliers.T[0])[0]),
                            set(np.nonzero(data.mask.T[0])[0])):
                data.mask = outliers
                break
            if num_outliers > 0:
                verbose_print("{}: {} outliers".format(name, sum(outliers)[0]),
                              operation="outlier",
                              verbosity=verbosity)
            data.mask = np.ma.mask_or(data.mask, outliers)

    # Build predicted light curve and residuals
    lightcurve = predictor.predict([[i] for i in phases])
    residuals = prediction_residuals(phase, mag, predictor)
    # determine phase shift for max light, if a specific shift was not provided
    if shift is None:
        arg_max_light = lightcurve.argmin()
        lightcurve = np.concatenate((lightcurve[arg_max_light:],
                                        lightcurve[:arg_max_light]))
        shift = arg_max_light/len(phases)
    # shift observed light curve to max light
    data.T[0] = rephase(data.data, period, shift).T[0]
    # use rephased phase points from *data* in residuals
    residuals = np.column_stack((data.T[0], residuals))

    # Grab the coefficients from the model
    coefficients = predictor.named_steps['Regressor'].coef_ \
        if isinstance(predictor, Pipeline) \
        else predictor.best_estimator_.named_steps['Regressor'].coef_

    # compute R^2 and MSE if they haven't already been
    # (one or zero have been computed, depending on the predictor)
    estimator = predictor.best_estimator_ \
        if hasattr(predictor, 'best_estimator_') \
        else predictor

    def get_score(scoring):
        if hasattr(predictor, 'best_score_') and predictor.scoring == scoring:
            return predictor.best_score_
        else:
            return cross_val_score(estimator, colvec(phase), mag,
                                   cv=scoring_cv, scoring=scoring,
                                   n_jobs=scoring_processes).mean()

    ## Formatting Coefficients ##
    # convert to phase-shifted form (A_k, Phi_k) if necessary
    if fourier_form != "sin_cos":
        coefficients = Fourier.phase_shifted_coefficients(coefficients,
                                                          form=fourier_form,
                                                          shift=shift)
        a_0 = coefficients[0]
        b_0 = 0
    else:
        # b_0 actually corresponds to A_0
        a_0 = 0
        b_0 = coefficients[0]
    # convert coefficients into a matrix of the form:
    #  /                  \
    # |  A_0/a_0 -----/A_0 |
    # |  A_1/a_1 Phi_1/b_1 |
    # |  .       .         |
    # |  .       .         |
    # |  .       .         |
    # |  A_n/a_n Phi_n/b_n |
    #  \                  /
    coeff_matrix = np.column_stack((
        # A_k's or A_0 + a_k's
        np.insert(coefficients[1::2], 0, a_0),
        # Phi_k's or b_k's, with a zero for the 0 term
        np.insert(coefficients[2::2], 0, b_0)))

    ## Coefficient Errors ##
    ## Note: currently we're not actually computing errors
    ##       for now we just give the SEM(predictions) as the error on A_0
    ##
    # initialize coefficient errors
    coeff_error_matrix = np.zeros_like(coeff_matrix)
    # use SEM(predictions) as err(A_0)
    coeff_error_matrix[0,0] = sem(lightcurve)

    ## Fourier Ratios ##
    ## Compute the Fourier ratios like R21 and Phi21,
    ## and put them in a table, but only if the form of the Fourier
    ## coefficients is (A_k, Phi_k), not (a_k, b_k)
    if fourier_form != "sin_cos":
        fourier_ratios = Fourier.fourier_ratios(coefficients)
        # TODO
        fourier_ratio_errors = np.zeros_like(fourier_ratios)
    else:
        fourier_ratios = None
        fourier_ratio_errors = None


    return {'name':                 name,
            'period':               period,
            'periodogram':          pgram,
            'lightcurve':           lightcurve,
            'coefficients':         coeff_matrix,
            'coefficient_errors':   coeff_error_matrix,
            'fourier_ratios':       fourier_ratios,
            'fourier_ratio_errors': fourier_ratio_errors,
            'phased_data':          data,
            'residuals':            residuals,
            'model':                predictor,
            'R2':                   get_score('r2'),
            'MSE':                  abs(get_score('mean_squared_error')),
            'degree':               estimator.get_params()['Fourier__degree'],
            'shift':                shift,
            'coverage':             coverage}


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
    data = np.loadtxt(file, skiprows=skiprows, usecols=use_cols)
    if len(data) != 0:
        masked_data = np.ma.array(data=data, mask=None, dtype=float)
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
#     data = np.ma.array(data, copy=copy)
#     time, mag, *err = data.T
#
#     tstart, tfinal = np.min(time), np.max(time)
#     periods = np.arange(tstart, tfinal+period, period)
#     data_range = (
#         data[np.logical_and(time>pstart, time<=pend),:]
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
#     data = np.ma.array(data=np.loadtxt(filename, usecols=use_cols,
#                                              skiprows=skiprows),
#                           mask=None, dtype=float)
#     return single_periods(data, *args, **kwargs)


def prediction_residuals(phase, mag, predictor):
    """prediction_residuals(phase, mag, predictor)

    Returns the residuals between the observed magnitudes, *mag*, and the
    magnitudes predicted by the *predictor* at the given *phase* points.

    **Parameters**

    phase : array-like, shape = [n_samples]
        Array of phases of observation.
    mag : array-like, shape = [n_samples]
        Array of observed magnitudes at the corresponding phases.
    predictor : object that has "fit" and "predict" methods, optional
        Object which predicts *mag* at the given *phase* points.

    **Returns**

    residuals : array-like, shape = [n_samples]
        Array of residuals between observed and fitted magnitudes.
    """
    # If one of the inputs is a masked array, the masked elements will be
    # skipped in the computation. This is undesirable, as we still want to know
    # the residuals for the outliers. As a workaround, we take out the masked
    # arrays' *data* and *mask* attributes, and perform the computation on the
    # data, reapplying the mask in the end. If the arrays are not masked, then
    # we treat them as if they are masked arrays with `mask=False`.

    # get data/mask information from *phase* array
    if np.ma.isMaskedArray(phase):
        phase_data = phase.data
        phase_mask = phase.mask
    else:
        phase_data = phase
        phase_mask = False
    # get data/mask information from *mag* array
    if np.ma.isMaskedArray(mag):
        mag_data = mag.data
        mag_mask = mag.mask
    else:
        mag_data = mag
        mag_mask = False
    # if an element is masked in either of the inputs, treat it as masked
    mask = np.logical_or(phase_mask, mag_mask)

    # compute the residuals, using the unmasked data
    residuals = predictor.predict(colvec(phase_data)) - mag_data
    # apply the mask to the residuals
    residuals = np.ma.array(residuals, mask=mask)

    return residuals


def find_outliers(data, predictor, sigma,
                  method=mad):
    """find_outliers(data, predictor, sigma, method=mad)

    Returns a boolean array indicating the outliers in the given *data* array.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Photometry array containing columns *phase*, *magnitude*, and (optional)
        *error*.
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
    abs_residuals = np.absolute(prediction_residuals(phase, mag, predictor))
    outliers = np.logical_and((abs_residuals > err[0]) if err else True,
                                 abs_residuals > sigma * method(abs_residuals))

    return np.tile(np.vstack(outliers), data.shape[1])


def savetxt_lightcurve(filename, phased_magnitudes,
                       fmt='%.18e', delimiter=' ',
                       header=""):
    """savetxt_lightcurve(filename, phased_magnitudes, fmt)

    Save a phased lightcurve to a text file.

    **Parameters**

    filename : str
        File to save lightcurve table to. Directory must exist.

    phased_magnitudes: array-like, shape = [n_samples]
        Array of phased magnitudes.

    fmt : str
        Number format string, as understood by :func:`np.savetxt`.

    **Returns**

        None
    """
    phases = np.linspace(0.0, 1.0, len(phased_magnitudes),
                            endpoint=False)

    data = np.column_stack((phases, phased_magnitudes))

    np.savetxt(filename, data,
                  fmt=fmt, delimiter=delimiter)


def plot_lightcurve(*args, engine='mpl', **kwargs):
    """plot_lightcurve(*args, engine='mpl', **kwargs)

    **Parameters**

    engine : str, optional
        Engine to use for plotting, choices are "mpl" and "tikz"
        (default "mpl")

    kwargs :
        See :func:`plot_lightcurve_mpl` and :func:`plot_lightcurve_tikz`,
        depending on *engine* specified.

    **Returns**

    plot : object
        Plot object. Type depends on *engine* used. "mpl" engine returns a
        `matplotlib.pyplot.Figure` object, and "tikz" engine returns a `str`.
    """
    if engine == "mpl":
        return(plot_lightcurve_mpl(*args, **kwargs))
    elif engine == "tikz":
        return(plot_lightcurve_tikz(*args, **kwargs))
    else:
        raise KeyError("engine '{}' does not exist".format(engine))


def plot_lightcurve_mpl(name, lightcurve, period, phased_data,
                        output=None, legend=False, sanitize_latex=False,
                        color=True, n_phases=100,
                        err_const=0.005,
                        **kwargs):
    """plot_lightcurve(name, lightcurve, period, phased_data, output=None, legend=False, color=True, n_phases=100, err_const=0.005, **kwargs)

    Save a plot of the given *lightcurve* to file *output*, using
    matplotlib and return the resulting plot object.

    **Parameters**

    name : str
        Name of the star. Used in plot title.
    lightcurve : array-like, shape = [n_samples]
        Fitted lightcurve.
    period : number
        Period to phase time by.
    phased_data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Photometry array containing columns *time*, *magnitude*, and
        (optional) *error*. *time* should be unphased.
    output : str, optional
        File to save plot to (default None).
    legend : boolean, optional
        Whether or not to display legend on plot (default False).
    color : boolean, optional
        Whether or not to display color in plot (default True).
    n_phases : integer, optional
        Number of phase points in fit (default 100).
    err_const : number, optional
        Constant to use in absence of error (default 0.005).

    **Returns**

    plot : matplotlib.pyplot.Figure
        Matplotlib Figure object which contains the plot.
    """
    phases = np.linspace(0, 1, n_phases, endpoint=False)

    # initialize Figure and Axes objects
    fig, ax = plt.subplots()

    # format the x- and y-axis
    ax.invert_yaxis()
    ax.set_xlim(0,2)

    # Plot points used
    phase, mag, *err = get_signal(phased_data).T

    error = err[0] if err else mag*err_const

    inliers = ax.errorbar(np.hstack((phase,1+phase)),
                          np.hstack((mag, mag)),
                          yerr=np.hstack((error, error)),
                          ls='None',
                          ms=.01, mew=.01, capsize=0)

    # Plot outliers rejected
    phase, mag, *err = get_noise(phased_data).T

    error = err[0] if err else mag*err_const

    outliers = ax.errorbar(np.hstack((phase,1+phase)),
                           np.hstack((mag, mag)),
                           yerr=np.hstack((error, error)),
                           ls='None', marker='o' if color else 'x',
                           ms=.01 if color else 4,
                           mew=.01 if color else 1,
                           capsize=0 if color else 1)

    # Plot the fitted light curve
    signal, = ax.plot(np.hstack((phases,1+phases)),
                      np.hstack((lightcurve, lightcurve)),
                      linewidth=1)

    if legend:
        ax.legend([signal, inliers, outliers],
                  ["Light Curve", "Inliers", "Outliers"],
                  loc='best')

    ax.set_xlabel('Phase ({0:0.7} day period)'.format(period))
    ax.set_ylabel('Magnitude')

    ax.set_title(utils.sanitize_latex(name) if sanitize_latex else name)
    fig.tight_layout(pad=0.1)

    if output is not None:
        fig.savefig(output)

    return fig


def plot_lightcurve_tikz(name, lightcurve, period, phased_data, coefficients,
                         output=None, legend=False, sanitize_latex=False,
                         color=True, n_phases=100,
                         err_const=0.005,
                         **kwargs):
    """plot_lightcurve(name, lightcurve, period, phased_data, output=None, legend=False, color=True, n_phases=100, err_const=0.005, **kwargs)

    Save TikZ source code for a plot of the given *lightcurve* to file
    *output*, and return the string holding the source code.

    **Parameters**

    name : str
        Name of the star. Used in plot title.
    lightcurve : array-like, shape = [n_samples]
        Fitted lightcurve.
    period : number
        Period to phase time by.
    phased_data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Photometry array containing columns *time*, *magnitude*, and
        (optional) *error*. *time* should be unphased.
    output : str, optional
        File to save plot to (default None).
    legend : boolean, optional
        Whether or not to display legend on plot (default False).
    color : boolean, optional
        Whether or not to display color in plot (default True).
    n_phases : integer, optional
        Number of phase points in fit (default 100).
    err_const : number, optional
        Constant to use in absence of error (default 0.005).

    **Returns**

    plot : str
        String containing the TikZ source code for the plot.
    """
    x_min = round(min(min(lightcurve), min(phased_data[:,1]))-0.05, 2)
    x_max = round(max(max(lightcurve), max(phased_data[:,1]))+0.05, 2)
    yticks = ", ".join("{:.2f}".format(x)
                       for x in np.linspace(x_min, x_max, 4))
    tikz = r"""\begin{tikzpicture}
    \begin{axis}[
        trig format plots=rad,
        xlabel near ticks,
        ylabel near ticks,
        width=0.95\linewidth,
        height=0.25\textheight,
        xmin=0,
        xmax=2,
        xtick={0, 0.5, 1, 1.5, 2},
        xlabel={Phase (%s day period)},
        ymin=%s,
        ymax=%s,
        ytick={%s},
        ylabel={Magnitude},
        y dir=reverse
    ]
    \addplot[
        domain=0:2,
        samples=200,
        color=black,
        solid,
        line width=0.75pt
    ] {
""" % (period, x_min, x_max, yticks)

    # Add light curve
    for (k, A) in enumerate(coefficients):
        if k == 0:
            tikz += r"""        %s"""%A
        elif (A == 0):
            continue
        elif k % 2:
            tikz += r""" +
        sin(2*pi*%s*(x+%s)) * %s""" % (int((k-1)/2+1), kwargs['shift'], A)
        else:
            tikz += r""" +
        cos(2*pi*%s*(x+%s)) * %s""" % (int(k/2), kwargs['shift'], A)

    # Add points
    tikz += r"""
    };
    \addplot[
        color=black!50!red,
        mark size=0pt,
        only marks
    ]
    plot[
        error bars/.cd,
        y dir = both,
        y explicit,
        error mark options={
            mark size=0pt,
            line width=1pt
        }
    ]
    table[
        row sep=crcr,
        y error plus index=2,
        y error minus index=3
    ] {"""
    for row in get_signal(phased_data):
        tikz += r"""
            %s %s %s %s \\
            %s %s %s %s \\""" % (row[0], row[1], row[2]/2, row[2]/2,
                               1+row[0], row[1], row[2]/2, row[2]/2)

    # Add outliers
    if (len(get_noise(phased_data))>0):
        tikz += r"""
        };
        \addplot[
            color=black,
            mark size=0pt,
            only marks
        ]
        plot[
            error bars/.cd,
            y dir = both,
            y explicit,
            error mark options={
                mark size=0pt,
                line width=1pt
            }
        ]
        table[
            row sep=crcr,
            y error plus index=2,
            y error minus index=3
        ] {"""
        for row in get_noise(phased_data):
            tikz += r"""
                %s %s %s %s \\
                %s %s %s %s \\""" % (row[0], row[1], row[2]/2, row[2]/2,
                                   1+row[0], row[1], row[2]/2, row[2]/2)

    # Done!
    tikz += r"""
        };
    \end{axis}
\end{tikzpicture}"""

    # save tikz to a file
    if output is not None:
        with open(output, "w") as f:
            f.write(tikz)

    return tikz


def plot_residual(*args, engine='mpl', **kwargs):
    """plot_residual(*args, engine='mpl', **kwargs)

    **Parameters**

    engine : str, optional
        Engine to use for plotting, choices are "mpl" and "tikz"
        (default "mpl")

    kwargs :
        See :func:`plot_residuals_mpl` and :func:`plot_residuals_tikz`,
        depending on *engine* specified.

    **Returns**

    plot : object
        Plot object. Type depends on *engine* used. "mpl" engine returns a
        `matplotlib.pyplot.Figure` object, and "tikz" engine returns a `str`.
    """
    if engine == "mpl":
        return(plot_residual_mpl(*args, **kwargs))
    elif engine == "tikz":
        return(plot_residual_tikz(*args, **kwargs))
    else:
        raise KeyError("engine '{}' does not exist".format(engine))


def plot_residual_mpl(name, residuals, period,
                      output=None, legend=False, sanitize_latex=False,
                      color=True,
                      **kwargs):
    """plot_residual_mpl(name, residuals, period, output='.', sanitize_latex=False, color=True, **kwargs)

    Save a plot of the given *residuals* to file *output*, using
    matplotlib and return the resulting plot object.

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
    # initialize Figure and Axes objects
    fig, ax = plt.subplots()

    phase, resid = residuals.T

    # format the x- and y-axis
    ax.invert_yaxis()
    ax.set_xlim(0,2)

    inliers = ax.scatter(np.concatenate((phase,1+phase)),
                         np.concatenate((resid, resid)))

    # Plot outliers rejected
#    phase, mag = get_noise(residuals).T

#    outliers = ax.scatter(np.hstack((phase,1+phase)),
#                          np.hstack((resid, resid)))

    if legend:
        ax.legend([inliers],
                  ["Inliers"],
                  loc='best')

    ax.set_xlabel('Phase ({0:0.7} day period)'.format(period))
    ax.set_ylabel('Magnitude Residuals')

    ax.set_title(utils.sanitize_latex(name) if sanitize_latex else name)
    fig.tight_layout(pad=0.1)

    if output is not None:
        fig.savefig(output)

    return fig


def plot_residual_tikz(name, residuals, period,
                       output=None, sanitize_latex=False,
                       color=True,
                       **kwargs):
    """plot_residual_tikz(name, residuals, period, output=None, sanitize_latex=False, color=True, **kwargs)

    Save TikZ source code for a plot of the given *residuals* to file
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
