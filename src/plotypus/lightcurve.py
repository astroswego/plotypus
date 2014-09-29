import numpy
numpy.random.seed(0)
from scipy.stats import sem
from sys import stderr
from math import floor
from os import path
from .utils import (verbose_print, make_sure_path_exists,
                    get_signal, get_noise, colvec, mad)
from .periodogram import find_period, rephase, get_phase
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
                   Selector=GridSearchCV, fourier_degree=(2,25),
                   selector_processes=1,
                   use_baart=False, scoring='r2', scoring_cv=3,
                   **kwargs):
    """Makes a predictor object for use in get_lightcurve.
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

def get_lightcurve(data, name=None, period=None,
                   predictor=make_predictor(),
                   min_period=0.2, max_period=32,
                   coarse_precision=0.001, fine_precision=0.0000001,
                   sigma=20, sigma_clipping='robust',
                   scoring='r2', scoring_cv=3, scoring_processes=1,
                   min_phase_cover=0.,
                   phases=numpy.arange(0, 1, 0.01),
                   verbosity=[], **ops):
    if predictor is None:
        predictor = make_predictor(scoring=scoring, scoring_cv=scoring_cv)

    while True:
        signal = get_signal(data)
        if len(signal) <= scoring_cv:
            return

        # Find the period of the inliers
        _period = period if period is not None else \
                  find_period(signal.T[0], signal.T[1],
                              min_period, max_period,
                              coarse_precision, fine_precision)
        phase, mag, *err = rephase(signal, _period).T

        # Determine whether there is sufficient phase coverage
        coverage = numpy.zeros((100))
        for p in phase:
            coverage[int(floor(p*100))] = 1
        coverage = sum(coverage)/100
        if coverage < min_phase_cover:
            verbose_print("{} {} {}".format(name, coverage, min_phase_cover),
                          operation="outlier",
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
            outliers = find_outliers(data.data, _period, predictor, sigma,
                                     sigma_clipping)
            num_outliers = sum(outliers)[0]
            if num_outliers == 0 or \
               set.issubset(set(numpy.nonzero(outliers.T[0])[0]),
                            set(numpy.nonzero(data.mask.T[0])[0])):
                data.mask = outliers
                break
            if num_outliers > 0:
                verbose_print("{} {} outliers".format(name, sum(outliers)[0]),
                              operation="outlier",
                              verbosity=verbosity)
            data.mask = numpy.ma.mask_or(data.mask, outliers)

    # Build light curve and shift to max light
    lightcurve = predictor.predict([[i] for i in phases])
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



    return {'name': name,
            'period': _period,
            'lightcurve': lightcurve,
            'coefficients': coefficients[0],
            'dA_0': sem(lightcurve),
            'phased_data': data,
            'model': predictor,
            'R2': get_score('r2'),
            'MSE': abs(get_score('mean_squared_error')),
            'degree': estimator.get_params()['Fourier__degree'],
            'shift': shift,
            'coverage': coverage}

def get_lightcurve_from_file(filename, *args, use_cols=None, **kwargs):
    data = numpy.ma.array(data=numpy.loadtxt(filename, usecols=use_cols),
                          mask=None, dtype=float)
    return get_lightcurve(data, *args, **kwargs)

def get_lightcurves_from_file(filename, directories, *args, **kwargs):
    return [get_lightcurve_from_file(path.join(d, filename), *args **kwargs)
            for d in directories]

def single_periods(data, period, min_points=10, *args, **kwargs):
    time, mag, *err = data.T

    tstart, tfinal = numpy.min(time), numpy.max(time)
    periods = numpy.arange(tstart, tfinal+period, period)
    data_range = (
        data[numpy.logical_and(time>pstart, time<=pend),:]
        for pstart, pend in zip(periods[:-1], periods[1:])
    )

    return (
        get_lightcurve(d, period=period, *args, **kwargs)
        for d in data_range
        if d.shape[0] > min_points
    )

def single_periods_from_file(filename, *args, use_cols=range(3), **kwargs):
    data = numpy.ma.array(data=numpy.loadtxt(filename, usecols=use_cols),
                          mask=None, dtype=float)
    return single_periods(data, *args, **kwargs)

def find_outliers(data, period, predictor, sigma,
                  sigma_clipping='robust'):
    # determine sigma clipping function
    sigma_clipper = mad if sigma_clipping == 'robust' else numpy.std

    phase, mag, *err = rephase(data, period).T
    residuals = numpy.absolute(predictor.predict(colvec(phase)) - mag)
    outliers = numpy.logical_and((residuals > err[0]) if err else True,
                                 residuals > sigma * sigma_clipper(residuals))
    return numpy.tile(numpy.vstack(outliers), data.shape[1])

def plot_lightcurve(name, lightcurve, period, data, output='.', legend=False,
                    color=True, phases=numpy.arange(0, 1, 0.01),
                    err_const=0.0004,
                    **ops):
    ax = plt.gca()
    #ax.grid(False
    ax.invert_yaxis()
    plt.xlim(0,2)

    # Plot points used
    phase, mag, *err = get_signal(data).T

    error = err[0] if err else mag*err_const

    inliers = plt.errorbar(numpy.hstack((phase,1+phase)),
                           numpy.hstack((mag, mag)),
                           yerr=numpy.hstack((error, error)),
                           ls='None',
                           ms=.01, mew=.01, capsize=0, elinewidth=0.1)

    # Plot outliers rejected
    phase, mag, *err = get_noise(data).T

    error = err[0] if err else mag*err_const

    outliers = plt.errorbar(numpy.hstack((phase,1+phase)),
                            numpy.hstack((mag, mag)),
                            yerr=numpy.hstack((error, error)),
                            ls='None', marker='o' if color else 'x',
                            ms=.01 if color else 4,
                            mew=.01 if color else 1,
                            capsize=0 if color else 1,
                            elinewidth=0.1)

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

    plt.title(name)
    plt.tight_layout(pad=0.1)
    make_sure_path_exists(output)
    plt.savefig(path.join(output, name))
    plt.clf()
