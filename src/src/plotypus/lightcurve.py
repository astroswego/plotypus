import numpy
from math import floor
from os import path
from .utils import make_sure_path_exists, get_signal, get_noise, colvec
from .periodogram import find_period, rephase, get_phase
from .Fourier import Fourier
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import warnings
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

__all__ = [
    'get_lightcurve',
    'find_outliers',
    'plot_lightcurve'
]

def get_lightcurve(filename, fourier_degree=15, cv=10,
                   min_period=0.2, max_period=32,
                   coarse_precision=0.001, fine_precision=0.0000001,
                   sigma=5, min_phase_cover=2/3.,
                   phases=numpy.arange(0, 1, 0.01), **options):
    
    # Initialize predictor
    pipeline = Pipeline([('Fourier', Fourier()), ('Lasso', LassoCV(cv=cv))])
    params = {'Fourier__degree': list(range(3, 1+fourier_degree))}
    predictor = GridSearchCV(pipeline, params)
    
    # Load file
    data = numpy.ma.masked_array(data=numpy.loadtxt(filename), mask=None)
    
    while True:
        # Find the period of the inliers
        signal = get_signal(data)
        period = find_period(signal.T[0], signal.T[1], min_period, max_period,
                             coarse_precision, fine_precision)
        phase, mag, err = rephase(signal, period).T
    
        # Determine whether there is sufficient phase coverage
        coverage = numpy.zeros((100))
        for p in phase:
            coverage[int(floor(p*100))] = 1
        if sum(coverage)/100. < min_phase_cover:
            print(sum(coverage)/100., min_phase_cover)
            print("Insufficient phase coverage")
            return None
    
        # Predict light curve
        with warnings.catch_warnings(record=True) as w:
            try:
                predictor = predictor.fit(colvec(phase), mag)
            except Warning:
                print(w)
                return None
    
        # Reject outliers and repeat the process if there are any
        if sigma:
            outliers = find_outliers(data.data, period, predictor, sigma)
            if set.issubset(set(numpy.nonzero(outliers.T[0])[0]),
                            set(numpy.nonzero(data.mask.T[0])[0])):
                break
            print("Rejecting", sum(outliers)[0], "outliers")
            data.mask = numpy.ma.mask_or(data.mask, outliers)
    
    # Build light curve
    lc = predictor.predict([[i] for i in phases])
    
    # Shift to max light
    arg_max_light = lc.argmin()
    lc = numpy.concatenate((lc[arg_max_light:], lc[:arg_max_light]))
    data.T[0] = numpy.array([get_phase(p, period, arg_max_light / 100.)
                             for p in data.data.T[0]])
    
    return period, lc, data

def find_outliers(data, period, predictor, sigma):
    phase, mag, err = rephase(data, period).T
    phase = numpy.resize(phase, (phase.shape[0], 1))
    residuals = abs(predictor.predict(phase) - mag)
    mse = numpy.array([0 if residual < error else (residual - error)**2
                       for residual, error in zip(residuals, err)])
    return numpy.tile(numpy.vstack(mse > sigma * mse.std()), data.shape[1])

def plot_lightcurve(output, filename, lc, period, data,
                    phases=numpy.arange(0, 1, 0.01),
                    grid=True, invert=True):
    ax = plt.gca()
    ax.grid(grid)
    if invert:
        ax.invert_yaxis()
    plt.xlim(-0.1,2.1)
    
    # Plot the fitted light curve
    plt.plot(numpy.hstack((phases,1+phases)), numpy.hstack((lc, lc)),
             linewidth=1.5, color='r')
    
    # Plot points used
    phase, mag, err = get_signal(data).T
    plt.errorbar(numpy.hstack((phase,1+phase)), numpy.hstack((mag, mag)),
                 yerr = numpy.hstack((err,err)), ls='None', ms=.01, mew=.01)
    
    # Plot outliers rejected
    phase, mag, err = get_noise(data).T
    plt.errorbar(numpy.hstack((phase,1+phase)), numpy.hstack((mag, mag)),
                 yerr = numpy.hstack((err,err)), ls='None', ms=.01, mew=.01,
                 color='r')
    
    plt.xlabel('Phase ({0:0.7} day period)'.format(period))
    plt.ylabel('Magnitude')
    plt.title(filename.split('.')[0])
    make_sure_path_exists(output)
    plt.savefig(path.join(output, filename + '.png'))
    plt.clf()
