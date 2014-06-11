import numpy
from sys import exit
from os import path, listdir
from math import floor
from utils import make_sure_path_exists, get_signal, get_noise
from periodogram import find_period, rephase, get_phase
from Fourier import Fourier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import warnings

def get_ops():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-i', '--input', type='string',
        default=path.join('..', 'data', 'lmc', 'i', 'cep', 'f'),
        help='location of stellar observations',)
    parser.add_option('-o', '--output', type='string',
        default=path.join('..', 'results'),
        help='location of results')
    parser.add_option('--min_period',       dest='min_period',       type='float',
      default=0.2,    help='minimum period of each star')
    parser.add_option('--max_period',       dest='max_period',       type='float',
      default=32.,    help='maximum period of each star')
    parser.add_option('--coarse_precision', dest='coarse_precision', type='int',
      default=0.001,  help='level of granularity on first pass')
    parser.add_option('--fine_precision',   dest='fine_precision',   type='int',
      default=0.0000001, help='level of granularity on second pass')
    parser.add_option('--fourier_degree',   dest='fourier_degree',   type='int',
      default=15,     help='number of coefficients to generate')
    parser.add_option('--sigma',            dest='sigma',            type='float',
      default=4,      help='rejection criterion for outliers')
    parser.add_option('--cv',               dest='cv', type='int',
      default=10,     help='number of folds in the Lasso regularization search')
    parser.add_option('--min_phase_cover', dest='min_phase_cover',   type='float',
      default=1/2.,   help='minimum fraction of phases that must have observations')
    (options, args) = parser.parse_args()
    return options

def main():
    ops = get_ops()
    lcs = []
    for filename in sorted(listdir(ops.input)):
        print(filename)
        star = get_lightcurve(path.join(ops.input, filename),
                              ops.fourier_degree, ops.cv,
                              ops.min_period, ops.max_period,
                              ops.coarse_precision, ops.fine_precision,
                              ops.sigma, ops.min_phase_cover)

        if star is not None:
            #period, min_light, mean_light, max_light, lc, data = star
            #lcs += [[period, min_light, mean_light, max_light] + list(lc)]
            period, lc, data = star
            lcs += [[period] + list(lc)]
            #plot_lightcurve(ops.output, filename, lc, period, data)#, mean_light)

    numpy.savetxt(path.join(ops.output, 'lightcurves.dat'), numpy.array(lcs), fmt='%.5f')#,
                  #header='Period ' + ' '.join(['Phase' + str(i) for i in range(100)]))
                  #header='Period MinLight MeanLight MaxLight ' + \
                  #       ' '.join(['Phase' + str(i) for i in range(100)]))

def get_lightcurve(filename, fourier_degree=15, cv=10,
                   min_period=0.2, max_period=32,
                   coarse_precision=0.001, fine_precision=0.0000001,
                   sigma=5, min_phase_cover=2/3.,
                   phases=numpy.arange(0, 1, 0.01)):

    # Initialize predictor
    predictor = GridSearchCV(Pipeline([('Fourier', Fourier()),
                                       ('Lasso',   LassoCV())]),
                             {'Fourier__degree': range(3, 1+fourier_degree)})

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
        X = numpy.resize(phase, (phase.shape[0], 1)) # Column vector
        with warnings.catch_warnings(record=True) as w:
            try:
                predictor = predictor.fit(X, mag)
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
    mean_light = lc.mean()
    min_light = lc.max() # Yes, it is unfortunately
    max_light = lc.min() # this way in astronomy.

    # Shift to max light
    arg_max_light = lc.argmin()
    lc = numpy.concatenate((lc[arg_max_light:], lc[:arg_max_light]))
    data.T[0] = numpy.array([get_phase(p, period, arg_max_light / 100.)
                             for p in data.data.T[0]])

    # Normalize magnitudes based on fitted light curve
    #data.T[1] = (data.T[1] - max_light) / (min_light - max_light)
    #lc = (lc - max_light) / (min_light - max_light)
    return period, lc, data #min_light, mean_light, max_light, lc, data

def find_outliers(data, period, predictor, sigma):
    phase, mag, err = rephase(data, period).T
    phase = numpy.resize(phase, (phase.shape[0], 1))
    residuals = abs(predictor.predict(phase) - mag)
    mse = numpy.array([0 if residual < error else (residual - error)**2
                       for residual, error in zip(residuals, err)])
    return numpy.tile(numpy.vstack(mse > sigma * mse.std()), data.shape[1])

def plot_lightcurve(output, filename, lc, period, data, #mean_light,
                    phases = numpy.arange(0, 1, 0.01)):
    ax = plt.gca()
    ax.grid(True)
    ax.invert_yaxis()
    plt.xlim(-0.1,2.1)
    #plt.ylim(max(data.T[1])+max(data.T[2]), min(data.T[1])+max(data.T[2]))
    #plt.ylim(1.2,-0.2)

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
    plt.ylabel('Magnitude') #({0:0.7} at mean light)'.format(mean_light))
    plt.title(filename.split('.')[0])
    make_sure_path_exists(output)
    plt.savefig(path.join(output, filename + '.png'))
    plt.clf()

if __name__ == "__main__":
    exit(main())
