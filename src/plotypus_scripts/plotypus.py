import numpy
from sys import exit, stdin
from os import path, listdir
from argparse import ArgumentError, ArgumentParser, FileType
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.grid_search import GridSearchCV
from plotypus.lightcurve import make_predictor, get_lightcurve, plot_lightcurve
from plotypus.preprocessing import Fourier
from plotypus.utils import pmap

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
        default=stdin,
        help='location of stellar observations',)
    parser.add_argument('-o', '--output', type=str,
        default=path.join('..', 'results'),
        help='location of results')
    parser.add_argument('-f', '--format', type=str,
        default='%.5f',
        help='format specifier for output table')
    parser.add_argument('-p', '--processes', type=int,
        default=1,
        help='number of stars to process in parallel')
    parser.add_argument('--periods', type=FileType('r'),
        default=None,
        help='file of star names and associated periods')
    parser.add_argument('--phase-points', type=int,
        default=100,
        help='number of phase points to use')
    parser.add_argument('--min-period', type=float,
        default=0.2,
        help='minimum period of each star')
    parser.add_argument('--max-period', type=float,
        default=32.0,
        help='maximum period of each star')
    parser.add_argument('--coarse-precision', type=int,
        default=0.001,
        help='level of granularity on first pass')
    parser.add_argument('--fine-precision', type=int,
        default=0.0000001,
        help='level of granularity on second pass')
    parser.add_argument('--fourier-degree', type=int, nargs=2,
        default=(3,15),
        help='number of coefficients to generate')
    parser.add_argument('-r', '--regressor',
        choices=['LassoCV', 'OLS'],
        default='LassoCV',
        help='type of regressor to use')
    parser.add_argument('--predictor', dest='Predictor',
        choices=['Baart', 'GridSearchCV'],
        default='GridSearchCV',
        help='type of model predictor to use')
    parser.add_argument('--sigma', dest='sigma', type=float,
        default=10.0,
        help='rejection criterion for outliers')
    parser.add_argument('--standard-sigma-clipping',
        dest='robust_sigma_clipping', action='store_false',
        help='use standard deviation sigma clipping')
    parser.add_argument('--robust-sigma-clipping',
        action='store_true',
        help='use median absolute deviation sigma clipping')
    parser.add_argument('--cv', type=int,
        default=10,
        help='number of folds in the L1-regularization search')
    parser.add_argument('--max-iter', type=int,
        default=1000,
        help='maximum number of iterations in the LassoCV')
    parser.add_argument('--min-phase-cover', type=float,
        default=1/2,
        help='minimum fraction of phases that must have points')
    args = parser.parse_args()

#    if args.input is stdin:
#        raise Exception("Reading from stdin working yet")
    if args.Predictor is 'Baart':
        raise ArgumentError("Baart's criteria not yet implemented")
    
    regressor_choices = {'LassoCV': LassoCV(cv=args.cv,
                                            max_iter=args.max_iter),
                         'OLS': LinearRegression()}

    predictor_choices = {'Baart': None,
                         'GridSearchCV': GridSearchCV}

    args.regressor = regressor_choices[args.regressor]
    args.Predictor = predictor_choices[args.Predictor]

    return args

def main():
    ops = get_args()
    if ops.periods is not None:
        periods = {name: float(period) for (name, period)
                   in (line.strip().split() for line
                       # generalize to all whitespace instead of just spaces
                       in ops.periods if ' ' in line)}
        ops.periods.close()


    formatter = lambda x: ops.format % x    
    max_coeffs = 2*ops.fourier_degree[1]+1
    phases=numpy.arange(0, 1, 1/ops.phase_points)

    
    filenames = list(map(lambda x: x.strip(), _get_files(ops.input)))
    filepaths = map(lambda filename:
                    filename if path.isfile(filename)
                             else path.join(ops.input, filename),
                    filenames)
    star_names = list(map(lambda filename:
                          path.basename(filename).split('.')[0],
                          filenames))
    _periods = map(lambda name: periods[name], star_names)
    # a dict containing all options which can be pickled
    # all parameters to pmap must be picklable
    picklable_ops = {k: ops.__dict__[k]
                     for k in ops.__dict__
                     if k not in {'input', 'output', 'periods'}}
    results = pmap(_get_lightcurve, zip(filepaths, _periods),
                   phases=phases, **picklable_ops)
    # print file header
    print(' '.join([
        '#',
        'Name',
        'Period',
        'R^2',
        'A_0',
        ' '.join(map('A_{0} Phi_{0}'.format, range(1, max_coeffs))),
        ' '.join(map('Phase{}'.format, range(ops.phase_points)))
        ])
    )
    # this needs to be parallelized as well
    for name, result in zip(star_names, results):
        if result is not None:
            period, lc, data, coefficients, R_squared = result
            _print_star(name, period, R_squared,
                       Fourier.phase_shifted_coefficients(coefficients),
                       max_coeffs, lc, formatter)
            plot_lightcurve(name, lc, period, data, phases=phases,
                            **ops.__dict__)

def _get_lightcurve(filename_period, **ops):
    filename, period = filename_period
    return get_lightcurve(filename, period=period, **ops)

def _get_files(input):
    if input is stdin:
        return input
    elif path.isdir(input):
        return sorted(listdir(input))
    else:
        with open(input, 'r') as f:
            return f.readlines()
            
def _print_star(name, period, R_squared,
               coefficients, max_coeffs, lc, formatter):
    print(' '.join([name, str(period), str(R_squared)]), end=' ')
    print(' '.join(map(formatter, coefficients)), end=' ')
    trailing_zeros = max_coeffs - len(coefficients)
    if trailing_zeros > 0:
        print(' '.join(map(formatter, numpy.zeros(trailing_zeros))), end=' ')
    print(' '.join(map(formatter, lc)))
                         
if __name__ == "__main__":
    exit(main())
