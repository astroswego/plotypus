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
    parser.add_argument('--predictor',
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
    if args.predictor is 'Baart':
        raise ArgumentError("Baart's criteria not yet implemented")

    regressor_choices = {'LassoCV': LassoCV(cv=args.cv,
                                            max_iter=args.max_iter),
                         'OLS': LinearRegression()}

    predictor_choices = {'Baart': None,
                         'GridSearchCV': GridSearchCV}

    args.regressor = regressor_choices[args.regressor]
    Predictor = predictor_choices[args.predictor] or GridSearchCV
    args.predictor = make_predictor(Predictor=Predictor,
                                    use_baart=args.predictor is 'Baart',
                                    **args.__dict__)
    args.phases=numpy.arange(0, 1, 1/args.phase_points)

    return args

def main():
    ops = get_args()
    if ops.periods is not None:
        periods = {name: float(period) for (name, period)
                   in (line.strip().split() for line
                       # generalize to all whitespace instead of just spaces
                       in ops.periods if ' ' in line)}
        ops.periods.close()

    max_coeffs = 2*ops.__dict__['fourier_degree'][1]+1
    filenames = list(map(lambda x: x.strip(), _get_files(ops.input)))
    filepaths = map(lambda filename:
                    filename if path.isfile(filename)
                             else path.join(ops.input, filename),
                    filenames)
    star_names = list(map(lambda filename:
                          path.basename(filename).split('.')[0],
                          filenames))
    # a dict containing all options which can be pickled, because
    # all parameters to pmap must be picklable
    picklable_ops = {k: ops.__dict__[k]
                     for k in ops.__dict__
                     if k not in {'input', 'periods'}}
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
    printer = _star_printer(max_coeffs, ops.__dict__['format'])
    pmap(process_star, filepaths, callback=printer, **picklable_ops)

def process_star(filename, periods={}, **ops):
    """Processes a star's lightcurve, prints its coefficients, and saves
    its plotted lightcurve to a file. Returns the results of get_lightcurve.
    """
    name = path.basename(filename).split('.')[0]
    _period = periods.get(name)
    result = get_lightcurve(filename, period=_period, **ops)

    if result is not None:
        period, lc, data, coefficients, R_squared = result
        plot_lightcurve(name, lc, period, data, **ops)
        return name, period, lc, data, coefficients, R_squared

def _star_printer(max_coeffs, fmt):
    return lambda results: _print_star(results, max_coeffs, fmt)

def _print_star(results, max_coeffs, fmt):
    if results is None: return
    formatter = lambda x: fmt % x

    name, period, lc, data, coefficients, R_squared = results
    print(' '.join([name, str(period), str(R_squared)]), end=' ')
    print(' '.join(map(formatter, coefficients)), end=' ')
    trailing_zeros = max_coeffs - len(coefficients)
    if trailing_zeros > 0:
        print(' '.join(map(formatter,
                           numpy.zeros(trailing_zeros))), end=' ')
    print(' '.join(map(formatter, lc)))

def _get_files(input):
    if input is stdin:
        return map(lambda x: x.strip(), input)
    elif path.isdir(input):
        return sorted(listdir(input))
    else:
        with open(input, 'r') as f:
            return map(lambda x: x.strip(), f.readlines())
                         
if __name__ == "__main__":
    exit(main())
