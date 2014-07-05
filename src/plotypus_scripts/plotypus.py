import numpy
from sys import exit, stdin, stderr
from os import path, listdir
from argparse import ArgumentError, ArgumentParser, FileType
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.grid_search import GridSearchCV
from plotypus.lightcurve import make_predictor, get_lightcurve, plot_lightcurve
from plotypus.preprocessing import Fourier
from plotypus.utils import pmap

def get_args():
    parser = ArgumentParser()
    general_group = parser.add_argument_group('General')
    period_group = parser.add_argument_group('Periodogram')
    fourier_group = parser.add_argument_group('Fourier')
    lasso_group = parser.add_argument_group('Lasso')
    gridsearch_group = parser.add_argument_group('GridSearch')

    general_group.add_argument('-i', '--input', type=str,
        default=stdin,
        help='location of stellar observations '
             '(default = stdin)')
    general_group.add_argument('-o', '--output', type=str,
        default='plots',
        help='location of plots '
             '(default = plots)')
    general_group.add_argument('-f', '--format', type=str,
        default='%.5f',
        help='format specifier for output table')
    general_group.add_argument('--data-extension', type=str,
        default='.dat', metavar='EXT',
        help='extension which follows a star\'s name in data filenames '
             '(default = .dat)')
    general_group.add_argument('--use-cols', type=int, nargs=3,
        default=range(3), metavar=('TIME', 'MAG', 'MAG_ERR'),
        help='columns to use from data file '
             '(default = 0 1 2)')
    general_group.add_argument('-p', '--processes', type=int,
        default=1, metavar='N',
        help='number of stars to process in parallel '
             '(default = 1)')
    general_group.add_argument('-s', '--scoring', type=str,
        choices=['MSE', 'R2'], default='R2',
        help='scoring metric to use '
             '(default = R2)')
    general_group.add_argument('--scoring-cv', type=int,
        default=3, metavar='N',
        help='number of folds in the scoring cross validation '
             '(default = 3)')
    general_group.add_argument('--phase-points', type=int,
        default=100, metavar='N',
        help='number of phase points to output '
             '(default = 100)')
    general_group.add_argument('--min-phase-cover', type=float,
        default=0, metavar='COVER',
        help='minimum fraction of phases that must have points '
             '(default = 0)')
    period_group.add_argument('--periods', type=FileType('r'),
        default=None,
        help='file of star names and associated periods '
             '(default = None)')
    period_group.add_argument('--min-period', type=float,
        default=0.2, metavar='P',
        help='minimum period of each star '
             '(default = 0.2)')
    period_group.add_argument('--max-period', type=float,
        default=32.0, metavar='P',
        help='maximum period of each star '
             '(default = 32.0)')
    period_group.add_argument('--coarse-precision', type=float,
        default=0.001,
        help='level of granularity on first pass '
             '(default = 0.001)')
    period_group.add_argument('--fine-precision', type=float,
        default=0.0000001,
        help='level of granularity on second pass '
             '(default = 0.0000001)')
    fourier_group.add_argument('--fourier-degree', type=int, nargs=2,
        default=(3,15), metavar=('MIN', 'MAX'),
        help='range of degrees of fourier fits to use '
             '(default = 3 15)')
    fourier_group.add_argument('-r', '--regressor',
        choices=['Lasso', 'OLS'],
        default='Lasso',
        help='type of regressor to use '
             '(default = Lasso)')
    fourier_group.add_argument('--predictor',
        choices=['Baart', 'GridSearch'],
        default='GridSearch',
        help='type of model predictor to use '
             '(default = GridSearch)')
    fourier_group.add_argument('--sigma', dest='sigma', type=float,
        default=10.0,
        help='rejection criterion for outliers '
             '(default = 10)')
    fourier_group.add_argument('--standard-sigma-clipping',
        dest='robust_sigma_clipping', action='store_false',
        help='use standard deviation sigma clipping '
             '(not the default)')
    fourier_group.add_argument('--robust-sigma-clipping',
        action='store_true',
        help='use median absolute deviation sigma clipping '
             '(the default)')
    lasso_group.add_argument('--lasso-cv', type=int,
        default=10, metavar='N',
        help='number of folds in the L1-regularization search '
             '(default = 10)')
    lasso_group.add_argument('--max-iter', type=int,
        default=1000, metavar='N',
        help='maximum number of iterations in the LassoCV '
             '(default = 1000)')
    
    args = parser.parse_args()


    regressor_choices = {'Lasso': LassoCV(cv=args.lasso_cv,
                                          max_iter=args.max_iter),
                         'OLS': LinearRegression()}

    predictor_choices = {'Baart': None,
                         'GridSearch': GridSearchCV}

    scoring_choices = {'R2': 'r2',
                       'MSE': 'mean_squared_error'}

    args.scoring = scoring_choices[args.scoring]
    args.regressor = regressor_choices[args.regressor]
    Predictor = predictor_choices[args.predictor] or GridSearchCV
    args.predictor = make_predictor(Predictor=Predictor,
                                    use_baart=(args.predictor == 'Baart'),
                                    **args.__dict__)
    args.phases=numpy.arange(0, 1, 1/args.phase_points)

    if args.periods is not None:
        periods = {name: float(period) for (name, period)
                   in (line.strip().split() for line
                       # generalize to all whitespace instead of just spaces
                       in args.periods if ' ' in line)}
        args.periods.close()
        args.periods = periods
    return args

def main():
    ops = get_args()

    min_degree, max_degree = ops.__dict__['fourier_degree']
    max_coeffs = 2*max_degree + 1
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
                     if k not in {'input'}}
    # print file header
    print(' '.join([
        '#',
        'Name',
        'Period',
        'R^2',
        'MSE',
        'A_0',
        ' '.join(map('A_{0} Phi_{0}'.format, range(1, max_degree+1))),
        ' '.join(map('Phase{}'.format, range(ops.phase_points)))
        ])
    )
    printer = _star_printer(max_coeffs, ops.__dict__['format'])
    pmap(process_star, filepaths, callback=printer, **picklable_ops)

def process_star(filename, periods={}, **ops):
    """Processes a star's lightcurve, prints its coefficients, and saves
    its plotted lightcurve to a file. Returns the results of get_lightcurve.
    """
    _name = path.basename(filename)
    extension = ops['data_extension']
    if _name.endswith(extension):
        name = _name[:-len(extension)]
    else:
        # file has wrong extension
        return
    _period = periods.get(name)
    result = get_lightcurve(filename, period=_period, **ops)

    if result is not None:
        period, lc, data, coefficients, R_squared, MSE = result
        plot_lightcurve(name, lc, period, data, **ops)
        return name, period, lc, data, coefficients, R_squared, MSE

def _star_printer(max_coeffs, fmt):
    return lambda results: _print_star(results, max_coeffs, fmt)

def _print_star(results, max_coeffs, fmt):
    if results is None: return
    formatter = lambda x: fmt % x

    name, period, lc, data, coefficients, R2, MSE = results
    print(' '.join([name, str(period), str(R2), str(MSE)]), end=' ')
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
