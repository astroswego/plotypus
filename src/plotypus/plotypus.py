import numpy
from sys import exit, stdin, stderr
from os import path, listdir
from argparse import ArgumentError, ArgumentParser, FileType, SUPPRESS
from sklearn.linear_model import LassoCV, LassoLarsIC, LinearRegression
from sklearn.grid_search import GridSearchCV
from matplotlib import rc_params_from_file
import plotypus.lightcurve
from plotypus.lightcurve import (make_predictor, get_lightcurve_from_file,
                                 plot_lightcurve)
import plotypus
from plotypus.preprocessing import Fourier
from plotypus.utils import pmap

default_matplotlibrc = plotypus.__file__.split(path.sep)[:-3]
default_matplotlibrc.append('matplotlibrc')
default_matplotlibrc = path.sep.join(default_matplotlibrc)

def get_args():
    parser = ArgumentParser()
    general_group = parser.add_argument_group('General')
    period_group = parser.add_argument_group('Periodogram')
    fourier_group = parser.add_argument_group('Fourier')
    outlier_group = parser.add_argument_group('Outlier Detection')
    lasso_group = parser.add_argument_group('Lasso')
    gridsearch_group = parser.add_argument_group('GridSearch')

    general_group.add_argument('-i', '--input', type=str,
        default=stdin,
        help='location of stellar observations '
             '(default = stdin)')
    general_group.add_argument('-o', '--output', type=str,
        default=None,
        help='location of plots, or nothing if no plots are to be generated '
             '(default = None)')
    general_group.add_argument('-f', '--format', type=str,
        default='%.5f',
        help='format specifier for output table')
    general_group.add_argument('--data-extension', type=str,
        default='.dat', metavar='EXT',
        help='extension which follows a star\'s name in data filenames '
             '(default = .dat)')
    general_group.add_argument('--use-cols', type=int, nargs=3,
        default=SUPPRESS, metavar=('TIME', 'MAG', 'MAG_ERR'),
        help='columns to use from data file '
             '(default = 0 1 2)')
    general_group.add_argument('-p', '--processes', type=int,
        default=SUPPRESS, metavar='N',
        help='number of stars to process in parallel '
             '(default = 1)')
    general_group.add_argument('-s', '--scoring', type=str,
        choices=['MSE', 'R2'], default=SUPPRESS,
        help='scoring metric to use '
             '(default = R2)')
    general_group.add_argument('--scoring-cv', type=int,
        default=SUPPRESS, metavar='N',
        help='number of folds in the scoring cross validation '
             '(default = 3)')
    general_group.add_argument('--phase-points', type=int,
        default=100, metavar='N',
        help='number of phase points to output '
             '(default = 100)')
    general_group.add_argument('--min-phase-cover', type=float,
        default=SUPPRESS, metavar='COVER',
        help='minimum fraction of phases that must have points '
             '(default = 0)')
    general_group.add_argument('--matplotlibrc', type=str,
        default=default_matplotlibrc, metavar='RC',
        help='matplotlibrc file to use for formatting plots '
             '(default = $PLOTYPUS_INSTALL/matplotlibrc)')
    period_group.add_argument('--periods', type=FileType('r'),
        default=None,
        help='file of star names and associated periods '
             '(default = None)')
    period_group.add_argument('--min-period', type=float,
        default=SUPPRESS, metavar='P',
        help='minimum period of each star '
             '(default = 0.2)')
    period_group.add_argument('--max-period', type=float,
        default=SUPPRESS, metavar='P',
        help='maximum period of each star '
             '(default = 32.0)')
    period_group.add_argument('--coarse-precision', type=float,
        default=SUPPRESS,
        help='level of granularity on first pass '
             '(default = 0.001)')
    period_group.add_argument('--fine-precision', type=float,
        default=SUPPRESS,
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
    fourier_group.add_argument('--selector',
        choices=['Baart', 'GridSearch'],
        default='GridSearch',
        help='type of model selector to use '
             '(default = GridSearch)')
    outlier_group.add_argument('--sigma', dest='sigma', type=float,
        default=SUPPRESS,
        help='rejection criterion for outliers '
             '(default = 10)')
    outlier_group.add_argument('--sigma-clipping', type=str,
        choices=['standard', 'robust'], default=SUPPRESS,
        help='sigma clipping metric to use '
             '(default = robust)')
    lasso_group.add_argument('--lasso-cv', type=int,
        default=SUPPRESS, metavar='N',
        help='number of folds in the L1-regularization search '
             '(default = 10)')
    lasso_group.add_argument('--max-iter', type=int,
        default=1000, metavar='N',
        help='maximum number of iterations in the Lasso '
             '(default = 1000)')
    
    args = parser.parse_args()

    rcParams = rc_params_from_file(fname=args.matplotlibrc,
                                   fail_on_error=True)
    plotypus.lightcurve.matplotlib.rcParams = rcParams

    regressor_choices = {'Lasso': LassoLarsIC(max_iter=args.max_iter,
                                              fit_intercept=False),
                         'OLS': LinearRegression(fit_intercept=False)}

    selector_choices = {'Baart': None,
                        'GridSearch': GridSearchCV}

    if hasattr(args, 'scoring'):
        scoring_choices = {'R2': 'r2',
                           'MSE': 'mean_squared_error'}

        args.scoring = scoring_choices[args.scoring]

    args.regressor = regressor_choices[args.regressor]
    Selector = selector_choices[args.selector] or GridSearchCV
    args.predictor = make_predictor(Selector=Selector,
                                    use_baart=(args.selector == 'Baart'),
                                    **vars(args))
    args.phases = numpy.arange(0, 1, 1/args.phase_points)

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

    min_degree, max_degree = vars(ops)['fourier_degree']
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
    picklable_ops = {k: vars(ops)[k]
                     for k in vars(ops)
                     if k not in {'input'}}
    # print file header
    print('\t'.join([
        'Name',
        'Period',
        'Shift',
        'Inliers',
        'Outliers',
        'R^2',
        'MSE',
        'A_0',
        'dA_0',
        '\t'.join(map('A_{0}\tPhi_{0}'.format, range(1, max_degree+1))),
        '\t'.join(map('R_{0}1\tphi_{0}1'.format, range(2,max_degree+1))),
        '\t'.join(map('Phase{}'.format, range(ops.phase_points)))
        ])
    )
    printer = _star_printer(max_degree, vars(ops)['format'])
    pmap(process_star, filepaths, callback=printer, **picklable_ops)

def process_star(filename, output, periods={}, **ops):
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
    _period = periods.get(name) if periods is not None and \
                                   name in periods else None
    result = get_lightcurve_from_file(filename, period=_period, **ops)

    if result is not None:
        period, lc, data, coefficients, R_squared, MSE, shift, dA_0 = result
        if output is not None:
            plot_lightcurve(name, lc, period, data, output=output, **ops)
        return name, period, shift, lc, data, coefficients, R_squared, MSE, dA_0

def _star_printer(max_degree, fmt):
    return lambda results: _print_star(results, max_degree, fmt) \
                           if results is not None else None

def _print_star(results, max_degree, fmt):
    if results is None: return
    formatter = lambda x: fmt % x

    name, period, shift, lc, data, coefficients, R2, MSE, dA_0 = results

    N = data[:,0].size # number of data points
    outliers = numpy.ma.count_masked(data[:,0]) # number of outliers
    inliers  = N - outliers # number of inliers

    phase_shifted_coeffs = Fourier.phase_shifted_coefficients(coefficients)
    coefficients_ = numpy.concatenate(([phase_shifted_coeffs[0]], [dA_0],
                                       phase_shifted_coeffs[1:]))
    fourier_ratios = Fourier.fourier_ratios(phase_shifted_coeffs, 1)

    print('\t'.join([name, str(period), str(shift), str(inliers), str(outliers),
                     str(R2), str(MSE)]),
          end='\t')
    print('\t'.join(map(formatter, coefficients_)),
          end='\t')
    trailing_zeros = 2*max_degree + 1 - len(phase_shifted_coeffs)
    if trailing_zeros > 0:
        print('\t'.join(map(formatter,
                            numpy.zeros(trailing_zeros))),
              end='\t')
    print('\t'.join(map(formatter, fourier_ratios)),
          end='\t')
    trailing_zeros = 2*(max_degree-1) - len(fourier_ratios)
    if trailing_zeros > 0:
        print('\t'.join(map(formatter,
                            numpy.zeros(trailing_zeros))),
              end='\t')
    print('\t'.join(map(formatter, lc)))

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
