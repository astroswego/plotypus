import numpy
import re
from sys import exit, stdin, stderr
from os import path, listdir
from argparse import ArgumentError, ArgumentParser, FileType, SUPPRESS
from sklearn.linear_model import LassoCV, LassoLarsIC, LinearRegression
from sklearn.grid_search import GridSearchCV
from matplotlib import rc_params_from_file
from functools import partial
from itertools import chain, repeat
import plotypus.lightcurve
from plotypus.lightcurve import (make_predictor, get_lightcurve_from_file,
                                 plot_lightcurve)
import plotypus
from plotypus.preprocessing import Fourier
from plotypus.utils import pmap
from plotypus.resources import matplotlibrc

def get_args():
    parser = ArgumentParser()
    general_group  = parser.add_argument_group('General')
    parallel_group = parser.add_argument_group('Parallel')
    period_group   = parser.add_argument_group('Periodogram')
    fourier_group  = parser.add_argument_group('Fourier')
    outlier_group  = parser.add_argument_group('Outlier Detection')
    lasso_group    = parser.add_argument_group('Lasso')


    general_group.add_argument('-i', '--input', type=str,
        default=None,
        help='location of stellar observations '
             '(default = stdin)')
    general_group.add_argument('-o', '--output', type=str,
        default=None,
        help='location of plots, or nothing if no plots are to be generated '
             '(default = None)')
    general_group.add_argument('-f', '--format', type=str,
        default='%.5f',
        help='format specifier for output table')
    general_group.add_argument('--legend', action='store_true',
        help='whether legends should be put on the output plots '
             '(default = False)')
    general_group.add_argument('--extension', type=str,
        default='.dat', metavar='EXT',
        help='extension which follows a star\'s name in data filenames '
             '(default = ".dat")')
    general_group.add_argument('--use-cols', type=int, nargs='+',
        default=SUPPRESS,
        help='columns to use from data file '
             '(default = 0 1 2)')
    general_group.add_argument('-s', '--scoring', type=str,
        choices=['MSE', 'R2'], default=SUPPRESS,
        help='scoring metric to use '
             '(default = "R2")')
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
        default=matplotlibrc,
        metavar='RC',
        help='matplotlibrc file to use for formatting plots '
             '(default file is in plotypus.resources.matplotlibrc)')
    general_group.add_argument('-v', '--verbosity', type=str, action='append',
        default=[], choices=['all', 'coverage', 'outlier', 'period'],
        metavar='OPERATION',
        help='specifies an operation to print verbose output for, or '
             '"all" to print all verbose output '
             '(default = None)')
    parallel_group.add_argument('--star-processes', type=int,
        default=1, metavar='N',
        help='number of stars to process in parallel '
             '(default = 1)')
    parallel_group.add_argument('--selector-processes', type=int,
        default=SUPPRESS, metavar='N',
        help='number of processes to use for each selector '
             '(default depends on selector used)')
    parallel_group.add_argument('--scoring-processes', type=int,
        default=SUPPRESS, metavar='N',
        help='number of processes to use for scoring, if not done by selector '
             '(default = 1)')
    period_group.add_argument('--periods', type=str,
        default=None,
        help='file of star names and associated periods, or a single period '
             'to use for all stars '
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
             '(default = 0.00001)')
    period_group.add_argument('--fine-precision', type=float,
        default=SUPPRESS,
        help='level of granularity on second pass '
             '(default = 0.000000001)')
    period_group.add_argument('--periodogram', type=str,
        choices=['Lomb_Scargle', 'conditional_entropy'],
        default=SUPPRESS,
        help='method for determining period '
             '(default = Lomb-Scargle)')
    fourier_group.add_argument('-d', '--fourier-degree', type=int, nargs=2,
        default=(2,20), metavar=('MIN', 'MAX'),
        help='range of degrees of fourier fits to use '
             '(default = 2 20)')
    fourier_group.add_argument('-r', '--regressor',
        choices=['Lasso', 'OLS'],
        default='Lasso',
        help='type of regressor to use '
             '(default = "Lasso")')
    fourier_group.add_argument('--selector',
        choices=['Baart', 'GridSearch'],
        default='GridSearch',
        help='type of model selector to use '
             '(default = "GridSearch")')
    outlier_group.add_argument('--sigma', dest='sigma', type=float,
        default=SUPPRESS,
        help='rejection criterion for outliers '
             '(default = 20)')
    outlier_group.add_argument('--sigma-clipping', type=str,
        choices=['standard', 'robust'], default=SUPPRESS,
        help='sigma clipping metric to use '
             '(default = "robust")')
    lasso_group.add_argument('--max-iter', type=int,
        default=1000, metavar='N',
        help='maximum number of iterations in the Lasso '
             '(default = 1000)')
    
    args = parser.parse_args()

    if args.output is not None:
        rcParams = rc_params_from_file(fname=args.matplotlibrc,
                                       fail_on_error=args.output)
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
        try:
            float(args.periods)
        except ValueError:
            with open(args.periods, 'r') as f:
                args.periods = {name: float(period)
                                for (name, period) in (line.strip().split()
                                for line in args.periods if ' ' in line)}
    return args

def main():
    ops = get_args()

    min_degree, max_degree = ops.fourier_degree
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
    print(*['Name',
            'Period',
            'Shift',
            'Coverage',
            'Inliers',
            'Outliers',
            'R^2',
            'MSE',
            'Degree',
            'A_0',
            'dA_0',
            '\t'.join(map('A_{0}\tPhi_{0}'.format, range(1, max_degree+1))),
            '\t'.join(map('R_{0}1\tphi_{0}1'.format, range(2, max_degree+1))),
            '\t'.join(map('Phase{}'.format, range(ops.phase_points)))],
        sep='\t')

    printer = lambda result: _print_star(result, max_degree, ops.format) \
                             if result is not None else None
    pmap(process_star, filepaths, callback=printer,
         processes=ops.star_processes, **picklable_ops)

def process_star(filename, output, periods={}, **ops):
    """Processes a star's lightcurve, prints its coefficients, and saves
    its plotted lightcurve to a file. Returns the result of get_lightcurve.
    """
    _name = path.basename(filename)
    extension = ops['extension']
    if _name.endswith(extension):
        name = _name[:-len(extension)]
    else:
        # file has wrong extension
        return
    try:
        _period = float(periods)
    except TypeError:
        _period = periods.get(name) if periods is not None and \
                                       name in periods else None
    
    result = get_lightcurve_from_file(filename, name=name, period=_period,
                                      **ops)
    if result is None:
        return
    if output is not None:
        plot_lightcurve(name, result['lightcurve'], result['period'],
                        result['phased_data'], output=output, **ops)
    return result


def _print_star(result, max_degree, fmt):
    if result is None: return

    # function which formats every number in a sequence according to fmt
    format_all = partial(map, lambda x: fmt % x)

    # count inliers and outliers
    points   = result['phased_data'][:,0].size
    outliers = numpy.ma.count_masked(result['phased_data'][:, 0])
    inliers  = points - outliers

    # get fourier coefficients and compute ratios
    coefs  = Fourier.phase_shifted_coefficients(result['coefficients'])
    _coefs = numpy.concatenate(([coefs[0]],
                               [result['dA_0']],
                               coefs[1:]))
    fourier_ratios = Fourier.fourier_ratios(coefs, 1)

    # create the vectors of zeroes
    coef_zeros  = repeat('0', times=(2*max_degree + 1 - len(coefs)))
    ratio_zeros = repeat('0', times=(2*(max_degree - 1) - len(fourier_ratios)))

    # print the entry for the star with tabs as separators
    # and itertools.chain to separate the different results into a
    # continuous list which is then unpacked
    print(*chain(*[[result['name']],
                   map(str,
                       [result['period'], result['shift'], result['coverage'],
                        inliers, outliers,
                        result['R2'],     result['MSE'],   result['degree']]),
                   # coefficients and fourier ratios with trailing zeros
                   # formatted defined by the user-provided fmt string
                   format_all(_coefs),         coef_zeros,
                   format_all(fourier_ratios), ratio_zeros,
                   format_all(result['lightcurve'])]),
        sep='\t')

def _get_files(input):
    if input is None:
        return stdin
    elif input[0] == "@":
        with open(input[1:], 'r') as f:
            return map(lambda x: x.strip(), f.readlines())
    elif path.isfile(input):
        return [input]
    elif path.isdir(input):
        return sorted(listdir(input))
    else:
        raise FileNotFoundError('file {} not found'.format(input))

if __name__ == "__main__":
    exit(main())
