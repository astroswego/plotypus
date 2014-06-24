import numpy
from sys import exit, stdin
from os import path, listdir
from argparse import ArgumentError, ArgumentParser, FileType
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.grid_search import GridSearchCV
from plotypus.lightcurve import make_predictor, get_lightcurve, plot_lightcurve
from plotypus.preprocessing import Fourier

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
    parser.add_argument('-p', '--periods', type=FileType('r'),
        default=None, help='file of star names and associated periods')
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

    for filename in get_files(ops.input):
        filename = filename.strip()
        # remove extension from end of filename
        name = filename.split('.')[0]
        try:
            filename_ = filename if path.isfile(filename) \
                                 else path.join(ops.input, filename)
        except AttributeError:
            print("File {} does not exist".format(filename))
            raise Exception
        star = get_lightcurve(filename_,
            period=periods[name] if name in periods else None,
            phases=phases,
            **ops.__dict__)

        if star is not None:
            period, lc, data, coefficients, R_squared = star
            print_star(name, period, R_squared,
                       Fourier.phase_shifted_coefficients(coefficients),
                       max_coeffs, lc, formatter)
            plot_lightcurve(path.basename(filename),
                            lc, period, data, phases=phases,
                            **ops.__dict__)

def get_files(input):
    if input is stdin:
        return input
    elif path.isdir(input):
        return sorted(listdir(input))
    else:
        with open(input, 'r') as f:
            return f.readlines()
            
def print_star(name, period, R_squared,
               coefficients, max_coeffs, lc, formatter):
    print(' '.join([name, str(period), str(R_squared)]), end=' ')
    print(' '.join(map(formatter, coefficients)), end=' ')
    trailing_zeros = max_coeffs - len(coefficients)
    if trailing_zeros > 0:
        print(' '.join(map(formatter, numpy.zeros(trailing_zeros))), end=' ')
    print(' '.join(map(formatter, lc)))
                         
if __name__ == "__main__":
    exit(main())
