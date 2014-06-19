import numpy
from sys import exit, stdin
from os import path, listdir
from argparse import ArgumentParser, FileType
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.grid_search import GridSearchCV
from plotypus.lightcurve import get_lightcurve, plot_lightcurve
from plotypus.preprocessing import Fourier

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
        default=path.join('..', 'data', 'lmc', 'i', 'cep', 'f'),
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
        default=32.,
        help='maximum period of each star')
    parser.add_argument('--coarse-precision', type=int,
        default=0.001,
        help='level of granularity on first pass')
    parser.add_argument('--fine-precision', type=int,
        default=0.0000001,
        help='level of granularity on second pass')
    parser.add_argument('--fourier-degree', type=int,
        default=15,
        help='number of coefficients to generate')
    parser.add_argument('-m', '--model',
        choices=['LassoCV', 'OLS'],
        default='LassoCV',
        help='type of model to use')
    parser.add_argument('--predictor',
        choices=['Baart', 'GridSearchCV'],
        default='GridSearchCV',
        help='type of model predictor to use')
    parser.add_argument('--sigma', dest='sigma', type=float,
        default=10,
        help='rejection criterion for outliers')
    parser.add_argument('--cv', type=int,
        default=10,
        help='number of folds in the L1-regularization search')
    parser.add_argument('--max-iter', type=int,
        default=1000,
        help='maximum number of iterations in the LassoCV')
    parser.add_argument('--min-phase-cover', type=float,
        default=1/2.,
        help='minimum fraction of phases that must have points')
    args = parser.parse_args()

    model_choices = {'LassoCV' : LassoCV, 'OLS' : LinearRegression}
    predictor_choices = {'Baart' : None, 'GridSearchCV' : GridSearchCV}

    args.model = model_choices[args.model]
    args.selector = predictor_choices[args.predictor]

    return args

def main():
    ops = get_args()
    if ops.periods is not None:
        periods = {name: float(period) for (name, period)
                   in (line.split() for line
                       in ops.periods if ' ' in line)}
        ops.periods.close()

    print(' '.join([
        '#',
        'Name',
        'Period',
        'R^2',
        'A_0',
        ' '.join(['A_{0} Phi_{0}'.format(i)
                  for i in range(1, ops.fourier_degree+1)]),
        ' '.join(['Phase{}'.format(i) for i in range(ops.phase_points)])
        ])
    )

    formatter = lambda x: ops.format % x    
    max_coeffs = 2*ops.fourier_degree+1
    phases=numpy.arange(0, 1, 1/ops.phase_points)

    file_list = get_files(ops.input)

    for filename in sorted(listdir(ops.input)):
        # remove extension from end of filename
        name = filename.split('.')[0]
        ## Doing the following instead would allow for naming schemes which
        ## include a dot in the star's ID. However, it wouldn't behave right
        ## for filetypes with two extensions, such as .tar.gz. Instead we
        ## should take a filetype parameter, and strip that from the end.
        # name = '.'.join(filename.split('.')[:-1])
        star = get_lightcurve(path.join(ops.input, filename),
            period=periods[name] if name in periods else None,
            phases=phases,
            **ops.__dict__)

        if star is not None:
            period, lc, data, coefficients, R_squared = star
            print_star(name, period, R_squared,
                       Fourier.phase_shifted_coefficients(coefficients),
                       max_coeffs, lc, formatter)
            plot_lightcurve(filename, lc, period, data, phases=phases,
                            **ops.__dict__)

def get_files(input):
    
            
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
