import numpy
from sys import exit, stdin
from os import path, listdir
from optparse import OptionParser
from plotypus.lightcurve import get_lightcurve, plot_lightcurve

def get_ops():
    parser = OptionParser()
    parser.add_option('-i', '--input', type='string',
        default=path.join('..', 'data', 'lmc', 'i', 'cep', 'f'),
        help='location of stellar observations',)
    parser.add_option('-o', '--output', type='string',
        default=path.join('..', 'results'),
        help='location of results')
    parser.add_option('-f', '--format', type='string',
        default='%.5f',
        help='format specifier for output table')
    parser.add_option('-p', '--periods', type='string',
        default=None, help='file of star names and associated periods')
# This might not be the best way to implement this option.
# Maybe use --phase-step instead.
#    parser.add_option('--num-phase', type='int',
#        default=100,
#        help='number of phase points to use')
    parser.add_option('--min_period', dest='min_period', type='float',
        default=0.2, help='minimum period of each star')
    parser.add_option('--max_period', dest='max_period', type='float',
        default=32., help='maximum period of each star')
    parser.add_option('--coarse_precision', dest='coarse_precision', type='int',
        default=0.001, help='level of granularity on first pass')
    parser.add_option('--fine_precision', dest='fine_precision', type='int',
        default=0.0000001, help='level of granularity on second pass')
    parser.add_option('--fourier_degree', dest='fourier_degree', type='int',
        default=15, help='number of coefficients to generate')
    parser.add_option('--sigma', dest='sigma', type='float',
        default=10, help='rejection criterion for outliers')
    parser.add_option('--cv',               dest='cv', type='int',
        default=10, help='number of folds in the L1-regularization search')
    parser.add_option('--min_phase_cover', dest='min_phase_cover', type='float',
        default=1/2., help='minimum fraction of phases that must have points')
    (options, args) = parser.parse_args()
    return options

def main():
    ops = get_ops()
    if ops.periods is not None:
        periods = {name: float(period) for (name, period)
                   in (line.split() for line
                       in open(ops.periods, 'r') if ' ' in line)}

    print(' '.join([
        '#',
        'Name',
        'Period',
        'A_0',
        ' '.join(['a_{0} b_{0}'.format(i)
                  for i in range(1, ops.fourier_degree+1)]),
        ' '.join(['Phase{}'.format(i) for i in range(100)])
        ])
    )

    formatter = lambda x: ops.format % x    
    max_coeffs = 2*ops.fourier_degree+1

    for filename in sorted(listdir(ops.input)):
        name = filename.split('.')[0]
        star = get_lightcurve(path.join(ops.input, filename),
            period=periods[name] if name in periods else None,
            **ops.__dict__)

        if star is not None:
            print_star(star, name, formatter, max_coeffs)
            period, lc, data, coeff = star
#            lcs += [[period] + list(lc)]
            plot_lightcurve(filename, lc, period, data, **ops.__dict__)

    # numpy.savetxt(path.join(ops.output, 'lightcurves.dat'),
    #               numpy.array(lcs), fmt=ops.format,
    #               header='Period ' + \
    #                      ' '.join(['Phase' + str(i) for i in range(100)]))

def print_star(star, name, formatter, max_coeffs):
    period, lc, data, coeff = star
    
    print(' '.join([name, str(period)]), end=' ')
    print(' '.join(map(formatter, coeff)), end=' ')
    trailing_zeros = max_coeffs - len(coeff)
    if trailing_zeros > 0:
        print(' '.join(map(formatter, numpy.zeros(trailing_zeros))))
    print(' '.join(map(formatter, lc)))
                         
if __name__ == "__main__":
    exit(main())
