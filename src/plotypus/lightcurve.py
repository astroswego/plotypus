import numpy
from sys import exit
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
    parser.add_option('-p', '--periods', type='string',
        default=None, help='file of star names and associated periods')
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
        default=6, help='rejection criterion for outliers')
    parser.add_option('--cv',               dest='cv', type='int',
        default=10, help='number of folds in the L1-regularization search')
    parser.add_option('--min_phase_cover', dest='min_phase_cover', type='float',
        default=1/2., help='minimum fraction of phases that must have points')
    (options, args) = parser.parse_args()
    return options

def main():
    ops = get_ops()
    lcs = []
    if ops.periods is not None:
        periods = {name: float(period) for (name, period)
                   in (line.split() for line
                       in open(ops.periods, 'r') if ' ' in line)}
    for filename in sorted(listdir(ops.input)):
        name = filename.split('.')[0]
        print(filename)
        star = get_lightcurve(path.join(ops.input, filename),
            period=periods[name] if name in periods else None,
            **ops.__dict__)
    
        if star is not None:
            period, lc, data = star
            lcs += [[period] + list(lc)]
            plot_lightcurve(filename, lc, period, data, **ops.__dict__)
    
    numpy.savetxt(path.join(ops.output, 'lightcurves.dat'),
                  numpy.array(lcs), fmt='%.5f',
                  header='Period ' + \
                         ' '.join(['Phase' + str(i) for i in range(100)]))

if __name__ == "__main__":
    exit(main())
