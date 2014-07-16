import numpy
from sys import exit
from os import path, listdir
from plotypus.lightcurve import get_lightcurve_from_file, plot_lightcurve

def main():
    filename = 'V715Cyg.dat'
    period = 1.
    period, lc, data, *c = get_lightcurve_from_file(filename, period, sigma=100)
    plot_lightcurve(filename, lc, period, data, filetype='.eps', legend=True)

if __name__ == "__main__":
    exit(main())
