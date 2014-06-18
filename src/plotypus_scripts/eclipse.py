import numpy
from sys import exit
from os import path, listdir
from plotypus.lightcurve import get_lightcurve, plot_lightcurve

def main():
    """http://ogledb.astrouw.edu.pl/~ogle/CVS/getobj.php?starcat=OGLE-LMC-CEP-0227&sqlsrv=localhost&database=cvs&sqldb=lmc_cepheids&target=lmc&qtype=ceph"""
    data = path.join('..', 'stellar', 'data', 'lmc', 'i', 'cep', 'f')
    filename = 'OGLE-LMC-CEP-0227.dat'
    period = 3.7970428
    period, lc, data = get_lightcurve(path.join(data, filename), period)
    plot_lightcurve(filename, lc, period, data, filetype='.eps', legend=True)

if __name__ == "__main__":
    exit(main())
