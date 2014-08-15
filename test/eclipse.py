import numpy
from sys import exit
from os import path, listdir
from plotypus.lightcurve import get_lightcurve_from_file, plot_lightcurve

def main():
    """http://ogledb.astrouw.edu.pl/~ogle/CVS/getobj.php?starcat=OGLE-LMC-CEP-0227&sqlsrv=localhost&database=cvs&sqldb=lmc_cepheids&target=lmc&qtype=ceph"""
    data = path.join('..', 'data', 'I')
    filename = 'OGLE-LMC-CEP-0227.dat'
    p = 3.7970428
    p, lc, data, *c = get_lightcurve_from_file(path.join(data, filename), p)
    plot_lightcurve(filename, lc, p, data, filetype='.eps',
                    legend=True, color=False)

if __name__ == "__main__":
    exit(main())
