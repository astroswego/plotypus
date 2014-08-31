import numpy
from sys import exit
from os import path
import numpy as np
from plotypus.lightcurve import get_lightcurve_from_file, make_predictor
from plotypus.preprocessing import Fourier
from plotypus.periodogram import rephase
from plotypus.utils import colvec
from sklearn.linear_model import LinearRegression, LassoCV, LassoLarsIC
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc_file
rc_file('matplotlibrc')
import matplotlib.pyplot as plt

color = True

def main():
    directory = path.join('data', 'I')
    name = 'OGLE-BLG-RRLYR-13317'#'OGLE-LMC-CEP-0209'
    filename = name+'.dat'
    p = 0.4986531#3.1227238
    X_true = numpy.arange(0, 1, 0.001)
    output = get_lightcurve_from_file(path.join(directory, filename),
                                      period=p, phases=X_true)
    las = output['lightcurve']
    data = output['phased_data']
    ols = get_lightcurve_from_file(path.join(directory, filename),
                                   period=p, phases=X_true,
            predictor=make_predictor(LinearRegression(), use_baart=True)
        )['lightcurve']
    
    ax = plt.gca()
    
    fd, = plt.plot(np.hstack((X_true,1+X_true)), np.hstack((ols, ols)), 
                   linewidth=2.5, color='darkred' if color else 'black', 
                   ls='dashed')
    
    lasso, = plt.plot(np.hstack((X_true,1+X_true)), np.hstack((las, las)), 
                      color='black',
                      linewidth=1.5, ls='solid')
    
    sc = plt.errorbar(np.hstack((data.T[0],1+data.T[0])),
                      np.hstack((data.T[1],  data.T[1])),
                      np.hstack((data.T[2],  data.T[2])),
                      color='darkblue' if color else 'black',
                      ls='None', ms=1, mew=1, capsize=0)
    
    plt.legend([sc, fd, lasso],
               ["Data", "Baart", "Lasso"],
               loc='best')
    
    plt.xlim(0,2)
    ax.grid(False)
    ax.invert_yaxis()
    plt.xlabel('Phase ({0:0.7} day period)'.format(p))
    plt.ylabel('Magnitude')
    plt.title(name)
    plt.tight_layout(pad=0.1)
    plt.savefig('gaps.eps')
    plt.clf()

if __name__ == '__main__':
    exit(main())
