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
from matplotlib import rcParams
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']#['Latin Modern']
rcParams['text.usetex'] = True
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
import matplotlib.pyplot as plt

color = True

def main():
    directory = path.join('..', 'data', 'I')
    filename = 'OGLE-LMC-CEP-0209.dat'
    p = 3.1227238
    X_true = numpy.arange(0, 1, 0.001)
    _, las, data, *c = get_lightcurve_from_file(path.join(directory, filename),
        p, phases=X_true)
    _, ols, *c = get_lightcurve_from_file(path.join(directory, filename),
        p, phases=X_true, predictor=make_predictor(LinearRegression(),
                                                   use_baart=True))
    
    ax = plt.gca()
    
    fd, = plt.plot(np.hstack((X_true,1+X_true)), np.hstack((ols, ols)), 
                   linewidth=1, color='darkred' if color else 'black', 
                   ls='dashed')
    
    lasso, = plt.plot(np.hstack((X_true,1+X_true)), np.hstack((las, las)), 
                      color='black',
                      linewidth=1, ls='solid')
    
    sc = plt.errorbar(np.hstack((data.T[0],1+data.T[0])),
                      np.hstack((data.T[1],  data.T[1])),
                      np.hstack((data.T[2],  data.T[2])),
                      color='darkblue' if color else 'black',
                      ls='None', ms=1, mew=1, capsize=1)
    
    plt.legend([sc, fd, lasso],
               ["Data", "FD", "Lasso"],
               loc='best')
    
    plt.xlim(0,2)
    ax.grid(False)
    ax.invert_yaxis()
    plt.xlabel('Phase ({0:0.7} day period)'.format(p))
    plt.ylabel('Magnitude')
    plt.title('OGLE-LMC-CEP-0209')
    plt.tight_layout(pad=0.1)
    plt.savefig('OGLE-LMC-CEP-0209-comparison.eps')
    plt.clf()

if __name__ == '__main__':
    exit(main())
