import numpy
from sys import exit
from os import path
import numpy as np
np.random.seed(0)
from plotypus.lightcurve import *
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
    
    ax = plt.gca()
    
    trials = []
    points = []
    n_trials = 10
    n_samples = 41
    for trial in range(n_trials):
        sample = data[numpy.random.choice(range(len(data)), n_samples)]
        _, trial, *c = get_lightcurve(sample, 1., phases=X_true)
        t, = plt.plot(np.hstack((X_true,1+X_true)),
                      np.hstack((trial, trial)), color="grey", ls="dotted")
        #points.append(plt.errorbar(np.hstack((sample.T[0],1+sample.T[0])),
        #                           np.hstack((sample.T[1],  sample.T[1])),
        #                           np.hstack((sample.T[2],  sample.T[2])),
        #                           ls='None', ms=1, mew=1, capsize=1))
    
    lasso, = plt.plot(np.hstack((X_true,1+X_true)), np.hstack((las, las)), 
                      color='black', linewidth=1, ls='solid')
    
    sc = plt.errorbar(np.hstack((data.T[0],1+data.T[0])),
                      np.hstack((data.T[1],  data.T[1])),
                      np.hstack((data.T[2],  data.T[2])),
                      color='darkblue' if color else 'black',
                      ls='None', ms=1, mew=1, capsize=1)
    
    plt.legend([sc, lasso, t],
               ["All Data", "Lasso fit on all data",
                "Lasso fit on %i random points" % n_samples],
               loc='best')
    
    plt.xlim(0,2)
    ax.grid(False)
    ax.invert_yaxis()
    plt.xlabel('Phase ({0:0.7} day period)'.format(p))
    plt.ylabel('Magnitude')
    plt.title('OGLE-LMC-CEP-0209')
    plt.tight_layout(pad=0.1)
    plt.savefig('OGLE-LMC-CEP-0209-sample.png')
    plt.clf()

if __name__ == '__main__':
    exit(main())
