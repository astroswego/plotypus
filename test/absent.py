import numpy
from sys import exit
from os import path
import numpy as np
from math import floor, ceil
from plotypus.lightcurve import get_lightcurve_from_file, make_predictor
from plotypus.preprocessing import Fourier
from plotypus.periodogram import rephase
from plotypus.utils import colvec, get_signal, get_noise
from plotypus.resources import matplotlibrc
from sklearn.linear_model import LinearRegression, LassoCV, LassoLarsIC
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc_file, rcParams
rc_file(matplotlibrc)
rcParams['figure.figsize'] = [6.97, 9.23]
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

color = True

def main():
    periods_file = path.join('data', 'OGLE-periods.dat')
    periods = {name: float(period) for (name, period)
               in (line.strip().split()
                   for line in open(periods_file, 'r') if ' ' in line)}
    directory = path.join('data', 'I')
    for val, name in enumerate(
        ["OGLE-BLG-RRLYR-04946", "OGLE-BLG-RRLYR-11479",
         "OGLE-LMC-CEP-0578", "OGLE-LMC-CEP-0804",
         "OGLE-LMC-CEP-0951", "OGLE-LMC-CEP-1370",
         "OGLE-LMC-CEP-1416", "OGLE-LMC-CEP-1528",
         "OGLE-LMC-CEP-2254", "OGLE-LMC-CEP-2434",
         "OGLE-LMC-CEP-2446", "OGLE-LMC-CEP-3268",
         "OGLE-SMC-CEP-0286", "OGLE-SMC-CEP-2991",
         "OGLE-SMC-CEP-3251", "OGLE-SMC-CEP-4495"]):
        filename = name+'.dat'
        p = periods[name]
        X_true = numpy.arange(0, 1, 0.001)
        output = get_lightcurve_from_file(path.join(directory, filename),
                     period=p, phases=X_true)
        las = output['lightcurve']
        data = output['phased_data']
        
        plt.subplot(8,2,(1+val))#plt.subplot(4,2,(1+val)%8)
        ax = plt.gca()
        
        sc = plt.errorbar(np.hstack((data.T[0],1+data.T[0])),
                          np.hstack((data.T[1],  data.T[1])),
                          np.hstack((data.T[2],  data.T[2])),
                          color='darkblue' if color else 'black',
                          ls='None', ms=1, mew=1, capsize=0)
     
        errs = get_noise(data)
        sc = plt.errorbar(np.hstack((errs.T[0],1+errs.T[0])),
                          np.hstack((errs.T[1],  errs.T[1])),
                          np.hstack((errs.T[2],  errs.T[2])),
                          color='darkred' if color else 'black',
                          ls='None', ms=1, mew=1, capsize=0)
     
        lasso, = plt.plot(np.hstack((X_true,1+X_true)), np.hstack((las, las)), 
                          color='black', linewidth=1, ls='solid')
     
        plt.xlim(0,2)
        ax.grid(False)
        ax.invert_yaxis()
        plt.xlabel('Phase ({0:0.7} day period)'.format(p))
        plt.ylabel('Magnitude')
        plt.title(name)
        plt.tight_layout(pad=0.1)
        ymin = floor(min(data.T[1]) * 10) / 10
        ymax = ceil(max(data.T[1]) * 10) / 10
        #plt.yticks(np.round([ymin, output['coefficients'][0], ymax], 1))
        plt.yticks(np.round(np.linspace(ymin, ymax, 3), 1))
        if val == 15:
            plt.savefig(path.join('results', 'absents'+str(val)+'.eps'))
            plt.clf()

if __name__ == '__main__':
    exit(main())
