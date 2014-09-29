import numpy
from sys import exit
from os import path
import numpy as np
from plotypus.lightcurve import get_lightcurve_from_file, make_predictor
from plotypus.preprocessing import Fourier
from plotypus.periodogram import rephase
from plotypus.utils import colvec, get_noise
from plotypus.resources import matplotlibrc
from sklearn.linear_model import LinearRegression, LassoCV, LassoLarsIC
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
from matplotlib import rc_file
rc_file(matplotlibrc)
import matplotlib.pyplot as plt

color = True

def main():
    directory = path.join('data', 'I')
    name = 'OGLE-LMC-CEP-0227'
    filename = name+'.dat'
    p = 3.7970428
    X_true = numpy.arange(0, 1, 0.001)
    output = get_lightcurve_from_file(path.join(directory, filename),
                 period=p, phases=X_true, sigma_clipping='standard', sigma=7,
                 predictor=make_predictor(LinearRegression(), use_baart=True,
                                          fourier_degree=(2,15)))
    fit = output['lightcurve']
    data = output['phased_data']
    
    f, axarr = plt.subplots(2, sharex=True)
    
    #axarr[0].ylabel('Magnitude')
    #f.text(0.02, 0.5, 'Magnitude',
    #                ha='center', va='center', rotation='vertical')
    axarr[0].set_title('Standard and Robust Outlier Detection for ' + name)
    axarr[0].set_ylabel("Magnitude")
    axarr[0].invert_yaxis()
    
    fd, = axarr[0].plot(np.hstack((X_true,1+X_true)), np.hstack((fit, fit)), 
                   linewidth=2, color='darkred' if color else 'black', 
                   ls='dashed')
    
    sc = axarr[0].errorbar(np.hstack((data.T[0],1+data.T[0])),
                      np.hstack((data.T[1],  data.T[1])),
                      np.hstack((data.T[2],  data.T[2])),
                      color='darkblue' if color else 'black',
                      ls='None', ms=1, mew=1, capsize=0)
    
    errs = get_noise(data)
    er = axarr[0].errorbar(np.hstack((errs.T[0],1+errs.T[0])),
                      np.hstack((errs.T[1],  errs.T[1])),
                      np.hstack((errs.T[2],  errs.T[2])),
                      color='darkred' if color else 'black',
                      ls='None', ms=1, mew=1, capsize=0)
    
    #plt.legend([fd, sc, er],
    #           ["Standard Baart", "Inliers", "Outliers"],
    #           loc='best')
    
    plt.xlim(0,2)
    plt.xlabel('Phase ({0:0.7} day period)'.format(p))
    #plt.ylabel('Magnitude')
    #plt.title(name)
    plt.tight_layout(pad=0.1)
    
    #plt.subplot(1,2,2)
    axarr[1].invert_yaxis()
    axarr[1].set_ylabel("Magnitude")
    
    output = get_lightcurve_from_file(path.join(directory, filename),
                                      period=p, phases=X_true, sigma=7)
    fit = output['lightcurve']
    data = output['phased_data']
    
    lasso, = axarr[1].plot(np.hstack((X_true,1+X_true)), np.hstack((fit, fit)), 
                      color='black',
                      linewidth=1, ls='solid')
    
    sc = axarr[1].errorbar(np.hstack((data.T[0],1+data.T[0])),
                      np.hstack((data.T[1],  data.T[1])),
                      np.hstack((data.T[2],  data.T[2])),
                      color='darkblue' if color else 'black',
                      ls='None', ms=1, mew=1, capsize=0)
    
    errs = get_noise(data)
    er = axarr[1].errorbar(np.hstack((errs.T[0],1+errs.T[0])),
                      np.hstack((errs.T[1],  errs.T[1])),
                      np.hstack((errs.T[2],  errs.T[2])),
                      color='darkred' if color else 'black',
                      ls='None', ms=1, mew=1, capsize=0)
    
    axarr[1].legend([fd, lasso, sc, er],
               ["Standard Baart", "Robust Lasso", "Inliers", "Outliers"],
               loc='best')
    
    plt.xlim(0,2)
    plt.xlabel('Phase ({0:0.7} day period)'.format(p))
    #plt.ylabel(' ')
    plt.tight_layout(pad=0.1)
    plt.savefig('eclipse.eps')
    plt.clf()

if __name__ == '__main__':
    exit(main())
