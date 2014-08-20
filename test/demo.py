from sys import exit
import numpy as np
np.random.seed(4) # chosen by fair dice roll. guaranteed to be random.
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.pipeline import Pipeline
from plotypus.preprocessing import Fourier
from plotypus.utils import colvec
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Latin Modern']
rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

color = True

def lc(X):
    return 10 + np.cos(2*np.pi*X) + 0.1*np.cos(18*np.pi*X)

def main():
    X_true = np.linspace(0, 1, 1001)
    y_true = lc(X_true)
    
    n_samples = 50
    X_sample = np.random.uniform(size=n_samples)
    y_sample = lc(X_sample) + np.random.normal(0, 0.1, n_samples)
    
    predictor = Pipeline([('Fourier', Fourier(9)),
                          ('OLS',   LinearRegression())])
    predictor = predictor.fit(colvec(X_sample), y_sample)
    y_pred = predictor.predict(colvec(X_true))
    
    predictor = Pipeline([('Fourier', Fourier(9)),
                          ('Lasso',   LassoCV())])
    predictor = predictor.fit(colvec(X_sample), y_sample)
    y_lasso = predictor.predict(colvec(X_true))
    
    ax = plt.gca()
    signal, = plt.plot(np.hstack((X_true,1+X_true)),
                       np.hstack((y_true, y_true)), 
                       linewidth=0.66, color='black')
    
    fd, = plt.plot(np.hstack((X_true,1+X_true)),
                   np.hstack((y_pred, y_pred)), 
                   linewidth=2.5, ls='dashed',
                   color='darkred' if color else 'black')
    
    lasso, = plt.plot(np.hstack((X_true,1+X_true)),
                      np.hstack((y_lasso, y_lasso)), 
                      linewidth=3, color='black', ls='dotted')
    
    sc = plt.scatter(np.hstack((X_sample,1+X_sample)),
                     np.hstack((y_sample, y_sample)),
                     marker='+', s=20,
                     color='darkblue' if color else 'black')
    
    plt.legend([signal, sc, fd, lasso],
               ["True Signal", "Noisy Data", "OLS", "Lasso"],
               loc='best')
    
    plt.xlim(0,2)
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.title('Simulated Lightcurve Example')
    plt.tight_layout(pad=0.1)
    plt.savefig('demo.eps')
    plt.clf()

if __name__ == '__main__':
    exit(main())
