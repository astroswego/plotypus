import numpy
numpy.random.seed(0)
from sys import exit
from os import path, listdir
from plotypus.lightcurve import make_predictor
from plotypus.periodogram import rephase
from plotypus.utils import colvec, mad
from plotypus.preprocessing import Fourier
from sklearn.linear_model import LinearRegression, LassoLarsIC
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.pipeline import Pipeline
from scipy.stats import sem

def run_trial(predictor, stars, train_size, n_iter=1000):
    return numpy.fromiter(
        (numpy.median(cross_val_score(predictor, colvec(data.T[0]), data.T[1],
          n_jobs=-1, cv=ShuffleSplit(len(data), n_iter, train_size=train_size)))
         for data in stars), dtype=float)

def main():
    periods_file = 'OGLE-periods.dat'
    periods = {name: float(period) for (name, period)
               in (line.strip().split()
                   for line in open(periods_file, 'r') if ' ' in line)}
    
    directory = path.join('data', 'good')
    filenames = listdir(directory)
    stars = [rephase(numpy.loadtxt(path.join(directory, filename)),
                     periods[filename.split('.')[0]])
             for filename in filenames]
    
    baart = make_predictor(regressor=LinearRegression(), use_baart=True)
    min_samples, max_samples = 5, 50
    for train_size in range(min_samples, max_samples):
        lasso = make_predictor(scoring_cv=train_size if train_size < 15 else 3)
        lassoR2s = run_trial(lasso, stars, train_size)
        baartR2s = run_trial(baart, stars, train_size)
        avg_Lasso = numpy.mean(lassoR2s)
        avg_Baart = numpy.mean(lassoR2s)
        spread_Lasso = sem(lassoR2s)
        spread_Baart = sem(baartR2s)
        output = (train_size, avg_Lasso, spread_Lasso, avg_Baart, spread_Baart)
        if avg_Lasso - spread_Lasso > avg_Baart + spread_Baart: # Lasso wins
            print('%d & \\textbf{%.4f $\pm$ %.4f} & %.4f $\pm$ %.4f \\\\' % output)
        elif avg_Baart - spread_Baart > avg_Lasso + spread_Lasso: # Baart wins
            print('%d & %.4f $\pm$ %.4f & \\textbf{%.4f $\pm$ %.4f} \\\\' % output)
        else: # Tie
            print('%d & %.4f $\pm$ %.4f & %.4f $\pm$ %.4f \\\\' % output)

if __name__ == '__main__':
    exit(main())
