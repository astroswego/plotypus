import numpy
numpy.random.seed(0)
from sys import exit
from os import path, listdir
from plotypus.lightcurve import make_predictor
from plotypus.periodogram import rephase
from plotypus.utils import colvec, mad
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from scipy.stats import sem

def run_trial(predictor, stars, train_size, n_iter=100):
    return numpy.fromiter(
        (numpy.median(cross_val_score(predictor, colvec(data.T[0]), data.T[1],
         n_jobs=8, cv=ShuffleSplit(len(data), n_iter, train_size=train_size)))
         for data in stars), dtype=float)

def main():
    periods_file = path.join('data', 'OGLE-periods.dat')
    periods = {name: float(period) for (name, period)
               in (line.strip().split()
                   for line in open(periods_file, 'r') if ' ' in line)}
    
    directory = path.join('data', 'good')
    filenames = listdir(directory)
    stars = [rephase(numpy.loadtxt(path.join(directory, filename)),
                     periods[filename.split('.')[0]])
             for filename in filenames]
    
    min_samples, max_samples = 5, 51
    for train_size in range(min_samples, max_samples):
        lasso = make_predictor(scoring_cv=train_size)
        lassoR2s = run_trial(lasso, stars, train_size)
        avg_Lasso = numpy.mean(lassoR2s)
        spread_Lasso = sem(lassoR2s)
        print('%d & %.4f $\pm$ %.4f \\\\' % output)

if __name__ == '__main__':
    exit(main())
