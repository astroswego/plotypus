import numpy
numpy.random.seed(0)
from sys import exit
from os import path, listdir
from plotypus.lightcurve import make_predictor
from plotypus.periodogram import rephase
from plotypus.utils import colvec, mad, get_periods
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from scipy.stats import sem

def run_trial(predictor, stars, train_size, n_iter=100):
    return numpy.fromiter(
        (numpy.median(
            cross_val_score(predictor, colvec(data.T[0]), data.T[1],
                            n_jobs=-1, cv=ShuffleSplit(len(data),
                            n_iter, train_size=train_size)))
         for data in stars),
        dtype=float)

def main():
    periods = get_periods()
    directory = path.join('data', 'good')
    filenames = listdir(directory)
    stars = [rephase(numpy.loadtxt(path.join(directory, filename)),
                     periods[filename.split('.')[0]])
             for filename in filenames]
    print("""\\begin{table}
  \\begin{center}
    \\begin{tabular}{ c c c c c }
N & Median $R^2$ & MAD & Mean $R^2$ & SD \\\\ \\hline""")
    min_samples, max_samples = 5, 50
    for train_size in range(min_samples, max_samples+1):
        lasso = make_predictor(scoring_cv=train_size)
        lassoR2s = run_trial(lasso, stars, train_size)
        median_lasso = numpy.median(lassoR2s)
        mad_lasso = mad(lassoR2s)
        mean_lasso = numpy.mean(lassoR2s)
        sd_lasso = numpy.std(lassoR2s)
        output = (train_size, median_lasso, mad_lasso, mean_lasso, sd_lasso)
        print('%d & %.4f & %.4f & %.4f & %.4f \\\\' % output)
    
    print("""\\hline
     \\end{tabular}
     \\caption{\label{tab:sampler}  }
  \\end{center}
\\end{table}""")

if __name__ == '__main__':
    exit(main())
