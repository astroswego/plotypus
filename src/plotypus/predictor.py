import numpy
from utils import autocorrelation

class Baart:
    def __init__(self, estimator, param_grid,
                 fit_params={}, lag=1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.fit_params = fit_params
        self._lag = lag

    def predict(self, X, y=None, **params):
        raise Exception("Not yet implemented")

    @staticmethod
    def cutoff(X):
        return (2 * (X.shape[0] - 1))**(-1/2)
