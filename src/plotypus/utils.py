from os import makedirs
from os.path import join, isdir
from sys import stderr
from multiprocessing import Pool
from numpy import absolute, concatenate, median, reshape

__all__ = [
    'verbose_print',
    'pmap',
    'make_sure_path_exists',
    'get_signal',
    'get_noise',
    'colvec',
    'mad',
    'autocorrelation'
]


def verbose_print(message, *, operation, verbosity):
    """Prints message to stdout only if the given operation is in the list of
    verbose operations. If "all" is in the list, all operations are printed.
    """
    if (operation in verbosity) or ("all" in verbosity):
        print(message, file=stderr)


def pmap(func, args, processes=None, callback=lambda *_, **__: None, **kwargs):
    """Parallel equivalent of map(func, args), with the additional ability of
    providing keyword arguments to func, and a callback function which is
    applied to each element in the returned list. Unlike map, the output is a
    non-lazy list. If processes=1, no thread pool is used.
    """
    if processes is 1:
        results = []
        for arg in args:
            result = func(arg, **kwargs)
            results.append(result)
            callback(result)
        return results
    else:
        with Pool() if processes is None else Pool(processes) as p:
            results = [p.apply_async(func, (arg,), kwargs, callback)
                       for arg in args]

            return [result.get() for result in results]


def make_sure_path_exists(path):
    """Creates the supplied path. Raises OS error if the path cannot be
    created.
    """
    try:
        makedirs(path)
    except OSError:
        if not isdir(path):
            raise


def get_periods(periods_file=join('data', 'OGLE-periods.dat')):
    """Parses a periods file whose lines contain Name Period.
    """
    return {name: float(period) for (name, period)
            in (line.strip().split()
                for line in open(periods_file, 'r') if ' ' in line)}


def get_signal(data):
    """Returns all of the values that are not outliers.
    """
    return data[~data.mask].data.reshape(-1, data.shape[1])


def get_noise(data):
    """Returns all identified outliers.
    """
    return data[data.mask].data.reshape(-1, data.shape[1])


def colvec(X):
    """Converts a row-vector into a column-vector.
    """
    return reshape(X, (-1, 1))


def rowvec(X):
    """Converts a column-vector into a row-vector.
    """
    return reshape(X, (1, -1))[0]


def mad(data, axis=None):
    """Computes the median absolute deviation of an array.
    """
    return median(absolute(data - median(data, axis)), axis)


def autocorrelation(data, lag=1):
    """Computes the autocorrelation of the data with the given lag.
    Autocorrelation is simply
    autocovariance(data) / covariance(data-mean, data-mean),
    where autocovariance is simply
    covariance((data-mean)[:-lag], (data-mean)[lag:]).
    """
    differences = data - data.mean()
    products = differences * concatenate((differences[lag:],
                                          differences[:lag]))

    return products.sum() / (differences**2).sum()
