from os import makedirs
from os.path import join, isdir
from sys import stderr
from multiprocessing import Pool
from numpy import absolute, concatenate, median, resize

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
    """
    Prints *message* to stderr only if the given *operation* is in the list
    *verbosity*. If "all" is in *verbosity*, all operations are printed.

    **Parameters**

    message : str
        The message to print.
    operation : str
        The type of operation being performed.
    verbosity : [str] or None
        The list of operations to print *message* for. If "all" is contained
        in the list, then all operations are printed. If None, no operation is
        printed.

    **Returns**

    None
    """
    if (verbosity is not None) and ((operation in verbosity) or
                                    ("all"     in verbosity)):
        print(message, file=stderr)


def pmap(func, args, processes=None, callback=lambda *_, **__: None, **kwargs):
    """pmap(func, args, processes=None, callback=do_nothing, **kwargs)

    Parallel equivalent of ``map(func, args)``, with the additional ability of
    providing keyword arguments to func, and a callback function which is
    applied to each element in the returned list. Unlike map, the output is a
    non-lazy list. If *processes* is 1, no thread pool is used.

    **Parameters**

    func : function
        The function to map.
    args : iterable
        The arguments to map *func* over.
    processes : int or None, optional
        The number of processes in the thread pool. If only 1, no thread pool
        is used to avoid useless overhead. If None, the number is chosen based
        on your system by :class:`multiprocessing.Pool` (default None).
    callback : function, optional
        Function to call on the return value of ``func(arg)`` for each *arg*
        in *args* (default do_nothing).
    kwargs : dict
        Extra keyword arguments are unpacked in each call of *func*.

    **Returns**

    results : list
        A list equivalent to ``[func(x, **kwargs) for x in args]``.
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
    """
    Creates the supplied *path* if it does not exist.
    Raises *OSError* if the *path* cannot be created.

    **Parameters**

    path : str
        Path to create.

    **Returns**

    None
    """
    try:
        makedirs(path)
    except OSError:
        if not isdir(path):
            raise


def get_signal(data):
    """
    Returns all of the values in *data* that are not outliers.

    **Parameters**

    data : masked array

    **Returns**

    signal : array
        Non-masked values in *data*.
    """
    return data[~data.mask].data.reshape(-1, data.shape[1])


def get_noise(data):
    """
    Returns all identified outliers in *data*.

    **Parameters**

    data : masked array

    **Returns**

    noise : array
        Masked values in *data*.
    """
    return data[data.mask].data.reshape(-1, data.shape[1])


def colvec(X):
    """
    Converts a row-vector *X* into a column-vector.

    **Parameters**

    X : array-like, shape = [n_samples]

    **Returns**

    out : array-like, shape = [n_samples, 1]
    """
    return resize(X, (X.shape[0], 1))


def rowvec(X):
    """
    Converts a column-vector *X* into a row-vector.

    **Parameters**

    X : array-like, shape = [n_samples, 1]

    **Returns*

    out : array-like, shape = [n_samples]
    """
    return resize(X, (1, X.shape[0]))[0]


def mad(data, axis=None):
    """
    Computes the median absolute deviation of *data* along a given *axis*.
    See `link <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_ for
    details.

    **Parameters**

    data : array-like

    **Returns**

    mad : number or array-like
    """
    return median(absolute(data - median(data, axis)), axis)


def autocorrelation(X, lag=1):
    """
    Computes the autocorrelation of *X* with the given *lag*.
    Autocorrelation is simply
    autocovariance(X) / covariance(X-mean, X-mean),
    where autocovariance is simply
    covariance((X-mean)[:-lag], (X-mean)[lag:]).

    See `link <https://en.wikipedia.org/wiki/Autocorrelation>`_ for details.

    **Parameters**

    X : array-like, shape = [n_samples]

    lag : int, optional
        Index difference between points being compared (default 1).
    """
    differences = X - X.mean()
    products = differences * concatenate((differences[lag:],
                                          differences[:lag]))

    return products.sum() / (differences**2).sum()
