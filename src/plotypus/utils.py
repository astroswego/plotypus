from os import makedirs
from os.path import isdir
from numpy import resize

def make_sure_path_exists(path):
    """Creates the supplied path. Raises OS error if the path cannot be
    created."""
    try:
      makedirs(path)
    except OSError:
      if not isdir(path):
        raise

def get_signal(data):
    """Returns all of the values that are not outliers."""
    return data[~data.mask].data.reshape(-1, data.shape[1])

def get_noise(data):
    """Returns all identified outliers"""
    return data[data.mask].data.reshape(-1, data.shape[1])

def colvec(X):
    return resize(X, (X.shape[0], 1))
