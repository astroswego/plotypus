from numpy.random import random
from numpy.testing import assert_array_almost_equal

def standardize(x):
    """Scales a matrix so that its mean is 0 and its standard deviation is 1
    >>> 5==4
    """
    x_mean, x_std = x.mean(), x.std()
    z = (x-x_mean)/x_std
    return z, x_mean, x_std

def unstandardize(z, x_mean, x_std):
    """Reverses standardization.
    
    >>> x = random([10])
    >>> assert_array_almost_equal(x, unstandardize(*standardize(x)))
    """
    return z*x_std + x_mean

def normalize(x):
    """Scales a matrix so that it ranges from 0 to 1"""
    x_max, x_min = x.max(), x.min()
    return (x-x_min)/(x_max-x_min), x_min, x_max

def unnormalize(y, x_min, x_max):
    """Reverses normalization.
    
    >>> x = random([10])
    >>> assert_array_almost_equal(x, unstandardize(*standardize(x)))
    """
    return y*(x_max-x_min)+x_min

def do(x):
    z, x_mean, x_std = standardize(x)
    y, z_min, z_max = normalize(z)
    return y, z_min, z_max, x_mean, x_std

def undo(y, z_min, z_max, x_mean, x_std):
    return unstandardize(unnormalize(y, z_min, z_max), x_mean, x_std)

def test():
    x = random([10000])
    assert_array_almost_equal(x, unstandardize(*standardize(x))) # Check that standardization is reversible
    assert_array_almost_equal(unnormalize(*normalize(x)), x) # Check that normalization is reversible
    assert_array_almost_equal(undo(*do(x)), x) # Check that standardization and normalization is reversible

if __name__ == '__main__':
    import doctest
    doctest.testmod()
