from numpy import array, fromiter, float as nfloat, reshape
from numpy.random import random
from numpy.testing import assert_array_almost_equal

def standardize(x):
    tolerance = 1e-6
    y = x.T
    y_mean = reshape(fromiter((col.mean() for col in y), nfloat), (1,-1))
    y_std = reshape(fromiter((col.std() if col.std() > tolerance else 1.0
                              for col in y), nfloat),
                             (1,-1))
    return ((y-y_mean.T)/y_std.T).T, y_mean, y_std

def unstandardize(x, x_mean, x_std):
    return x*x_std + x_mean

def normalize(x):
    """Scales a matrix so that each row ranges from 0 to 1"""
    x_min = reshape(fromiter((row.min() for row in x), nfloat), (1,-1))
    x_max = reshape(fromiter((row.max() for row in x), nfloat), (1,-1))
    return (x-x_min.T)/(x_max.T-x_min.T), x_min, x_max

def unnormalize(x, x_min, x_max):
    """Reverses normalization of a matrix's rows"""
    print("x: {}, x_max: {}, x_min: {}"
          .format(x.shape, x_max.shape, x_min.shape))
    return x*(x_max-x_min) + x_min

def normalize_single(x):
    """Scales a vector so that it ranges from 0 to 1"""
    x_max, x_min = x.max(), x.min()
    return (x-x_min)/(x_max-x_min), x_min, x_max

def unnormalize_single(y, x_min, x_max):
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
