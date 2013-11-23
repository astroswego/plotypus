import numpy
#import mdp # Use this for implementing PCA in python

x = numpy.arange(0, 1, 0.01)

def least_squares_polynomial(data, degree):
    """Returns the coefficients to and an evaluator for a least-square
    polynomial interpolation for the given data."""
    sorted = data[data[:, 0].argsort()].T
    return numpy.polyfit(sorted[0], sorted[1], degree)

def polynomial_evaluator(coefficients, x=x):
    """Returns the value at the given point when evaluated by the polynomial
    function defined by the given coefficients."""
    y = 0
    for c in coefficients: y = y*x+c # Horner's method
    return y

def trigonometric(data, degree):
    sorted = data[data[:,0].argsort()].T
    phases, mags = sorted[:2]
    coefficient_matrix = trigonometric_coefficient_matrix(phases, degree)
    return numpy.linalg.lstsq(coefficient_matrix, mags)[0]
    
def trigonometric_evaluator(coefficients, x=x):
    degree = int((len(coefficients)-1) / 2)
    coefficient_matrix = trigonometric_coefficient_matrix(x, degree)
    return numpy.dot(coefficients, coefficient_matrix.T)

def trigonometric_coefficient_matrix(phases, degree):
    (n_rows, n_cols) = (len(phases), 2*degree + 1)
    return numpy.array([
        [1 if j == 0
         else numpy.cos((j+1)*numpy.pi*phases[i]) if j % 2
         else numpy.sin(j*numpy.pi*phases[i])
         for j in range(n_cols)]
        for i in range(n_rows)])

def ak_bk2Ak_Phik(ak_bk_coefficients):
    A0 = ak_bk_coefficients[0]
    ak_coeffs = ak_bk_coefficients[1::2]
    bk_coeffs = ak_bk_coefficients[2::2]

    ak_bk_ratio = bk_coeffs / ak_coeffs
    Phik_coeffs = numpy.arctan(ak_bk_ratio)
    Ak_coeffs = ak_coeffs / numpy.sin(Phik_coeffs)
    
    Ak_Phik_coeffs = [None]*len(ak_bk_coefficients)
    Ak_Phik_coeffs[0] = A0
    Ak_Phik_coeffs[1::2] = Ak_coeffs
    Ak_Phik_coeffs[2::2] = Phik_coeffs

#    assert False, str(Ak_Phik_coeffs.shape)
    return numpy.array(Ak_Phik_coeffs).reshape(1,-1)

def lightcurve_matrix(stars, evaluator, x=x):
    m = numpy.vstack(tuple(numpy.array(evaluator(s.coefficients, x))
                           for s in stars))
#    iterable = (evaluator(s.coefficients, x) for s in stars)
#    m = numpy.vstack(numpy.fromiter((evaluator(s.coefficients, x),numpy.float) for s in stars))

 #    m = numpy.vstack(tuple(numpy.fromiter(iter(evaluator(s.coefficients, x)),
 #                                          numpy.float)
 #                           for s in stars))
    return m

 # def principle_component_analysis(data, degree):
 #     standardized_data, data_mean, data_std = standardize(data)
 #     pcanode = mdp.nodes.PCANode(output_dim=degree)
 #     pcanode.train(standardized_data.T)
 #     pcanode.stop_training()
 #     eigenvectors = pcanode.execute(standardized_data.T)
 #     principle_scores = numpy.dot(standardized_data, eigenvectors)
 #     standardized_reconstruction_matrix = pca_reconstruction(eigenvectors,
 #                                                             principle_scores)
 #     reconstruction_matrix = unstandardize(standardized_reconstruction_matrix,
 #                                           data_mean, data_std)
 #     return eigenvectors, principle_scores, reconstruction_matrix

def pca_reconstruction(eigenvectors, principle_scores):
    """Returns an array in which each row contains the magnitudes of one star's
    lightcurve. eigenvectors is a (number of phases)x(order of PCA) array,
    principle_components is a (number or stars)x(order of PCA) array, and the
    return array has shape (number of stars)x(number of phases)."""
    return numpy.dot(eigenvectors, principle_scores.T).T  