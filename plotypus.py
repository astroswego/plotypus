import logging
import math
import multiprocessing

import numpy
import re

import os
import interpolation
import linearmodel
from pcat_interface import pcat
from star import (lightcurve, lightcurve_matrix, plot_lightcurves,
                  pca_reconstruction)
from scale import normalize, unnormalize, unnormalize_single, standardize
from utils import (get_files, make_sure_path_exists, map_reduce, save_cache,
                   load_cache)

def main():
#    logger = multiprocessing.log_to_stderr()
#    logger.setLevel(multiprocessing.SUBDEBUG)
    options = get_options()
    clean_options = {}
    files = get_files(options.input, options.format)#[:10]
    stars = options.cache.get('stars') or map_reduce(lightcurve, files, options)
#   For un-normalizing
    star_mins = numpy.reshape(
        numpy.fromiter((star.y_min for star in stars), numpy.float),
        (-1,1))
    star_maxs = numpy.reshape(
        numpy.fromiter((star.y_max for star in stars), numpy.float),
        (-1,1))
#    stars = [lightcurve(f, options=options) for f in files]
#    stars = map_reduce(lightcurve, files, options)
    if options.verbose:
        print("\nAnalyzing {0} of {1} stars".format(len(stars), len(files)))
    if options.output:
        make_sure_path_exists(options.output)
    if (options.PCA_degree):
        pca_input_matrix = lightcurve_matrix(stars, options.evaluator)
#        pca_input_matrix, mmin, mmax = normalize(pca_input_matrix)
        eigenvectors, principle_scores, reconstruction = pcat(pca_input_matrix)
#        (EVs, PCs, pca_lcs) = principle_component_analysis(pca_input_matrix,
#                                                           options.PCA_degree)
#        print(pca_results)
        vanilla_reconstruction = unnormalize(reconstruction,
                                             star_mins, star_maxs)
    if (options.plot_lightcurves_observed or
            options.plot_lightcurves_interpolated or
            options.plot_lightcurves_pca):
#        for s in stars: plot_lightcurves(s, options.evaluator,
#                                         options.output, options=options)
        if options.plot_lightcurves_pca:
            for PCA, s in zip(vanilla_reconstruction, stars):
                plot_lightcurves(s, options.evaluator, options.output,
                                 PCA, options=options)
        else:
            map_reduce(plot_lightcurves, stars, options)
    if options.linear_model:
        A0 = numpy.fromiter(
                 (unnormalize_single(s.coefficients[0],s.y_min,s.y_max)
                  for s in stars),
                 numpy.float)
#        assert False, "A0: {}".format(A0)
        logP = numpy.fromiter((math.log(s.period, 10)
                               for s in stars), numpy.float)
        PC1, PC2 = numpy.hsplit(principle_scores[:,:2], 2)
        PLPC1model, PLPC1fit = linearmodel.linear_model(A0, logP, PC1)
        PLPC2model, PLPC2fit = linearmodel.linear_model(A0, logP, PC2)
        PLPC1model.title = "PLPC1"
        PLPC2model.title = "PLPC2"
        linearmodel.plot_linear_model(PLPC1model, A0, logP, PC1, options.output)
        linearmodel.plot_linear_model(PLPC2model, A0, logP, PC2, options.output)
# Do the plot by doing coeff[0]*logP+coeff[1]*PC_i
#        print("PC1:\n{}\n\nPC2:\n{}".format(PLPC1.summary(),PLPC2.summary()))
            
## Linear Model not yet implemented ##
#    if options.linear_model:
#        coeff_matrix = numpy.fromiter((s.coefficients for s in stars),
#                                      None)
#        coeff_matrix = numpy.array([s.coefficients for s in stars])
#        for h in linear_model_handlers:
#            model = linearmodel.linear_model(
#                h.evaluate_coefficient(coeff_matrix, "A", 0),
#               *h.evaluate_coefficients(coeff_matrix))
#            print(model.summary())
    if options.save_cache:
        save_cache(stars, options)

    #x = numpy.arange(0, 1, 0.01)
    """
    evaluator = options.evaluator
    lcs = numpy.array([evaluator(star.coefficients) for star in stars])
    zs, means, stds = zip(*[standardize(col) for col in lcs.T])
    zs = numpy.array(zs).T
    map_reduce(plot, zs, options)
    eigvals, eigvecs = numpy.linalg.eig(numpy.cov(zs))
    print means, stds
    """

def get_options():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-i', '--input',
      dest='input',                  type='string',
      help='location of stellar observations')
    parser.add_option('-o', '--output',
      dest='output',                 type='string',
      help='output directory')
    parser.add_option('-f', '--format',
      dest='format',                 type='string', default='.dat',
      help='file format')
    parser.add_option('-c', '--save-cache',
      dest='save_cache',             type='string', default=None,
      help='specify the file to save the cache file to')
    parser.add_option('-l', '--load-cache',
      dest='load_cache',             type='string', default=None,
      help='load the specified cache file')
    parser.add_option('--plot-lightcurves-observed',
      dest='plot_lightcurves_observed',             default=False,
      action='store_true',
      help='include observed data in the lightcurve plots')
    parser.add_option('--plot-lightcurves-interpolated',
      dest='plot_lightcurves_interpolated',         default=False,
      action='store_true',
      help='include interpolated data in the lightcurve plots')
    parser.add_option('--plot-lightcurves-pca',
      dest='plot_lightcurves_pca',                  default=False,
      action='store_true',
      help='include PCA data in lightcurve plots')
## Linear Model not yet implemented ##
    parser.add_option('--linear-model',
      dest='linear_model',           type='string', default=None,
      help='perform multiple regression on the interpolation results')
    parser.add_option('--interpolant',
      dest='interpolant',            type='string', default=None,
      help='type of interpolation (poly, spline, trig, lin)')
    parser.add_option('--interpolation-degree',
      dest='interpolation_degree',   type='int',    default=10,
      help='degree of interpolation')
    parser.add_option('--PCA-degree',
      dest='PCA_degree',             type='int',    default=10,
      help='degree of PCA')
    parser.add_option('--min-obs',
      dest='min_obs',                type='int',    default=100,
      help='minimum number of observations per star')
    parser.add_option('--period-bins',
      dest='period_bins',            type='int',    default=50000,
      help='the size of the period space')
    parser.add_option('--min-period',
      dest='min_period',             type='float',  default=0.2,
      help='minimum period of each star')
    parser.add_option('--max-period',
      dest='max_period',             type='float',  default=32.,
      help='maximum period of each star')
    parser.add_option('-s', '--sigma',
      dest='sigma',                  type='float',  default=1,
      help='reject points sigma stds away from the mean')
    parser.add_option('--processors',
      dest='processors',             type='int',    default=None,
      help='number of processors to use')
    parser.add_option('-v', '--verbose',
      dest='verbose', action='store_true',          default=True,
      help='verbose printing')
    parser.add_option('-q', '--quiet',
      dest='verbose', action='store_false',
      help='prevent printing')
    (options, args) = parser.parse_args()

    if options.input is None:
        parser.error('Need input file')
    if options.output is None:
        parser.error('Need output location')
    
    options.cache = load_cache(options) if options.load_cache else {}
#    if options.load_cache:
#        options.interpolant = (options.interpolant or
#                               options.cache['interpolant'])
#        if options.interpolant != options.cache['interpolant']:
#            parser.error(
#                'Specified interpolant does not match cached interpolant')
    # If no interpolant specified, defaults to trigonometric
    options.interpolant = options.interpolant or 'trigonometric'
    if options.interpolant in 'least_squares_polynomial':
        options.interpolant = interpolation.least_squares_polynomial
        options.evaluator = interpolation.polynomial_evaluator
#        if options.linear_model:
#            parser.error('Linear model not yet implemented for polynomial')
    elif options.interpolant in 'spline':
        parser.error('Spline interpolation not yet implemented.')
        options.interpolant = interpolation.spline
        options.evaluator = interpolation.spline_evaluator
    elif options.interpolant in 'trigonometric':
        options.interpolant = interpolation.trigonometric
        options.evaluator = interpolation.trigonometric_evaluator
#        if options.linear_model:
#            linear_model_handlers = linearmodel.parseLinearModelString(
#                options.linear_model,
#                linearmodel.TrigonometricHandler,
#                options.interpolation_degree,
#                options.PCA_degree)
    elif options.interpolant in 'piecewise_linear':
        parser.error('Piecewise linear interpolation not yet implemented.')
        options.interpolant = interpolation.piece_wise_linear
        options.evaluator = interpolation.piece_wise_linear_evaluator
    else:
        parser.error('Could not interpret ' + options.interpolant)
        
    return options#, linear_model_handlers

if __name__ == "__main__":
    main()
