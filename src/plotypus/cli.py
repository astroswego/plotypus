import numpy as np
from numpy import std
from sys import exit, stdin, stdout, stderr
from os import path, listdir
from argparse import ArgumentError, ArgumentParser, SUPPRESS
from pandas import read_table
from sklearn.linear_model import (LassoCV, LassoLarsCV, LassoLarsIC,
                                  LinearRegression, RidgeCV, ElasticNetCV)
from sklearn.grid_search import GridSearchCV
from matplotlib import rc_params_from_file
from collections import ChainMap
from functools import partial
from itertools import chain, repeat
import plotypus.lightcurve
from plotypus.lightcurve import (make_predictor, get_lightcurve_from_file,
                                 savetxt_lightcurve, plot_lightcurve,
                                 plot_residual)
from plotypus.periodogram import Lomb_Scargle, conditional_entropy
import plotypus
from plotypus.preprocessing import Fourier
from plotypus.utils import (colvec, mad, make_sure_path_exists, pmap,
                            valid_basename, verbose_print)
from plotypus.resources import matplotlibrc

import pkg_resources # part of setuptools
__version__ = pkg_resources.require("plotypus")[0].version

# Dict mapping higher level output types (table, plot) to the specific
# quantaties output (lightcurve, residual, etc). Any new output types need to
# be added here, with "-"'s replaced with "_"'s.
outputs = {
    "table": ["lightcurve", "residual", "periodogram",
              "fourier_coeffs", "fourier_ratios"],
    "plot":  ["lightcurve", "residual", "periodogram"]
}

def get_args():
    parser = ArgumentParser()
    general_group    = parser.add_argument_group('General')
    lightcurve_group = parser.add_argument_group('Light Curve')
    param_group      = parser.add_argument_group('Star Parameters')
    parallel_group   = parser.add_argument_group('Parallel')
    period_group     = parser.add_argument_group('Periodogram')
    fourier_group    = parser.add_argument_group('Fourier')
    outlier_group    = parser.add_argument_group('Outlier Detection')
#    regression_group = parser.add_argument_group('Regression')

    parser.add_argument('--version', action='version',
        version='%(prog)s {version}'.format(version=__version__))

    ## General Group #########################################################

    general_group.add_argument('-i', '--input', type=str,
        default=None,
        help='location of stellar observations '
             '(default = stdin)')
    general_group.add_argument('-o', '--output-all', type=str,
        default=None, metavar="DIR",
        help='(optional) location to output all file types (plots and tables '
             'alike), creating a sub-directory for each star. individual '
             'filenames have defaults which can be overridden by the options '
             'of the same name. '
             'cannot be used in conjunction with --output-{table,plot}-all '
             'option')
    general_group.add_argument('--output-table-all', type=str,
        default=None, metavar="DIR",
        help='(optional) location to output all table file types, creating a '
             'sub-directory for each star. individual filenames have defaults '
             'which can be overridden by the options of the same name. '
             'cannot be used in conjunction with --output-all option')
    general_group.add_argument('--output-plot-all', type=str,
        default=None, metavar="DIR",
        help='(optional) location to output all plot file types, creating a '
             'sub-directory for each star. individual filenames have defaults '
             'which can be overridden by the options of the same name. '
             'cannot be used in conjunction with --output-all option')
    general_group.add_argument('-n', '--star-name', type=str,
        default=None,
        help='name of star '
             '(default = name of input file)')
    general_group.add_argument('-f', '--format', type=str,
        default='%.5f',
        help='format specifier for output table')
    general_group.add_argument('--output-sep', type=str,
        default='\t',
        help='column separator string in output table '
             '(default = TAB)')
    general_group.add_argument('--no-header', action='store_true',
        help='suppress header row in table output')
    general_group.add_argument('--sanitize-latex', action='store_true',
        help='enable to sanitize star names for LaTeX formatting')
    general_group.add_argument('--legend', action='store_true',
        help='whether legends should be put on the output plots '
             '(default = False)')
    general_group.add_argument('--extension', type=str,
        default='.dat', metavar='EXT',
        help='extension which follows a star\'s name in data filenames '
             '(default = ".dat")')
    general_group.add_argument('--skiprows', type=int,
        default=0,
        help='number of rows at the head of each file to skip')
    general_group.add_argument('--use-cols', type=int, nargs='+',
        default=SUPPRESS,
        help='columns to use from data file '
             '(default = 0 1 2)')
    general_group.add_argument('-s', '--scoring', type=str,
        choices=['MSE', 'R2'], default=SUPPRESS,
        help='scoring metric to use '
             '(default = "R2")')
    general_group.add_argument('--scoring-cv', type=int,
        default=SUPPRESS, metavar='N',
        help='number of folds in the scoring regularization_cv validation '
             '(default = 3)')
    general_group.add_argument('--shift', type=float,
        default=None,
        help='phase shift to apply to each light curve, or shift to max '
             'light if None given '
             '(default = None)')
    general_group.add_argument('--min-phase-cover', type=float,
        default=SUPPRESS, metavar='COVER',
        help='minimum fraction of phases that must have points '
             '(default = 0)')
    general_group.add_argument('--min-observations', type=int,
        default=1, metavar='N',
        help='minimum number of observation needed to avoid skipping a star '
             '(default = 1)')
    general_group.add_argument('--plot-engine', type=str,
        default='mpl', choices=['mpl', 'tikz'],
        metavar='ENGINE',
        help='engine to use for plotting. mpl will output image formats, while '
             'tikz will output tikz source code for use in LaTeX '
             '(default = "mpl")')
    general_group.add_argument('--matplotlibrc', type=str,
        default=matplotlibrc,
        metavar='RC',
        help='matplotlibrc file to use for formatting plots '
             '(default file is in plotypus.resources.matplotlibrc)')
    general_group.add_argument('-v', '--verbosity', type=str, action='append',
        default=None, choices=['all', 'coverage', 'outlier', 'period'],
        metavar='OPERATION',
        help='specifies an operation to print verbose output for, or '
             '"all" to print all verbose output '
             '(default = None)')

    ## Light Curve Group #####################################################

    lightcurve_group.add_argument('--output-table-lightcurve', type=str,
        default=None,
        help='(optional) location to save fitted light curve tables, '
             'or prefix to filename if any of `--output-*-all` options used.')
    lightcurve_group.add_argument('--output-plot-lightcurve', type=str,
        default=None,
        help='(optional) location to save light curve plots, '
             'or prefix to filename if any of `--output-*-all` options used.')
    lightcurve_group.add_argument('--output-table-residual', type=str,
        default=None,
        help='(optional) location to save the fit residuals tables, '
             'or prefix to filename if any of `--output-*-all` options used.')
    lightcurve_group.add_argument('--output-plot-residual', type=str,
        default=None,
        help='(optional) location to save the fit residuals plots, '
             'or prefix to filename if any of `--output-*-all` options used.')
    lightcurve_group.add_argument('--phase-points', type=int,
        default=100, metavar='N',
        help='number of phase points to output '
             '(default = 100)')

    ## Parameter Group #######################################################

    param_group.add_argument('--parameters', type=str,
        default=None, metavar='FILE',
        help='file containing table of parameters such as period and shift '
             '(default = None)')
    param_group.add_argument('--param-sep', type=str,
        default="\\s+",
        help='string or regex to use as column separator when reading '
             'parameters file '
             '(default = any whitespace)')
    param_group.add_argument('--period-label', type=str,
        default='Period', metavar='LABEL',
        help='title of period column in parameters file '
             '(default = Period)')
    param_group.add_argument('--shift-label', type=str,
        default='Shift', metavar='LABEL',
        help='title of shift column in parameters file '
             '(default = Shift)')

    ## Parallel Group ########################################################

    parallel_group.add_argument('--star-processes', type=int,
        default=1, metavar='N',
        help='number of stars to process in parallel '
             '(default = 1)')
    parallel_group.add_argument('--selector-processes', type=int,
        default=SUPPRESS, metavar='N',
        help='number of processes to use for each selector '
             '(default depends on selector used)')
    parallel_group.add_argument('--scoring-processes', type=int,
        default=SUPPRESS, metavar='N',
        help='number of processes to use for scoring, if not done by selector '
             '(default = 1)')
    parallel_group.add_argument('--period-processes', type=int,
        default=1, metavar='N',
        help='number of periods to process in parallel '
             '(default = 1)')

    ## Period Group ##########################################################

    period_group.add_argument('--output-table-periodogram', type=str,
        default=None,
        help='(optional) location to save the periodogram table, '
             'or prefix to filename if any of `--output-*-all` options used.')
    period_group.add_argument('--output-plot-periodogram', type=str,
        default=None,
        help='(optional) location to save the periodogram plots, '
             'or prefix to filename if any of `--output-*-all` options used.')
    period_group.add_argument('--output-periodogram-form', type=str,
        default='period', choices=['frequency', 'period'],
        help='form to output periodogram in, for both tables and plots '
             '(default = "period")')
    period_group.add_argument('--period', type=float,
        default=None,
        help='period to use for all stars '
             '(default = None)')
    period_group.add_argument('--min-period', type=float,
        default=SUPPRESS, metavar='P',
        help='minimum period of each star '
             '(default = 0.2)')
    period_group.add_argument('--max-period', type=float,
        default=SUPPRESS, metavar='P',
        help='maximum period of each star '
             '(default = 32.0)')
    period_group.add_argument('--coarse-precision', type=float,
        default=SUPPRESS,
        help='level of granularity on first pass '
             '(default = 0.00001)')
    period_group.add_argument('--fine-precision', type=float,
        default=SUPPRESS,
        help='level of granularity on second pass '
             '(default = 0.000000001)')
    period_group.add_argument('--periodogram', type=str,
        choices=["Lomb_Scargle", "conditional_entropy"],
        default="Lomb_Scargle",
        help='method for determining period '
             '(default = Lomb_Scargle)')

    ## Fourier Group #########################################################

    fourier_group.add_argument('--output-table-fourier-coeffs', type=str,
        default=None,
        help='(optional) location to save the fourier coefficient table, '
             'or prefix to filename if any of `--output-*-all` options used.')
    fourier_group.add_argument('--output-table-fourier-ratios', type=str,
        default=None,
        help='(optional) location to save the fourier ratio table, '
             'or prefix to filename if any of `--output-*-all` options used.')
    fourier_group.add_argument('-d', '--fourier-degree', type=int, nargs=2,
        default=(2, 20), metavar=('MIN', 'MAX'),
        help='range of degrees of fourier fits to use '
             '(default = 2 20)')
    fourier_group.add_argument('-r', '--regressor',
        choices=['LassoCV', 'LassoLarsCV', 'LassoLarsIC', 'OLS', 'RidgeCV',
                 'ElasticNetCV'],
        default='LassoLarsIC',
        help='type of regressor to use '
             '(default = "Lasso")')
    fourier_group.add_argument('--selector',
        choices=['Baart', 'GridSearch'],
        default='GridSearch',
        help='type of model selector to use '
             '(default = "GridSearch")')
    fourier_group.add_argument('--fourier-form', type=str,
        default='cos', choices=['sin_cos', 'sin', 'cos'],
        help='form of Fourier series to use in coefficient output, '
             'does not affect the fit '
             '(default = "cos")')
    fourier_group.add_argument('--max-iter', type=int,
        default=1000, metavar='N',
        help='maximum number of iterations in the regularization path '
             '(default = 1000)')
    fourier_group.add_argument('--regularization-cv', type=int,
        default=None, metavar='N',
        help='number of folds used in regularization regularization_cv '
             'validation '
             '(default = 3)')

    ## Outlier Group #########################################################

    outlier_group.add_argument('--sigma', type=float,
        default=SUPPRESS,
        help='rejection criterion for outliers '
             '(default = 20)')
    outlier_group.add_argument('--sigma-clipping', type=str,
        choices=["std", "mad"], default="mad",
        help='sigma clipping metric to use '
             '(default = "mad")')

    args = parser.parse_args()

    # configure Matplotlib only if necessary
    plot_outputs = [args.output_plot_lightcurve]
    if (args.plot_engine == "mpl" and
            any(output is not None for output in plot_outputs)):
        rcParams = rc_params_from_file(fname=args.matplotlibrc,
                                       fail_on_error=True)
        plotypus.lightcurve.matplotlib.rcParams = rcParams

    regressor_choices = {
        "LassoCV"             : LassoCV(max_iter=args.max_iter,
                                        cv=args.regularization_cv,
                                        fit_intercept=False),
        "LassoLarsCV"         : LassoLarsCV(max_iter=args.max_iter,
                                            cv=args.regularization_cv,
                                            fit_intercept=False),
        "LassoLarsIC"         : LassoLarsIC(max_iter=args.max_iter,
                                            fit_intercept=False),
        "OLS"                 : LinearRegression(fit_intercept=False),
        "RidgeCV"             : RidgeCV(cv=args.regularization_cv,
                                        fit_intercept=False),
        "ElasticNetCV"        : ElasticNetCV(max_iter=args.max_iter,
                                             cv=args.regularization_cv,
                                             fit_intercept=False)
    }
    selector_choices = {
        "Baart"               : None,
        "GridSearch"          : GridSearchCV
    }
    periodogram_choices = {
        "Lomb_Scargle"        : Lomb_Scargle,
        "conditional_entropy" : conditional_entropy
    }
    sigma_clipping_choices = {
        "std"                 : std,
        "mad"                 : mad
    }

    if hasattr(args, 'scoring'):
        scoring_choices = {
            'R2'              : 'r2',
            'MSE'             : 'mean_squared_error'
        }
        args.scoring = scoring_choices[args.scoring]

    args.regressor = regressor_choices[args.regressor]
    Selector = selector_choices[args.selector] or GridSearchCV
    args.periodogram = periodogram_choices[args.periodogram]
    args.sigma_clipping = sigma_clipping_choices[args.sigma_clipping]

    args.predictor = make_predictor(Selector=Selector,
                                    use_baart=(args.selector == 'Baart'),
                                    **vars(args))
    args.phases = np.arange(0, 1, 1/args.phase_points)

    if args.parameters is not None:
        args.parameters = read_table(args.parameters, args.param_sep,
                                     index_col=0, engine='python')

    output_setup(args)

    return args


def main():
    args = get_args()
    args_dict = vars(args)

    min_degree, max_degree = args.fourier_degree
    filenames = list(map(lambda x: x.strip(), get_files(args.input)))
    filepaths = map(lambda filename:
                    filename if path.isfile(filename)
                             else path.join(args.input, filename),
                    filenames)

    # create a dict containing all args which can be pickled, because
    # all parameters to pmap must be picklable
    non_picklable_args = {"input"}
    picklable_args = {k: args_dict[k]
                      for k in args_dict
                      if k not in non_picklable_args}
    sep = args.output_sep

    if not args.no_header:
        # print file header
        print(*['Name',
                'A_0',
                'dA_0',
                'Period',
                'Shift',
                'Coverage',
                'Inliers',
                'Outliers',
                'R^2',
                'MSE',
                'MaxDegree',
                'Params'],
              sep=sep)

    def printer(result):
        if result is not None:
            print_star(result, max_degree, args.fourier_form,
                       args.format, sep)
    pmap(process_star, filepaths, callback=printer,
         processes=args.star_processes, **picklable_args)


def fname_fn_bulk(directory, prefix):
    """fname_fn_bulk(directory, prefix)
    Returns a function which takes a name, and returns a filename in the given
    `directory`, whose basename has the given `prefix` and `suffix`.
    """
    def fn(name, ext):
        full_directory = path.join(directory, name)
        make_sure_path_exists(full_directory)
        return path.join(full_directory, prefix+ext)

    return fn


def fname_fn_individual(directory):
    """fname_fn_individual(directory)
    Returns a function which takes a name, and returns a filename in the given
    `directory`, whose basename has the given `prefix` and `suffix`.
    """
    def fn(name, ext):
        return path.join(directory, name+ext)

    return fn


def output_attr_name_from_parent_child(parent, child):
    """output_attr_name_from_parent_child(parent, child)

    Returns the attribute name of an output type. These attributes have names
    of the form output_<parent>_<child>.

    Examples include "output_plot_lightcurve" and "output_table_residuals".
    """
    return "_".join(["output", parent, child])


def output_setup_bulk(args, directory, parent, child):
    """output_setup_bulk(args, directory, parent, child)

    Sets the output filename function for the given parent-child output type,
    assuming this is a bulk output.
    """
    # string representing the name of the output type attribute
    attr_name = output_attr_name_from_parent_child(parent, child)

    # the attribute should either be the string representation of the filename
    # prefix, or if one wasn't given, default to the name of the output type
    prefix = getattr(args, attr_name, None)
    if prefix is None:
        prefix = child

    # exit in error if prefix contains a directory or is the empty string
    if not valid_basename(prefix) or prefix == "":
        print(prefix + " is not a valid filename prefix",
              file=stderr)
        exit(1)

    # replace the output type attribute with a function taking as input a star's
    # name and file extension, and returning the desired filename string
    setattr(args, attr_name, fname_fn_bulk(directory, prefix))


def output_setup_individual(args, parent, child):
    """output_setup_individual(args, parent, child)

    Sets the output filename function for the given parent-child output type,
    assuming this is an individual output.
    """
    # string representing the name of the output type attribute
    attr_name = output_attr_name_from_parent_child(parent, child)
    # the attribute should either be the string representation of the output
    # directory, or None if one wasn't given
    directory = getattr(args, attr_name, None)

    # replace the output type attribute with a function taking as input a star's
    # name and file extension, and returning the desired filename string,
    # assuming a directory was given
    if directory is not None:
        make_sure_path_exists(directory)
        setattr(args, attr_name, fname_fn_individual(directory))


def output_setup(args):
    """output_setup(args)

    Sets the output filename functions for all of the various output types
    requested.
    """
    # check for illegal combinations
    if (args.output_all is not None and
        any(x is not None for x in [args.output_table_all,
                                    args.output_plot_all])):
        print("Cannot combine --output-all with --output-table-all or",
              "--output-plot-all flags.",
              file=stderr)
        exit(1)

    # output all filetypes
    if args.output_all is not None:
        make_sure_path_exists(args.output_all)
        for parent, children in outputs.items():
            for child in children:
                output_setup_bulk(args, args.output_all,
                                  parent, child)

    # output all table/plot filetypes to separate directories
    if args.output_table_all is not None:
        make_sure_path_exists(args.output_table_all)
        for child in outputs["table"]:
            output_setup_bulk(args, args.output_table_all,
                              "table", child)
    if args.output_plot_all is not None:
        make_sure_path_exists(args.output_plot_all)
        for child in outputs["plot"]:
            output_setup_bulk(args, args.output_plot_all,
                              "plot", child)

    # output all individual filetypes explicitly requested, assuming none of
    # the bulk operations above were selected
    if all(x is None for x in [args.output_all,
                               args.output_table_all,
                               args.output_plot_all]):
        for parent, children in outputs.items():
            for child in children:
                output_setup_individual(args, parent, child)


def process_star(filename,
                 *,
                 extension, star_name, period, shift,
                 parameters, period_label, shift_label,
                 plot_engine,
                 output_table_fourier_coeffs, output_table_fourier_ratios,
                 output_table_lightcurve,     output_plot_lightcurve,
                 output_table_residual,       output_plot_residual,
                 output_table_periodogram,    output_plot_periodogram,
                 **kwargs):
    """process_star(filename, *, extension, star_name, period, shift, parameters, period_label, shift_label, plot_engine, output_table_fourier_coeffs, output_table_fourier_ratios, output_table_lightcurve,     output_plot_lightcurve, output_table_residual,       output_plot_residual, output_table_periodogram,    output_plot_periodogram, **kwargs)

    Processes a star's lightcurve, prints its coefficients, and saves its
    plotted lightcurve to a file. Returns the result of get_lightcurve.
    """
    if star_name is None:
        basename = path.basename(filename)
        if basename.endswith(extension):
            star_name = basename[:-len(extension)]
        else:
            # file has wrong extension
            return
    if parameters is not None:
        if period is None:
            try:
                period = parameters[period_label][star_name]
            except KeyError:
                pass
            if shift is None:
                try:
                    shift = parameters.loc[shift_label][star_name]
                except KeyError:
                    pass

    result = get_lightcurve_from_file(
        filename, name=star_name, period=period, shift=shift,
        output_periodogram=bool(output_table_periodogram or
                                output_plot_periodogram),
        **kwargs)
    # no results for this star, skip
    if result is None:
        return

    # currenlty only .tikz plot output needs an explicit extension, mpl
    # provides its own
    plot_extension = ".tikz" if plot_engine == "tikz" else ""

    if output_table_fourier_coeffs is not None:
        # construct the filename for the output table
        filename = output_table_fourier_coeffs(result["name"], extension)

        coeffs = result["coefficients"]
        coeff_errors = result["coefficient_errors"]

        # generate wavenumbers (only handles single-period case)
        n_coeffs = np.size(coeffs, 0)
        ks = colvec(np.arange(n_coeffs))

        coeff_table = np.hstack((ks, coeffs, coeff_errors))

        # construct the header
        header = kwargs["output_sep"].join(
            ["k", "a_k", "b_k",   "da_k", "db_k"]
                if kwargs["fourier_form"] == "sin_cos" else
            ["k", "A_k", "Phi_k", "dA_k", "dPhi_k"]
        )

        # save the table to a file
        np.savetxt(filename, coeff_table,
                   # improve this by making ks integers
                   fmt=kwargs["format"],
                   delimiter=kwargs["output_sep"],
                   header=header)

    if (output_table_fourier_ratios is not None
        and kwargs["fourier_form"] != "sin_cos"):
        # construct the filename for the output table
        filename = output_table_fourier_ratios(result["name"], extension)

        ratios = result["fourier_ratios"]
        ratio_errors = result["fourier_ratio_errors"]
        # generate ratio indices (e.g. Phi_ji)
        # NOTE: currently only supports Phi_j1
        n_ratios, *_ = np.shape(ratios)
        j = colvec(np.arange(2, n_ratios+2))
        i = np.ones_like(j)

        ratio_table = np.hstack((j, i, ratios, ratio_errors))

        # construct the header
        header = kwargs["output_sep"].join(
            ["j", "i", "R_ji", "Phi_ji", "dR_ji", "dPhi_ji"]
        )
        np.savetxt(filename, ratio_table,
                   fmt=kwargs["format"],
                   delimiter=kwargs["output_sep"],
                   header=header)


    # output the phased lightcurve as a table
    if output_table_lightcurve is not None:
        # construct the filename for the output table
        filename = output_table_lightcurve(result["name"], extension)
        # construct the header
        header = kwargs["output_sep"].join(
            ["Phase", "Magnitude"]
        )
        # save the table to a file
        savetxt_lightcurve(filename, result["lightcurve"],
                           fmt=kwargs["format"],
                           delimiter=kwargs["output_sep"],
                           header=header)

    # output the lightcurve as a plot
    if output_plot_lightcurve is not None:
        # construct the filename for the output plot
        filename = output_plot_lightcurve(result["name"], plot_extension)
        # make the plot
        plot = plot_lightcurve(output=filename,
                               engine=plot_engine,
                               **ChainMap(kwargs, result))
        # allow figure to get garbage collected
        if plot_engine == "mpl":
            # Need to use a local import, a top level import had to be avoided,
            # because the backend must be configured *before* importing pyplot.
            # Fret not, the import is a no-op after the first call
            from matplotlib.pyplot import close
            close(plot)

    # output the residuals as a table
    if output_table_residual is not None:
        # construct the filename for the output table
        filename = output_table_residual(result["name"], extension)
        # construct the header
        header = kwargs["output_sep"].join(
            ["Phase", "Magnitude"]
        )
        # save the table to a file
        np.savetxt(filename, result["residuals"],
                   fmt=kwargs["format"],
                   delimiter=kwargs["output_sep"],
                   header=header)

    # output the residuals as a plot
    if output_plot_residual is not None:
        # construct the filename for the output plot
        filename = output_plot_residual(result["name"], plot_extension)
        # make the plot
        plot = plot_residual(result["name"], result["residuals"],
                             output=filename,
                             sanitize_latex=kwargs["sanitize_latex"])
        # allow figure to get garbage collected
        if plot_engine == "mpl":
            # Need to use a local import, a top level import had to be avoided,
            # because the backend must be configured *before* importing pyplot.
            # Fret not, the import is a no-op after the first call
            from matplotlib.pyplot import close
            close(plot)

    # output the periodogram as a table
    if all(x is not None for x in [output_table_periodogram,
                                   result["periodogram"]]):
        if kwargs["output_periodogram_form"] == "frequency":
            # frequency
            pgram = 2*np.pi / result["periodogram"]
        else:
            # period
            pgram = result["periodogram"]
            # flip array vertically because periods are in reverse order
            pgram = np.flipud(pgram)

        # construct the filename for the output table
        filename = output_table_periodogram(result["name"], extension)
        # construct the header
        header = kwargs["output_sep"].join(
            ["Period", "Pgram"]
        )
        # save the table to a file
        np.savetxt(filename, pgram,
                   fmt=kwargs["format"],
                   delimiter=kwargs["output_sep"],
                   header=header)

    # TODO: output the periodogram as a plot

    return result


def print_star(result, max_degree, form, fmt, sep):
    """print_star(result, max_degree, form, fmt, sep)

    Prints a star's entry in the table printed to stdout.
    """
    if result is None:
        return

    def format_one(x):
        """Format one number string according to fmt."""
        return fmt % x

    def format_all(x):
        """Format every number in a sequence according to fmt."""
        return map(format_one, x)

    def format_keys(*keys):
        """Format result[key] for the list of keys provided."""
        return format_all(result[key] for key in keys)

    # count inliers and outliers
    points   = result['phased_data'][:,0].size
    outliers = np.ma.count_masked(result['phased_data'][:, 0])
    inliers  = points - outliers

    # get fourier coefficients and compute ratios
    coefs          = result["coefficients"]
    coef_errors    = result["coefficient_errors"]

    max_degree = max(np.trim_zeros(coefs[1:,0], 'b').size,
                     np.trim_zeros(coefs[1:,1], 'b').size)
    n_params   = np.count_nonzero(coefs[1::2])

    # print the entry for the star with tabs as separators
    # and itertools.chain to separate the different results into a
    # continuous list which is then unpacked
    print(*chain(*[[result['name']],
                   # A_0
                   format_all([coefs[0,0], coef_errors[0,0]]),
                   format_keys('period', 'shift', 'coverage'),
                   map(str,
                       [inliers, outliers]),
                   format_keys('R2', 'MSE'),
                   map(str,
                       [max_degree, n_params])]),
          sep=sep)


def get_files(input):
    """get_files(input)

    Returns an iterable of filenames containing data to process.
    """
    if input is None:
        return stdin
    elif input[0] == "@":
        with open(input[1:], 'r') as f:
            # TODO:
            # check if mapping over `f` instead of `f.readlines()` has same
            # result
            return map(lambda x: x.strip(), f.readlines())
    elif path.isfile(input):
        return [input]
    elif path.isdir(input):
        return sorted(listdir(input))
    else:
        raise FileNotFoundError('file {} not found'.format(input))


if __name__ == "__main__":
    exit(main())
