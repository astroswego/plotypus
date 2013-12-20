#import commands
import os
import subprocess
import sys

import numpy
from numpy import fromstring, savetxt, vstack

from scale import standardize, unstandardize
from interpolation import pca_reconstruction
import sectparse

# When and if degree can be something other than 7, swap degree with output
# output should be the last argument, and give degree no default value
def pcat(star_matrix, output, degree=7):
    assert degree == 7, \
        "PCA of degree != 7 is not yet implemented. Please use degree == 7"
    pcat_source_template_fname = "pcat_template.f"
    pcat_compile_fname = os.path.join(output, "pcat")
    pcat_source_fname = pcat_compile_fname + ".f"
    pcat_input_fname = os.path.join(output, "pcat_input.txt")
    pcat_output_fname = os.path.join(output, "pcat_output.txt")
    eigenvectors_fname = os.path.join(output, "eigenvectors.txt")
    principle_scores_fname = os.path.join(output, "principle_scores.txt")
    reconstruction_fname = os.path.join(output, "reconstruction.txt")
    root = sys.path[0]
    os.chdir(root)
    
    number_of_stars = star_matrix.shape[0]
    with open(pcat_source_template_fname, "r") as pcat_template, \
         open(pcat_source_fname, "w") as pcat_source:
        tempcat_source_text = "".join(pcat_template.readlines())
        pcat_source_text = pcat_template_text.replace("PYTHON_NUMBER_OF_STARS",
                                                      str(number_of_stars))
        pcat_source.write(pcat_source_text)
    pcat_compile = subprocess.check_output(["gfortran", pcat_source_fname,
                                            "-o", pcat_compile_fname])

    savetxt(pcat_input_fname, star_matrix)
    pcat_output = (subprocess.check_output(pcat_compile_fname).decode("utf-8")
                                                              .splitlines())
    ## DEBUG ## Saves pcat's raw output to pcat_output.txt
    savetxt(pcat_output_fname, pcat_output)
    ###########
    textiter = iter(pcat_output)
    keylist = [" CORRELATION MATRIX FOLLOWS.",
               "0EIGENVECTORS FOLLOW.",
               "0PROJECTIONS OF ROW-POINTS FOLLOW.",
               "0PROJECTIONS OF COLUMN-POINTS FOLLOW."]
    parser = sectparse.SectionParser(textiter, keylist)
    pcat_map = {heading: text for heading, text in parser}
#   Takes text parsed from pcat's output, removes the top 2 rows, which are
#   labels, parses the rest into a 2D array, and then removes the first column,
#   which is also just labels, giving the actual eigenvector and principle
#   score matrices.
    eigenvectors = parse_to_array(
        pcat_map["0EIGENVECTORS FOLLOW."],
        start_row=2, start_col=1)
    ## DEBUG ## STILL DO SAVETXTS HERE!!!!
    principle_scores = parse_to_array(
        pcat_map["0PROJECTIONS OF ROW-POINTS FOLLOW."],
        start_row=2, start_col=1)
    reconstruction_matrix = pca_reconstruction(eigenvectors, principle_scores)
    ## DEBUG ## Saves parsed output and reconstruction matrix to files
    savetxt(eigenvectors_fname, eigenvectors)
    savetxt(principle_scores_fname, principle_scores)
    savetxt(reconstruction_fname, reconstruction_matrix)
#    standardized_x, x_mean, x_std = standardize(star_matrix)
#    reconstruction_matrix = unstandardize(reconstruction_matrix, x_mean, x_std)
#   Delete temporary files
    for f in [pcat_compiled_fname, pcat_source_fname]:
        os.remove(f)
    return eigenvectors, principle_scores, reconstruction_matrix

def parse_to_array(string, dtype=numpy.float, sep=' ',
                   start_row=None, end_row=None, start_col=None, end_col=None):
    return vstack(
        fromstring(line, dtype=dtype, sep=sep) for line in
        string.splitlines()[start_row:end_row])[:,start_col:end_col]
