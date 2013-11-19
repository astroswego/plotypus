#import commands
import os
import subprocess
import sys

import numpy
from numpy import fromstring, savetxt, vstack

from scale import standardize, unstandardize
from star import pca_reconstruction
import sectparse

def pcat(star_matrix, degree=7):
    root = sys.path[0]
    os.chdir(root)
    
    N = star_matrix.shape[0]
    with open("pcat_template.f", "r") as tempcat, open("pcat.f", "w") as pcat:
        tempcat_source = "".join(tempcat.readlines())
        pcat_source = tempcat_source.replace("PYTHON_NUMBER_OF_STARS", str(N))
        pcat.write(pcat_source)
    pcat_compile = subprocess.check_output(["gfortran", "-o", "pcat", "pcat.f"])
    savetxt("data", star_matrix)
    pcat_output = subprocess.check_output("./pcat").decode("utf-8").splitlines()
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
    eigenvectors = vstack(
        fromstring(line,dtype=numpy.float,sep=' ') for line in
        pcat_map["0EIGENVECTORS FOLLOW."].splitlines()[2:])[:,1:]
    principle_scores = vstack(
        fromstring(line,dtype=numpy.float,sep=' ') for line in
        pcat_map["0PROJECTIONS OF ROW-POINTS FOLLOW."].splitlines()[2:])[:,1:]
    reconstruction_matrix = pca_reconstruction(eigenvectors, principle_scores)
#    standardized_x, x_mean, x_std = standardize(star_matrix)
#    reconstruction_matrix = unstandardize(reconstruction_matrix, x_mean, x_std)
#   Delete temporary files
    for f in ["pcat.f", "pcat", "data"]:
        os.remove(f)
    return eigenvectors, principle_scores, reconstruction_matrix
