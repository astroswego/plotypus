import os
import subprocess
import sys

import numpy
from numpy import fromstring, savetxt, vstack

from scale import standardize, unstandardize
from interpolation import pca_reconstruction
import sectparse

def pcat(input_matrix, output, degree=7):
    """Performs Principle Component Analysis (PCA) on input_matrix, a
    rectangular array where each row contains evenly phased observations of
    an object.  All rows must be observed at the same sequence of phases.

    This is done by taking pcat_template.f, a FORTRAN program which will not
    compile as-is, because it has the text PYTHON_NUMBER_OF_OBJECTS where the
    program requires an actual number.  This function takes the text of this
    file, and replaces PYTHON_NUMBER_OF_OBJECTS with the number of rows in
    input_matrix, and saves that to output/pcat.f.  output/pcat.f is compiled
    to output/pcat, and input_matrix is saved to output/pcat_input.txt.  Now 
    when the pcat executable is run, it will take input_matrix, perform PCA on
    it, and output a long string of output.  This output contains the various
    data obtained from the PCA -- including the eigenvectors and principle
    scores -- separated into sections with headings.  sectparse is used to
    split the output string where those sections are labeled, and return a
    mapping from section heading to the section which it labels.
    The eigenvector and principle score sections are parsed into arrays, and
    they are used to produce the reconstruction_matrix.

    The eigenvector, principle score, and reconstruction matrices are all
    returned.
    """
##!! TODO !!##
# When and if degree can be something other than 7, swap degree with output
# in the argument list. output should be the last argument, and give degree no
# default value
    assert degree == 7, \
        "PCA of degree != 7 is not yet implemented. Please let degree == 7"
#   The number of objects == the rows in the input_matrix
    number_of_objects = input_matrix.shape[0]
## Defining all paths used in pcat() ##
#   Original directory script was executed from
    original_directory = os.getcwd()
#   pcat FORTRAN source template to edit and compile
    pcat_template_path = os.path.join(sys.path[0], "pcat_template.f")
#   pcat executable file
    pcat_compiled_path = os.path.join(output, "pcat")
#   pcat FORTRAN source file -- ready to be compiled
    pcat_source_path = pcat_compiled_path + ".f"
#   pcat input matrix file
    pcat_input_path = os.path.join(output, "pcat_input.txt")
#   file to redirect pcat's STDOUT to
    pcat_output_path = os.path.join(output, "pcat_output.txt")
#   file to save eigenvector matrix to
    eigenvectors_path = os.path.join(output, "eigenvectors.txt")
#   file to save principle score matrix to
    principle_scores_path = os.path.join(output, "principle_scores.txt")
#   file to save reconstruction matrix to
    reconstruction_path = os.path.join(output, "reconstruction.txt")

#   Changing working directory to output directory so that pcat.f outputs there
    os.chdir(output)
    with open(pcat_template_path, "r") as pcat_template_file, \
         open(pcat_source_path, "w") as pcat_source_file:
#       Read the text from pcat_template.f into a string
        pcat_template_text = "\n".join(pcat_template_file.readlines())
#       Replace PYTHON_NUMBER_OF_OBJECTS and PYTHON_INPUT_FILENAME with their
#       appropriate values.
        pcat_source_text = pcat_template_text.replace(
            "PYTHON_NUMBER_OF_OBJECTS", str(number_of_objects))
        pcat_source_text = pcat_source_text.replace(
            "PYTHON_INPUT_FILENAME", pcat_input_path)
#       Save the source string to the file output/pcat.f
        pcat_source_file.write(pcat_source_text)
#   Compile output/pcat.f as output/pcat
    pcat_compile = subprocess.check_output(["gfortran", pcat_source_path,
                                            "-o", pcat_compiled_path])
#   Save pcat input matrix to file
    savetxt(pcat_input_path, input_matrix)
#   Run pcat executable and store the bytes of its output as pcat_output
    pcat_output = subprocess.check_output(pcat_compiled_path)
#   Decode the bytes of pcat's output into a string
    decoded_pcat_output = pcat_output.decode("utf-8").splitlines()
#   Change back to the original directory, now that no more work is done
#   outside of python
    os.chdir(original_directory)
    ## DEBUG ## Saves pcat's raw output to pcat_output.txt
    with open(pcat_output_path, "wb") as pcat_output_file:
        pcat_output_file.write(pcat_output)
    ###########

#   Prepare iterator and keylist for sectparse to read pcat's output
    textiter = iter(decoded_pcat_output)
    keylist = ["CORRELATION MATRIX SECTION.",
               "EIGENVALUE SECTION.",
               "EIGENVECTOR SECTION.",
               "PRINCIPLE SCORE ROW SECTION.",
               "PRINCIPLE SCORE COLUMN SECTION."]
    parser = sectparse.SectionParser(textiter, keylist)
    pcat_map = {heading: text for heading, text in parser}
#   Takes text parsed from pcat's output, removes the top 2 rows, which are
#   labels, parses the rest into a 2D array, and then removes the first column,
#   which is also just labels, giving the actual eigenvector and principle
#   score matrices.
    eigenvectors = parse_to_array(pcat_map["EIGENVECTOR SECTION."])
    principle_scores = parse_to_array(pcat_map["PRINCIPLE SCORE ROW SECTION."])
    reconstruction_matrix = pca_reconstruction(eigenvectors, principle_scores)
    ## DEBUG ## Saves parsed output and reconstruction matrix to files
    savetxt(eigenvectors_path, eigenvectors)
    savetxt(principle_scores_path, principle_scores)
    savetxt(reconstruction_path, reconstruction_matrix)
    ###########
#    standardized_x, x_mean, x_std = standardize(input_matrix)
#    reconstruction_matrix = unstandardize(reconstruction_matrix, x_mean, x_std)
#   Delete temporary files
    for f in [pcat_compiled_path, pcat_source_path]:
        os.remove(f)
    return eigenvectors, principle_scores, reconstruction_matrix

def parse_to_array(string, dtype=numpy.float, sep=' ',
                   start_row=None, end_row=None, start_col=None, end_col=None):
    return vstack(
        fromstring(line, dtype=dtype, sep=sep) for line in
        string.splitlines()[start_row:end_row])[:,start_col:end_col]
