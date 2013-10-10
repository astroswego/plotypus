import re

import numpy
import statsmodels.api as sm

from utils import map_reduce, splitAtFirst

# linear_model_regex = re.compile("^(\w+\d+(:\d+)*(,\w+\d+(:\d+)*)*)$")

class LinearModelHandler:
    def __init__(self, coefficients, coeffmaps):
        self.coefficients = coefficients
        self.coeffmap = coeffmaps[0]
        if len(coeffmaps) > 1:
            for cm in coeffmaps[1:]:
                self.coeffmap.update(cm)

#        print("1 {} 2".format(coefficients))
        for coeffseq in coefficients:
#            print("coeffseq: {}".format(coeffseq))
            coeffn = self.coeffmap.get(coeffseq[0])
#            print(coeffn)
            if not coeffn:
                raise LinearModelUndefinedCoefficientError(coeffseq[0])
            degree = coeffseq[1]
            if coeffn[1] > degree or degree > coeffn[2]:
                raise LinearModelInvalidDegreeError(coeffseq[0], degree)
            
            # for coeff in coeffseq:
            #     print("coeff: " + coeff)
            #     coeffn = self.coeffmap.get(coeff[0])
            #     if not coeffn:
            #         raise LinearModelUndefinedCoefficientError(coeff[0])
            #     degree = coeff[1]
            #     if coeffn[1] > degree or degree > coeffn[2]:
            #         raise LinearModelInvalidDegreeError(coeff[0], degree)

    def evaluate_coefficient(self, coeffvector, coeff, number):
        return coeffvector[self.coeffmap[coeff][0](number)]

    def evaluate_coefficients(self, coeffvector):
        coeffvalues = (evaluate_coefficient(coeffvector, *pair)
                       for pair in self.coefficients)
        try:
            value = next(coeffvalues)
            for v in coeffvalues:
                value /= v
            return value
        except StopIteration:
            print('uh oh')

class TrigonometricHandler(LinearModelHandler):
    def __init__(self, coefficients, trig_degree, PCA_degree=0):
        trigmap = trigonometric_coefficient_map(trig_degree)
        coeffmaps = ([trigmap, principle_component_coefficient_map(PCA_degree)]
                     if PCA_degree > 0 else [trigmap])
        super().__init__(coefficients, coeffmaps)
        
def trigonometric_coefficient_map(degree):
    """Maps the names of trig interpolation coefficients to a function which,
    given the number of the coefficient, returns the index in the coefficient
    vector corresponding to that coefficient"""
    return {"A":   (lambda n: (n+1)/2 if n else 0, 0, degree),
            "Phi": (lambda n: n/2, 1, degree)}
def principle_component_coefficient_map(degree):
    """Maps the names of PCA coefficients to a function which,
    given the number of the coefficient, returns the index in the coefficient
    vector corresponding to that coefficient"""
    return {"PC": (lambda n: n+1, 1, degree)}

class LinearModelError(Exception):
    """Base class for exceptions in linearmodel.py"""
    pass
    
class LinearModelFormattingError(LinearModelError):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return "Invalid formatting of Linear Model input: " + self.value

class LinearModelInvalidCoefficientError(LinearModelError):
    def __init__(self, coefficent):
        self.coefficent = coefficent
    def __str__(self):
        return "Undefined coefficient " + self.coefficient

class LinearModelInvalidDegreeError(LinearModelError):
    def __init__(self, coefficent, degree):
        self.coefficient = coefficient
        self.degree = degree
    def __str__(self):
        return "Degree {0} is invalid for token {1}".format(degree, token)

def parseLinearModelString(s, handler, interpolation_degree, PCA_degree):
    # Checks that string matches the proper syntax
    if not re.match("""^(\w+\d+(:\w+\d+)*(,\w+\d+(:\w+\d+)*)*)$""", s):
        raise LinearModelFormattingError(s)
    tokenized = ((splitAtFirst(int, token) # Splits letters from numbers
                    for token in group.split(":"))
                 for group in s.split(","))
#    print("if you're seeing this, linear modeling won't work")
#    for t in tokenized:
#        for a in t: print(a)
#    print("tokenized: {}".format(list(tokenized)))
    handlers = (handler(coeff_sequence, interpolation_degree, PCA_degree)
                for coeff_sequence in tokenized)
    return handlers

def trigonometric_length(degree):
    return 2*degree + 1
def trigonometric_varmap(degree):
    return {"A":   lambda i: (i+1)/2 if 0<i<=degree else 0 if i==0 else None,
            "Phi": lambda i: i/2     if 1<=i<=degree else None}
def pca_varmap(degree, shift):
    return {"PC":  lambda i: shift+i - 1}

def parseLinearModelInput(s, varmap):
    if not re.match("""^(\w+\d*(:\w+\d*)*(,\w+\d*(:\w+\d*)*)*)$""", s):
        raise LinearModelFormattingError(s)
    split_input = [[splitAtFirst(int, token)
                    for token in expression.split(":")]
                   for expression in s.split(",")]
    for expression in split_input:
        for token in expression:
            (coefficient, degree) = token
            f = varmap.get(coefficient)
            if not f:
                raise LinearModelInvalidCoefficientError(coefficient)
            elif f(degree) is None:
                raise LinearModelInvalidDegreeError(coefficient, degree)
                
    linear_model_handler = makeLinearModelHandler(varmap)
    return split_input, linear_model_handler

def makeLinearModelHandler(varmap):
    def mapvar(var, coeffmatrix, valuematrix):
        return varmap[var[0]](None if len(var) == 1 else var[1])
    def f(depvar, indepvars, coeffmatrix, valuematrix):
        depvalue = mapvar(depvar, coeffmatrix, valuematrix)
        indepvalues = (mapvar(i, coeffmatrix, valuematrix) for i in indepvars)
        return linear_model(depvalue, *indepvalues)
    return f

def linear_model(dependent_variable, *independent_variables):
    """Performs multiple regression on the equation
    y = *(a_i x_i) + c
    Where y is the dependent_variable, and 
    """
    # Initializes array with all ones
    M = numpy.ones((len(dependent_variable), len(independent_variables) + 1))
    # Replaces all but the first column with the independent variables,
    # This leaves the first column as ones, which will make it act as a constant
    M[:,1:] = numpy.dstack(x for x in independent_variables)
    # Sets up the linear model
    model = sm.OLS(dependent_variable, M)
    # Performs the fit
    fit = model.fit()
    return fit

