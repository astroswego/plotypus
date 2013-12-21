plotypus
========

A library and command line utility written in python for plotting stellar
lightcurves, performing interpolation, and principle component analysis

authors
=======

Earl Bellinger

Dan Wysocki

usage
=====

python plotypus -i /path/to/input -o /path/to/output --interpolant=trig
--interpolation-degree=8 --PCA-degree=7 --linear-model=true
--plot-lightcurves-observed --plot-lightcurves-interpolated
--plot-lightcurves-pca

external resources
==================

pcat.f by F. Murtagh used for Principle Component Analysis
