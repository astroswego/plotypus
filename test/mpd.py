import numpy
from plotypus.lightcurve import *
filename = 'data/I/OGLE-LMC-CEP-0008.dat'
ops = {'periodogram': 'conditional_entropy',
       'min_period':0.01, 'max_period':100}

data = numpy.loadtxt(filename)
lc = get_lightcurve_from_file(filename, **ops)
for i in range(0,30):
  plot_lightcurve(str(i), lc['lightcurve'], lc['period'], lc['phased_data'])
  if lc['degree'] < 1:
    break
  phased_data = lc['phased_data'].data
  residuals = phased_data.T[1] - lc['model'].predict(phased_data)
  new_data = numpy.vstack((data.T[0], residuals, data.T[2])).T
  lc = get_lightcurve(numpy.ma.array(new_data, mask=None), 
                      predictor=make_predictor(fourier_degree=(0,8)), **ops)
