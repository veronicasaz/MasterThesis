import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

from AstroLib_Plots import Plot

pl = Plot('Test', nrows = 2, ncols = 2)
pl.createPlot()

x = np.arange(0, 100,1)
y = x*2

pl.size(titleSize = 20)

pl.addSubplot('Test0', 'Test0', 1)
pl.plot(x, y, 1, type_plot = 'scatter')
pl.addSubplot('Test0', 'Test0', 2)
pl.plot(x, y, 2, type_plot = 'scatter', markers = ['*'])

pl.show()
