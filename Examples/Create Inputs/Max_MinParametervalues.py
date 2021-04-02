# -*- coding: utf-8 -*-
"""
this code creates the max and min values of the parameters from the 10 scenarios
of the HBV-SIMREG

@author: mofarrag
"""
from IPython import get_ipython
get_ipython().magic("reset -f")
# Comp = "F:/"
Comp = "F:/Users/mofarrag/Documents/"
# import os
import gdal
# import osr
# import numpy as np
# import gdalconst
from Hapi.raster import Raster as R

par = "Par_UZL"
path = Comp + "01Algorithms/HBV-SIMREG/parameters/" + par
parameters = R.ReadRastersFolder(path, WithOrder=False)
# MaxValue = parameters.max(axis=2)
# MinValue = parameters.min(axis=2)
MeanValue = parameters.mean(axis=2)

Path1 = path + "/" + par + "-1.tif"
# Saveto1 = path + "/" + par + "-Max.tif"
# Saveto2 = path + "/" + par + "-Min.tif"
Saveto3 = path + "/" + par + "-mean.tif"
src = gdal.Open(Path1)

# R.RasterLike(src,MaxValue,Saveto1,pixel_type=1)
# R.RasterLike(src,MinValue,Saveto2,pixel_type=1)
R.RasterLike(src,MeanValue,Saveto3,pixel_type=1)
