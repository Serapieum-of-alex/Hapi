# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:15:47 2020

@author: mofarrag
"""
from IPython import get_ipython
get_ipython().magic("reset -f")
# import os
# os.chdir("")
import geopandas as gpd
import Hapi.inputs as IN


# BasinF = "F:/02Case studies/Coello/base_data/GIS/delineation/features/basins.shp"
BasinF = "F:/02Case studies/Coello/base_data/GIS/GIS/BasinExtractParameters.shp"
ParametersPath = "F:/01Algorithms/HAPI/Hapi/Parameters"
#%%
Basin = gpd.read_file(BasinF)
UB, LB = IN.ParametersBoundaries(Basin)

"""
zoom to the place where the catchment exist to check if the basin polygon overlay
the right location, if not there is a problem in the coordinate reference system
transformation
"""

