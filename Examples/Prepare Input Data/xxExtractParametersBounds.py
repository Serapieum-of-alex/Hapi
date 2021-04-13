# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:15:47 2020

@author: mofarrag
"""
# from IPython import get_ipython
# get_ipython().magic("reset -f")
# import os
# os.chdir("")
import geopandas as gpd
# import numpy as np
import pandas as pd
import Hapi.inputs as IN

# BasinF = "F:/02Case studies/Coello/base_data/GIS/delineation/features/basins.shp"
BasinF = "F:/02Case studies/Coello/base_data/GIS/GIS/BasinExtractParameters.shp"
ParametersPath = "F:/01Algorithms/HAPI/Hapi/Parameters"
SaveTo = "F:/02Case studies/Coello/Hapi/Data/00inputs/Basic_inputs"
#%%
Basin = gpd.read_file(BasinF)
# parameters name with the same order inside the Input module
ind = ["tt","sfcf","cfmax","cwh","cfr","fc","beta","lp","k0","k1","k2","uzl","perc","maxbas"]
Par = pd.DataFrame(index = ind)
# extract parameters boundaries
Par['UB'], Par['LB'] = IN.ExtractParametersBoundaries(Basin)
# extract parameters in a specific scenarion from the 10 scenarios
Par['1'] = IN.ExtractParameters(Basin,"10")
"""
zoom to the place where the catchment exist to check if the basin polygon overlay
the right location, if not there is a problem in the coordinate reference system
transformation
"""
#%% save the parameters
Par['UB'].to_csv(SaveTo + "/UB-Extracted.txt", header=None)
Par['LB'].to_csv(SaveTo + "/LB-Extracted.txt", header=None)
Par['1'].to_csv(SaveTo + "/scenario10.txt", header=None)
