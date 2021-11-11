"""
Created on Sun Mar 14 18:45:56 2021

@author: mofarrag
"""
"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/Example
"""

from Hapi.inputs import Inputs

dem_path = "Data/GIS/Hapi_GIS_Data/acc4000.tif"
outputpath = "data/parameters/03/"

Inputs.ExtractParameters(dem_path, "03", AsRaster=True, SaveTo=outputpath)
