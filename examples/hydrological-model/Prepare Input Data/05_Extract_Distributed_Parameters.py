"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi
"""

from Hapi.rrm.inputs import Inputs

dem_path = "examples/hydrological-model/data/gis_data/acc4000.tif"
outputpath = "examples/hydrological-model/data/gis_data/parameters/03/"

Inputs.extractParameters(dem_path, "03", as_raster=True, save_to=outputpath)
