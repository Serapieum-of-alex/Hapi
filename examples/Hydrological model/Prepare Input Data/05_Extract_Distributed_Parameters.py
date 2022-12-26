"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi
"""

from Hapi.rrm.inputs import Inputs

dem_path = "examples/Hydrological model/data/gis_data/acc4000.tif"
outputpath = "examples/Hydrological model/data/gis_data/parameters/03/"

<<<<<<< Updated upstream
Inputs.extractParameters(dem_path, "03", AsRaster=True, SaveTo=outputpath)
=======
Inputs.extractParameters(dem_path, "03", as_raster=True, save_to=outputpath)
>>>>>>> Stashed changes
