# Hapi Inputs
One of the Inputs to Hapi in the Conceptual Hydrological model, HBV is one of the most used lumped conceptual hydrological model


## Parameters

You will find the following example in the `ExtractParametersBounds.py` file under the folder `/Examples/Create Inputs`. There is no need for copy paste work.

To Extract the parameters range needed for the Calibration you have to prepare a shapefile of the catchment you are developing a distributed model and read it using `geopandas`, 

	import geopandas as gpd
	import numpy as np
	import pandas as pd
	import Hapi.inputs as IN

	BasinF = "Path to shapefile"
	Basin = gpd.read_file(BasinF)
	# parameters name with the same order inside the Input module
	ind = ["tt","sfcf","cfmax","cwh","cfr","fc","beta","lp","k0","k1","k2","uzl","perc","maxbas"]
	Par = pd.DataFrame(index = ind)

the `inputs` module in Hapi has a `ExtractParametersBoundaries` method to overlay the basin shapefile with the global parameters rasters and extract the max and min parameter values within the basin and plots your basin shapefile in top of the world map to make sure of the projection transformation from whatever projection your basin shapefile to the `WGS64` that the parameters rasters have

	# extract parameters boundaries
	Par['UB'], Par['LB'] = IN.ExtractParametersBoundaries(Basin)

To extract the parameters from one of the ten scenarios developed to derive the Global model `ExtractParameters` method takes the number of the scenario as a string and return the parameters

	# extract parameters in a specific scenarion from the 10 scenarios
	Par['1'] = IN.ExtractParameters(Basin,"01")

the extracted parameters needs to be modified incase you are not considering the snow bucket the first 5 parameters are disregarded
