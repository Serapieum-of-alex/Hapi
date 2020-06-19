# Hapi Inputs
One of the Inputs to Hapi in the Conceptual Hydrological model, HBV is one of the most used lumped conceptual hydrological model





## Parameters

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


the `inputs` module in Hapi has a `ExtractParametersBoundaries` method to extract the 

```
some code
```

Optional packages are:


### Download

	pip install Hapi


## Project layout



*Above: Overview about functionality of the Hapi package*
